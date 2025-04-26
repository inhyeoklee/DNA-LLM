"""
dna-llm-finetuning.py

Fine-tuning script for TF-binding regression using a pretrained genomic GPT.
Freezes transformer weights and trains only a regression head.
Supports on-the-fly reverse-complement augmentation.
Sections separated by `#%%` for notebook-like execution.
"""

#%% Imports & Config
import os, gzip, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr

# Paths
SEQ_PATH   = "tf-binding-prediction/data/chr22_sequences.txt.gz"
SCORE_PATH = "tf-binding-prediction/data/chr22_scores.txt.gz"
META_PATH  = "data/meta.pkl"           # from pretraining step

# Hyperparameters
BATCH_SIZE = 64
LR         = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS   = 100
PATIENCE     = 10

#%% DataSet with Reverse-Complement Augmentation
class TFBindingDataset(Dataset):
    """Loads sequences and per-base scores; applies RC augmentation randomly."""
    def __init__(self, seq_path, score_path, stoi, augment=True):
        # Load data
        self.seqs = pd.read_csv(seq_path, sep="\t", compression="gzip")["sequence"]
        scores_df = pd.read_csv(score_path, sep="\t", compression="gzip")
        # Tokenize
        self.ids = [ [stoi.get(b, 0) for b in s.upper()] for s in self.seqs ]
        # Align score vectors (columns → windows)
        self.scores = [ scores_df[c].values.astype(np.float32) for c in scores_df.columns ]
        self.augment = augment
        # Build complement map for token IDs
        comp = { stoi.get("A"): stoi.get("T"),
                 stoi.get("T"): stoi.get("A"),
                 stoi.get("C"): stoi.get("G"),
                 stoi.get("G"): stoi.get("C") }
        self.comp_map = { i: comp.get(i, i) for i in range(len(stoi)) }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = list(self.ids[idx])
        scores = self.scores[idx].copy()
        # 50% RC augmentation on training
        if self.augment and random.random() < 0.5:
            ids = [self.comp_map[i] for i in ids[::-1]]
            scores = scores[::-1]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)

#%% Load Vocabulary & Pretrained GPT
# Load token mapping
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
stoi = meta["stoi"]

# Import transformers (copy from pretraining)
import math, torch.nn.functional as F
from dataclasses import dataclass

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, causal=False, scale=None):
    L,S = q.size(-2), k.size(-2)
    sf = (1/math.sqrt(q.size(-1))) if scale is None else scale
    bias = torch.zeros(L,S,device=q.device)
    if causal:
        mask = torch.ones(L,S,device=q.device).tril()
        bias = bias.masked_fill(~mask, float("-inf"))
    w = (q @ k.transpose(-2,-1))*sf + bias
    w = torch.softmax(w, dim=-1)
    w = torch.dropout(w, dropout_p, train=True)
    return w, w @ v

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias): 
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        nh, ne = cfg.n_head, cfg.n_embd
        self.c_attn = nn.Linear(ne, 3*ne, bias=cfg.bias)
        self.c_proj = nn.Linear(ne, ne,   bias=cfg.bias)
        self.n_head, self.dropout = nh, cfg.dropout
    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(C, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        w, out = scaled_dot_product_attention(q,k,v, causal=True, dropout_p=self.dropout)
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.c_proj(out), w

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1    = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.act    = nn.GELU()
        self.fc2    = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop   = nn.Dropout(cfg.dropout)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn= CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = MLP(cfg)
    def forward(self, x):
        a,_ = self.attn(self.ln1(x)); x = x + a
        m   = self.mlp(self.ln2(x));   x = x + m
        return x

@dataclass
class GPTConfig:
    block_size:int; vocab_size:int
    n_layer:int; n_head:int; n_embd:int
    dropout:float; bias:bool

class GPT(nn.Module):
    """GPT core from pretraining."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop= nn.Dropout(cfg.dropout)
        self.h   = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f= LayerNorm(cfg.n_embd, cfg.bias)
    def forward(self, idx):
        b,t = idx.size()
        pos = torch.arange(t,device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x  # hidden states

#%% Regression Wrapper
class GPTRegressor(nn.Module):
    """Wraps GPT, freezes its weights, adds per-position regression head."""
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters(): p.requires_grad = False
        self.head = nn.Linear(self.gpt.cfg.n_embd, 1)
    def forward(self, idx):
        x = self.gpt(idx)                # (B, T, C)
        y = self.head(x).squeeze(-1)     # (B, T)
        return y

#%% Training & Evaluation
def pearson(pred, tgt):
    """Pearson R over flattened predictions/targets."""
    p = pred.detach().cpu().numpy().ravel()
    t = tgt.detach().cpu().numpy().ravel()
    mask = np.isfinite(p)&np.isfinite(t)
    if mask.sum()<2: return 0.0
    return pearsonr(p[mask], t[mask])[0]

def train():
    # Load pretrained GPT checkpoint
    ckpt = torch.load("out/ckpt.pt", map_location=device)
    cfg = GPTConfig(**ckpt['model_args'])
    gpt = GPT(cfg).to(device)
    gpt.load_state_dict(ckpt['model'])
    model = GPTRegressor(gpt).to(device)

    # Data
    ds = TFBindingDataset(SEQ_PATH, SCORE_PATH, stoi, augment=True)
    n = len(ds)
    v = int(0.2*n); t = n - v
    train_ds, val_ds = random_split(ds, [t,v])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Optimizer & Loss
    optimr = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_loss, no_improve = float('inf'), 0
    for epoch in range(1, NUM_EPOCHS+1):
        # Training
        model.train()
        tloss=0
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            optimr.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward(); optimr.step()
            tloss += loss.item()
        tloss /= len(train_dl)

        # Validation
        model.eval()
        vloss, prs = 0, []
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                pred= model(x)
                vloss += criterion(pred,y).item()
                prs.append(pearson(pred,y))
        vloss /= len(val_dl); pr   = np.mean(prs)

        print(f"Epoch {epoch}: Train L={tloss:.4f}, Val L={vloss:.4f}, Val R={pr:.4f}")

        # Early stopping & save
        if vloss < best_loss:
            best_loss, no_improve = vloss, 0
            print("Saving best model...")
            torch.jit.save(torch.jit.script(model), "tf-binding-prediction/script/gpt_binding_model.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping.")
                break

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train()
    print("Fine-tuning complete.")
