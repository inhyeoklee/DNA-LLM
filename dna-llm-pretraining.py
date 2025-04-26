"""
dna-llm-pretraining.py

Pretraining script for a genomic language model (gLM) on the GRCh38 human reference genome.
Sections are separated by `#%%` markers for notebook-like execution.
"""

#%% Imports
import os
import pickle
import math
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext

#%% Download and Prepare the Genome Dataset
# Download GRCh38 genome if not already present
os.makedirs("data", exist_ok=True)
fasta_file = os.path.join("data", "genome.fa")
if not os.path.isfile(fasta_file):
    print("Downloading GRCh38 reference genome...")
    os.system(f"wget -O - https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz | gunzip -c > {fasta_file}")

# Read genome into memory
with open(fasta_file, "r") as f:
    raw_text = f.read()

# Build vocabulary and encoder/decoder
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s): return [stoi[c] for c in s]
def decode(l): return "".join([itos[i] for i in l])

# Save metadata for later fine-tuning
meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open("data/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

#%% Tokenization: Split, Encode, and Save as Binary
# 90/10 train/val split
n = len(raw_text)
train_text = raw_text[:int(n*0.9)]
val_text   = raw_text[int(n*0.9):]
del raw_text

# Helper to append tokens to a binary file
def append_to_bin(path, arr):
    with open(path, "ab") as f:
        arr.tofile(f)

# Create empty files
for split in ["train","val"]:
    open(f"data/{split}.bin","wb").close()

# Encode in chunks to avoid RAM blowup
for i in np.arange(0,1,0.05):
    start_t = int(len(train_text)*i)
    end_t   = int(len(train_text)*(i+0.05))
    arr_t = np.array(encode(train_text[start_t:end_t]), dtype=np.uint16)
    append_to_bin("data/train.bin", arr_t)

    start_v = int(len(val_text)*i)
    end_v   = int(len(val_text)*(i+0.05))
    arr_v = np.array(encode(val_text[start_v:end_v]), dtype=np.uint16)
    append_to_bin("data/val.bin", arr_v)

print("Tokenization complete.")

#%% Attention & Model Definitions
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L,S = q.size(-2), k.size(-2)
    scale_factor = 1/math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    if is_causal:
        mask = torch.ones(L, S, device=q.device).tril()
        attn_bias = attn_bias.masked_fill(~mask, float("-inf"))
    if attn_mask is not None:
        attn_bias += attn_mask
    weights = (q @ k.transpose(-2,-1)) * scale_factor + attn_bias
    weights = torch.softmax(weights, dim=-1)
    weights = torch.dropout(weights, dropout_p, train=True)
    return weights, weights @ v

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn       = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        self.c_proj       = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout= nn.Dropout(config.dropout)
        self.n_head       = config.n_head
        self.n_embd       = config.n_embd
        self.dropout      = config.dropout

    def forward(self, x):
        B,T,C = x.size()
        q,k,v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        attn_w, out = scaled_dot_product_attention(q,k,v, is_causal=True, dropout_p=self.dropout)
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.resid_dropout(self.c_proj(out))
        return out, torch.mean(attn_w, dim=0)

class MLP(nn.Module):
    """Feedforward network in Transformer block."""
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x); x = self.gelu(x)
        x = self.c_proj(x); x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → LayerNorm → MLP."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        res, w = self.attn(self.ln_1(x))
        x = x + res
        x2,_ = self.attn(self.ln_2(x))
        x = x + self.mlp(self.ln_2(x))
        return x, w

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size : int = 50304
    n_layer    : int = 12
    n_head     : int = 12
    n_embd     : int = 768
    dropout    : float = 0.0
    bias       : bool  = True

class GPT(nn.Module):
    """GPT model for language modeling."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias)
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for n,p in self.named_parameters():
            if n.endswith('c_proj.weight'):
                nn.init.normal_(p, 0.0, 0.02/math.sqrt(2*config.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        b,t = idx.size()
        pos = torch.arange(t, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        attn_weights = []
        for block in self.transformer.h:
            x, w = block(x)
            attn_weights.append(w)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss, attn_weights

#%% Training Loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float16' if device=='cuda' else 'float32'
print(f"Using {device}, dtype={dtype}")
model_args = dict(n_layer=4, n_head=4, n_embd=384, block_size=300,
                  bias=False, vocab_size=vocab_size, dropout=0.2)
config = GPTConfig(**model_args)
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
max_iters = 4000
eval_interval = 1000
best_val_loss = float('inf')
iter_num = 0

# Learning rate schedule
def get_lr(it):
    if it < 100: return 1e-3*it/100
    if it > max_iters: return 1e-4
    decay = 0.5*(1+math.cos(math.pi*(it-100)/(max_iters-100)))
    return 1e-4 + decay*(1e-3-1e-4)

# Fetch batch
def get_batch(split):
    data = np.memmap(f"data/{split}.bin", dtype=np.uint16, mode='r')
    ix = torch.randint(len(data)-300, (64,))
    x = torch.stack([torch.from_numpy(data[i:i+300].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+301].astype(np.int64)) for i in ix]).to(device)
    return x, y

# Training
while iter_num < max_iters:
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups: pg['lr']=lr

    model.train()
    X,Y = get_batch('train')
    logits, loss, _ = model(X,Y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if iter_num % eval_interval == 0:
        model.eval()
        losses = {'train':0.0, 'val':0.0}
        for split in ['train','val']:
            lsum=0.0
            for _ in range(100):
                Xv,Yv = get_batch(split)
                _,l,_ = model(Xv,Yv)
                lsum += l.item()
            losses[split] = lsum/100
        print(f"Iter {iter_num}: train={losses['train']:.4f}, val={losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {'model_args': model_args, 'model': model.state_dict(),
                    'best_val_loss': best_val_loss, 'iter': iter_num}
            torch.save(ckpt, "out/ckpt.pt")
            print("Saved checkpoint")
    iter_num += 1

print("Pretraining complete.")
