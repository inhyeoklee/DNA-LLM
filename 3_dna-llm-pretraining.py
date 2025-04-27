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
import argparse
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(description="gLM pretraining")
parser.add_argument('--data_dir', type=str, default='data/full', help='data directory for bins and meta')
parser.add_argument('--output_dir', type=str, default='outputs', help='directory to save benchmarks')
args = parser.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
mode = os.path.basename(data_dir.rstrip('/\\'))

#%% Load dataset metadata
# Use pre-generated bins from data_prep.py
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]
print(f"Loaded dataset from {data_dir}, vocab size = {vocab_size}")

#%% Attention & Model Definitions
def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L,S = q.size(-2), k.size(-2)
    scale_factor = 1/math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    if is_causal:
        mask = torch.ones(L, S, device=q.device).tril()
        attn_bias = attn_bias.masked_fill(~mask.bool(), float("-inf"))
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
start_time = time.time()
# Lists to track losses over training
train_losses = []
val_losses = []
all_iterations = []
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
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data)-300, (64,))
    x = torch.stack([torch.from_numpy(data[i:i+300].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+301].astype(np.int64)) for i in ix]).to(device)
    return x, y

# Training
# Function to evaluate the model on train and val datasets
def evaluate_model():
    model.eval()
    losses = {'train':0.0, 'val':0.0}
    for split in ['train','val']:
        lsum=0.0
        for _ in range(100):
            Xv,Yv = get_batch(split)
            _,l,_ = model(Xv,Yv)
            lsum += l.item()
        losses[split] = lsum/100
    return losses

# Main training loop
while iter_num < max_iters:
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups: pg['lr']=lr

    model.train()
    X,Y = get_batch('train')
    logits, loss, _ = model(X,Y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if iter_num % eval_interval == 0:
        losses = evaluate_model()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        all_iterations.append(iter_num)
        print(f"Iter {iter_num}: train={losses['train']:.4f}, val={losses['val']:.4f}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {'model_args': model_args, 'model': model.state_dict(),
                    'best_val_loss': best_val_loss, 'iter': iter_num}
            ckpt_path = os.path.join(output_dir, f"ckpt_{mode}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    iter_num += 1

# Final evaluation at max_iters
losses = evaluate_model()
train_losses.append(losses['train'])
val_losses.append(losses['val'])
all_iterations.append(max_iters)
print(f"Iter {max_iters}: train={losses['train']:.4f}, val={losses['val']:.4f}")

print("Pretraining complete.")
end_time = time.time()
elapsed = end_time - start_time
print(f"Total training time: {elapsed:.2f}s")

# Plot the loss curves
plt.figure()
plt.plot(all_iterations, train_losses, label='train')
plt.plot(all_iterations, val_losses, label='val')
plt.title('Loss curves')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, f"loss_{mode}.png"))
print(f"Saved loss curve to {os.path.join(output_dir, f'loss_{mode}.png')}")
plt.show()

# Save loss data and timing information for future reference
loss_data = {
    "iterations": all_iterations,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "elapsed_time": elapsed,
    "vocab_size": vocab_size
}
with open(os.path.join(output_dir, f"losses_{mode}.json"), "w") as f:
    json.dump(loss_data, f)
print(f"Saved loss data to {os.path.join(output_dir, f'losses_{mode}.json')}")

# Write summary
summary = {
    "elapsed_time": elapsed,
    "best_val_loss": best_val_loss,
    "checkpoint": ckpt_path,
    "loss_curve": os.path.join(output_dir, f"loss_{mode}.png"),
    "loss_data": os.path.join(output_dir, f"losses_{mode}.json")
}
with open(os.path.join(output_dir, f"summary_{mode}.json"), "w") as f:
    json.dump(summary, f)
print(f"Saved summary to {os.path.join(output_dir, f'summary_{mode}.json')}")

#%% Plot Training Time Comparison
# If we've just finished training the vocab4 dataset, also generate a training time comparison plot
if mode == 'vocab4':
    try:
        # Try to load loss data files for both configurations
        loss_data_files = {}
        for config in ['full', 'vocab4']:
            loss_data_path = os.path.join(output_dir, f"losses_{config}.json")
            if os.path.exists(loss_data_path):
                with open(loss_data_path, 'r') as f:
                    loss_data_files[config] = json.load(f)
        
        # If we have both loss data files, create a comparison plot
        if len(loss_data_files) == 2:
            plt.figure(figsize=(10, 6))
            configs = list(loss_data_files.keys())
            times = [loss_data_files[config]['elapsed_time'] for config in configs]
            vocab_sizes = [loss_data_files[config]['vocab_size'] for config in configs]
            
            # Create bar chart
            bars = plt.bar(configs, times, color=['steelblue', 'darkorange'])
            
            # Add labels and title
            plt.xlabel('Configuration')
            plt.ylabel('Training Time (seconds)')
            plt.title('DNA-LLM Pretraining Time Comparison')
            
            # Add exact training time as text on each bar
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{times[i]:.2f}s\nVocab Size: {vocab_sizes[i]}',
                        ha='center', va='bottom')
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save and show the plot
            plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
            print(f"Saved training time comparison to {os.path.join(output_dir, 'training_time_comparison.png')}")
            plt.show()
    except Exception as e:
        print(f"Could not generate training time comparison: {e}")

# %%
