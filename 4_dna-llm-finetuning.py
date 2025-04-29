"""
dna-llm-finetuning.py

Fine-tuning script for TF-binding regression using a pretrained genomic GPT.
Freezes transformer weights and trains only a regression head.
Supports on-the-fly reverse-complement augmentation.
Sections separated by `#%%` for notebook-like execution.
"""

#%% Imports & Config
import os, gzip, random, pickle, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset # Added ConcatDataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Parse arguments
parser = argparse.ArgumentParser(description="Fine-tune GPT for TF-binding")
parser.add_argument('--config', type=str, default='full', choices=['full', 'vocab4'],
                   help='Configuration to use (full or vocab4)')
parser.add_argument('--checkpoint_dir', type=str, default='outputs',
                   help='Directory containing pretrained checkpoints')
parser.add_argument('--output_dir', type=str, default='models',
                   help='Directory to save fine-tuned models')
parser.add_argument('--max_shift', type=int, default=10,
                    help='Maximum shift amount for sequence shifting augmentation')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Paths
SEQ_PATH   = "tf-binding-prediction/data/chr22_sequences.txt.gz"
SCORE_PATH = "tf-binding-prediction/data/chr22_scores.txt.gz"
META_PATH  = f"data/{args.config}/meta.pkl"  # Use appropriate meta file based on config
CKPT_PATH  = os.path.join(args.checkpoint_dir, f"ckpt_{args.config}.pt")
MODEL_PATH = os.path.join(args.output_dir, f"gpt_binding_model_{args.config}.pt")

# Hyperparameters
BATCH_SIZE = 64
LR         = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS   = 100
PATIENCE     = 10
# MAX_SHIFT is now handled by args.max_shift
MOTIF_LEN    = 10  # Assumed length of binding motifs

#%% DataSet with Sequence Shifting Augmentation
class TFBindingDataset(Dataset):
    """
    Dataset for TF binding prediction.
    Accepts pre-loaded sequences (token IDs) and scores.
    Applies sequence shifting augmentation if enabled.
    Assumes motifs are contiguous blocks of MOTIF_LEN identical non-zero scores.
    """
    def __init__(self, all_ids, all_scores, stoi, indices, augment=True, max_shift=10, motif_len=MOTIF_LEN): # Default max_shift here for clarity
        self.all_ids = all_ids
        self.all_scores = all_scores
        self.stoi = stoi
        self.indices = indices
        self.augment = augment
        self.max_shift = max_shift # Use the passed max_shift
        self.motif_len = motif_len
        
        # Determine window size from the data (assuming all are same length)
        self.window_size = len(all_scores[0]) if all_scores else 0
        # Define padding token ID (use index for 'N' if available, else 0)
        self.pad_token_id = stoi.get('N', 0) 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        original_ids_list = self.all_ids[actual_idx]
        original_scores = self.all_scores[actual_idx] # This is already a numpy array

        if self.augment and self.max_shift > 0: # Only augment if max_shift > 0
            #  Find motifs first to determine valid shift range 
            motifs = []
            i = 0
            while i <= self.window_size - self.motif_len:
                window = original_scores[i : i + self.motif_len]
                first_score = window[0]
                # Check for non-zero and identical scores in the motif window
                if first_score != 0 and np.all(window == first_score):
                    motifs.append({'start': i, 'score': first_score})
                    i += self.motif_len # Skip the rest of the identified motif
                else:
                    i += 1

            #  Determine shift amount based on motifs 
            if motifs:
                min_motif_start = min(m['start'] for m in motifs)
                # Calculate end position (exclusive) for the rightmost motif
                max_motif_end = max(m['start'] + self.motif_len for m in motifs) 

                lower_bound = max(-self.max_shift, -min_motif_start)
                # Ensure upper bound calculation prevents motif end from exceeding window size
                upper_bound = min(self.max_shift, self.window_size - max_motif_end) 

                if lower_bound <= upper_bound:
                    # Valid shift range exists
                    shift_amount = random.randint(lower_bound, upper_bound)
                else:
                    # No valid shift possible without cutting off motifs
                    shift_amount = 0 
            else:
                # No motifs found, allow full shift range as before
                shift_amount = random.randint(-self.max_shift, self.max_shift)

            #  Apply the calculated shift 
            # Initialize shifted tensors
            shifted_ids = np.full(self.window_size, self.pad_token_id, dtype=np.int64)
            shifted_scores = np.zeros(self.window_size, dtype=np.float32)

            # Calculate copy ranges for sequence using the determined shift_amount
            src_start = max(0, -shift_amount)
            src_end = min(self.window_size, self.window_size - shift_amount)
            dst_start = max(0, shift_amount)
            dst_end = min(self.window_size, self.window_size + shift_amount)
            length_to_copy = src_end - src_start

            # Copy sequence segment
            if length_to_copy > 0:
                 shifted_ids[dst_start : dst_start + length_to_copy] = original_ids_list[src_start : src_start + length_to_copy]

            # Place motifs in the shifted score vector (use motifs found earlier)
            for motif in motifs:
                new_motif_start = motif['start'] + shift_amount
                motif_score = motif['score']
                
                # Calculate the valid range to fill within the shifted_scores vector
                fill_start = max(0, new_motif_start)
                fill_end = min(self.window_size, new_motif_start + self.motif_len)
                
                # Fill the scores if the motif is at least partially within bounds
                if fill_start < fill_end:
                    shifted_scores[fill_start:fill_end] = motif_score
            
            return torch.tensor(shifted_ids, dtype=torch.long), torch.tensor(shifted_scores, dtype=torch.float32)
            
        else:
            # No augmentation or max_shift is 0, return original data
            return torch.tensor(original_ids_list, dtype=torch.long), torch.tensor(original_scores, dtype=torch.float32)


#%% Load Vocabulary & Pretrained GPT
# Load token mapping
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
stoi = meta["stoi"]

# Import transformers (copy from pretraining)
import math, torch.nn.functional as F
from dataclasses import dataclass

def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L,S = q.size(-2), k.size(-2)
    scale_factor = 1/math.sqrt(q.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=q.dtype, device=q.device)
    if is_causal:
        # Create causal mask without using bool() for TorchScript compatibility
        mask = torch.tril(torch.ones(L, S, device=q.device))
        mask_inverse = 1.0 - mask  # inverse the mask (0 where tril is 1, 1 where tril is 0)
        attn_bias = attn_bias.masked_fill(mask_inverse > 0, float("-inf"))
    if attn_mask is not None:
        attn_bias += attn_mask
    weights = (q @ k.transpose(-2,-1)) * scale_factor + attn_bias
    weights = torch.softmax(weights, dim=-1)
    weights = torch.dropout(weights, dropout_p, train=True)
    return weights, weights @ v

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias): 
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn       = nn.Linear(cfg.n_embd, 3*cfg.n_embd, bias=cfg.bias)
        self.c_proj       = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout= nn.Dropout(cfg.dropout)
        self.n_head       = cfg.n_head
        self.n_embd       = cfg.n_embd
        self.dropout      = cfg.dropout

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
    def __init__(self, cfg):
        super().__init__()
        self.c_fc    = nn.Linear(cfg.n_embd, 4*cfg.n_embd, bias=cfg.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4*cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        x = self.c_fc(x); x = self.gelu(x)
        x = self.c_proj(x); x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block: LayerNorm → Attention → LayerNorm → MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x):
        res, w = self.attn(self.ln_1(x))
        x = x + res
        x = x + self.mlp(self.ln_2(x))
        return x, w

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
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(cfg.vocab_size, cfg.n_embd),
            'wpe': nn.Embedding(cfg.block_size, cfg.n_embd),
            'drop': nn.Dropout(cfg.dropout),
            'h': nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            'ln_f': LayerNorm(cfg.n_embd, cfg.bias)
        })
        # We need this to match the checkpoint structure, but we won't use it for finetuning
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Share weights with the embedding layer
        self.lm_head.weight = self.transformer.wte.weight
        
    def forward(self, idx):
        b,t = idx.size()
        pos = torch.arange(t, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        # Each block returns (x, attn_weights) but we only need x for finetuning
        for block in self.transformer.h:
            x, _ = block(x)
        x = self.transformer.ln_f(x)
        return x  # hidden states

#%% Improved Regression Head with 1D CNN
class ImprovedRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 1D Convolution with parameters:
        # - kernel size of 11 (slightly wider than binding motif)
        # - padding of 5 (maintains sequence length)
        # - stride of 1 (check every position)
        # - 128 filters (capture diverse binding patterns)
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=11,
            stride=1,
            padding=5,
            bias=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.layer_norm(x)
        
        # Transpose for 1D convolution (channels first)
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        x = self.conv1d(x)     # [batch_size, hidden_dim, seq_len]
        
        # Transpose back
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        
        # Apply activation then dropout
        x = self.activation(x)
        x = self.dropout(x)
        
        # Final linear layer
        x = self.fc(x)         # [batch_size, seq_len, 1]
        
        return x.squeeze(-1)   # [batch_size, seq_len]

#%% Regression Wrapper
class GPTRegressor(nn.Module):
    """Wraps GPT, freezes its weights, adds improved regression head."""
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters(): p.requires_grad = False
        self.head = ImprovedRegressionHead(input_dim=self.gpt.cfg.n_embd, hidden_dim=128, dropout_rate=0.2)
    
    def forward(self, idx):
        x = self.gpt(idx)                # (B, T, C)
        y = self.head(x)                 # (B, T)
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
    print(f"Fine-tuning using {args.config} configuration")
    print(f"Loading checkpoint from {CKPT_PATH}")
    
    # Start timing
    start_time = time.time()
    
    # Check if checkpoint exists
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint file {CKPT_PATH} not found!")
        return
        
    # Load pretrained GPT checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = GPTConfig(**ckpt['model_args'])
    gpt = GPT(cfg).to(device)
    gpt.load_state_dict(ckpt['model'])
    model = GPTRegressor(gpt).to(device)

    #  Data Loading and Splitting 
    print("Loading and preprocessing data...")
    # Load all data first
    all_seqs = pd.read_csv(SEQ_PATH, sep="\t", compression="gzip")["sequence"]
    all_scores_df = pd.read_csv(SCORE_PATH, sep="\t", compression="gzip")
    # Tokenize all sequences
    all_ids = [ [stoi.get(b, 0) for b in s.upper()] for s in all_seqs ]
    # Get all score vectors as numpy arrays
    all_scores = [ all_scores_df[c].values.astype(np.float32) for c in all_scores_df.columns ]
    
    # Create indices and shuffle
    indices = list(range(len(all_ids)))
    random.shuffle(indices) # Shuffle once for consistent train/val split
    
    # Split indices
    n = len(indices)
    v_count = int(0.2 * n)
    val_indices = indices[:v_count]
    train_indices = indices[v_count:]
    print(f"Total samples: {n}, Training (original): {len(train_indices)}, Validation: {len(val_indices)}")

    # Create Datasets: Original training, Augmented training, and Validation
    train_ds_original = TFBindingDataset(all_ids, all_scores, stoi, train_indices, augment=False, max_shift=args.max_shift, motif_len=MOTIF_LEN)
    train_ds_augmented = TFBindingDataset(all_ids, all_scores, stoi, train_indices, augment=True, max_shift=args.max_shift, motif_len=MOTIF_LEN)
    
    # Combine original and augmented training datasets
    combined_train_ds = ConcatDataset([train_ds_original, train_ds_augmented])
    print(f"Combined training dataset size (original + augmented): {len(combined_train_ds)}")
    
    # Validation dataset (no augmentation)
    val_ds = TFBindingDataset(all_ids, all_scores, stoi, val_indices, augment=False, max_shift=args.max_shift, motif_len=MOTIF_LEN) 

    # Create DataLoaders
    train_dl = DataLoader(combined_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # Use combined dataset
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Optimizer & Loss
    optimr = optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    # Track metrics
    epoch_times = []
    train_losses = []
    val_losses = []
    pearson_scores = []
    epochs = []
    
    best_loss, no_improve = float('inf'), 0
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_start_time = time.time()
        
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
        vloss /= len(val_dl); pr = np.mean(prs)

        # Record metrics
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        train_losses.append(tloss)
        val_losses.append(vloss)
        pearson_scores.append(pr)
        epochs.append(epoch)
        print(f"Epoch {epoch}: Train L={tloss:.4f}, Val L={vloss:.4f}, Val R={pr:.4f}, Time={epoch_time:.2f}s")

        # Track best model and early stopping
        if vloss < best_loss:
            best_loss, no_improve = vloss, 0
            best_epoch = epoch
            best_pearson = pr
            # Keep track of best model state
            best_model_state = {
                'regressor_state': model.head.state_dict(),
                'epoch': epoch,
                'val_loss': vloss,
                'pearson_r': pr
            }
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss={best_loss:.4f}, R={best_pearson:.4f}")
                break
    
    # Record total time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total fine-tuning time: {elapsed:.2f}s")
    
    # Plot training curves
    plot_path = os.path.join(args.output_dir, f"training_curves_{args.config}.png")
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot 2: Pearson's Correlation
    plt.subplot(1, 2, 2)
    plt.plot(epochs, pearson_scores, 'g-', label="Pearson's R")
    plt.xlabel('Epoch')
    plt.ylabel("Pearson's Correlation")
    plt.title("Pearson's Correlation over Training")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight best epoch
    best_epoch_idx = val_losses.index(min(val_losses))
    plt.axvline(x=epochs[best_epoch_idx], color='r', linestyle='--', alpha=0.5, 
                label=f'Best Epoch ({epochs[best_epoch_idx]})')
    plt.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved training curves to {plot_path}")
    
    # Save the best model at the end of training
    print(f"Saving best model (epoch {best_epoch})...")
    save_dict = {
        'regressor_state': best_model_state['regressor_state'],
        'gpt_state': model.gpt.state_dict(),
        'config': model.gpt.cfg,
        'best_val_loss': best_loss,
        'epoch': best_epoch,
        'elapsed_time': elapsed,
        'epoch_times': epoch_times,
        'vocab_size': cfg.vocab_size,
        'pearson_r': best_pearson,
        'training_history': {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'pearson_scores': pearson_scores
        }
    }
    torch.save(save_dict, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    
    return elapsed, epoch_times, plot_path

#%% Plot Fine-tuning Time Comparison
def plot_finetuning_time_comparison():
    try:
        # Check for both fine-tuned models
        models = {}
        for config in ['full', 'vocab4']:
            model_path = os.path.join('models', f"gpt_binding_model_{config}.pt")
            if os.path.exists(model_path):
                models[config] = torch.load(model_path, map_location='cpu')
        
        # If both models are available, create a comparison plot
        if len(models) == 2:
            plt.figure(figsize=(10, 6))
            configs = list(models.keys())
            times = [models[config]['elapsed_time'] for config in configs]
            vocab_sizes = [models[config]['vocab_size'] for config in configs]
            
            # Create bar chart with same style as pretraining plot
            bars = plt.bar(configs, times, color=['steelblue', 'darkorange'])
            
            # Add labels and title
            plt.xlabel('Configuration')
            plt.ylabel('Fine-tuning Time (seconds)')
            plt.title('DNA-LLM Fine-tuning Time Comparison')
            
            # Add exact training time as text on each bar
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                        f'{times[i]:.2f}s\nVocab Size: {vocab_sizes[i]}',
                        ha='center', va='bottom')
            
            # Add grid for better readability
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Calculate speedup
            speedup = times[0] / times[1]
            plt.figtext(0.5, 0.01, f"Fine-tuning with vocab4 is {speedup:.2f}x faster than with full.", 
                       ha='center', fontsize=12)
            
            # Save and show the plot
            plot_path = os.path.join('models', 'finetuning_time_comparison.png')
            plt.savefig(plot_path)
            print(f"Saved fine-tuning time comparison to {plot_path}")
            print(f"Fine-tuning with vocab4 is {speedup:.2f}x faster than with full.")
            
            return plot_path
        else:
            if len(models) == 0:
                print("No fine-tuned models found.")
            else:
                print(f"Only found fine-tuned model for {list(models.keys())[0]}. Need both configurations for comparison.")
            return None
    except Exception as e:
        print(f"Could not generate fine-tuning time comparison: {e}")
        return None

if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train the model and record time
    train()
    print("Fine-tuning complete.")
    
    # Try to generate comparison plot
    plot_finetuning_time_comparison()
