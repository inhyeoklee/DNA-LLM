"""
dna-baseline.py

Baseline CNN model for TF-binding regression.
Uses one-hot encoding and the same regression head + augmentation
as the fine-tuning script for comparison.
Sections separated by `#%%` for notebook-like execution.
"""

#%% Imports & Config
import os, gzip, random, pickle, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json # For loading pretraining/finetuning results
from dataclasses import dataclass # Needed for dummy GPTConfig

# Define a dummy GPTConfig class at the module level for torch.load
@dataclass
class GPTConfig:
    block_size: int = 0; vocab_size: int = 0; n_layer: int = 0
    n_head: int = 0; n_embd: int = 0; dropout: float = 0.0; bias: bool = True

# Parse arguments (only need output dir and augmentation shift)
parser = argparse.ArgumentParser(description="Train Baseline CNN for TF-binding")
parser.add_argument('--output_dir', type=str, default='models',
                   help='Directory to save baseline model and plots')
parser.add_argument('--max_shift', type=int, default=10,
                    help='Maximum shift amount for sequence shifting augmentation')
parser.add_argument('--plot-only', action='store_true',
                    help='Skip training and only generate comparison plots using existing model files.')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Paths
SEQ_PATH   = "tf-binding-prediction/data/chr22_sequences.txt.gz"
SCORE_PATH = "tf-binding-prediction/data/chr22_scores.txt.gz"
# No meta path needed for baseline
MODEL_PATH = os.path.join(args.output_dir, "cnn_baseline_model.pt")
CURVES_PATH = os.path.join(args.output_dir, "training_curves_baseline.png")
RUNTIME_COMP_PATH = os.path.join(args.output_dir, "finetuning_time_comparison.png") # Overwrite existing
PERF_COMP_PATH = os.path.join(args.output_dir, "performance_comparison.png") # New plot

# Hyperparameters (kept same as fine-tuning)
BATCH_SIZE = 64
LR         = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS   = 100
PATIENCE     = 10
MOTIF_LEN    = 10  # Assumed length of binding motifs
# MAX_SHIFT is handled by args.max_shift

#%% Helper Functions

def one_hot_encode(sequence):
    """Converts a DNA sequence string to a one-hot encoded tensor (4 x SeqLen)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    encoded = torch.zeros((4, seq_len), dtype=torch.float32)
    for i, base in enumerate(sequence.upper()):
        idx = mapping.get(base, -1) # Use -1 for non-ACGT chars
        if idx != -1:
            encoded[idx, i] = 1.0
    # 'N' or other characters will result in all zeros for that position
    return encoded

def pearson(pred, tgt):
    """Pearson R over flattened predictions/targets."""
    p = pred.detach().cpu().numpy().ravel()
    t = tgt.detach().cpu().numpy().ravel()
    mask = np.isfinite(p)&np.isfinite(t)
    if mask.sum()<2: return 0.0
    # Use try-except for robustness against zero variance etc.
    try:
        r, _ = pearsonr(p[mask], t[mask])
        return r if np.isfinite(r) else 0.0
    except ValueError:
        return 0.0

#%% DataSet with Sequence Shifting Augmentation (Adapted for Raw Sequences)
class TFBindingDataset(Dataset):
    """
    Dataset for TF binding prediction.
    Accepts pre-loaded raw sequences and scores.
    Applies sequence shifting augmentation if enabled.
    Returns raw sequence string and score tensor.
    """
    def __init__(self, all_seq_strings, all_scores, indices, augment=True, max_shift=10, motif_len=MOTIF_LEN):
        self.all_seq_strings = all_seq_strings # List of sequence strings
        self.all_scores = all_scores # List of score numpy arrays
        self.indices = indices
        self.augment = augment
        self.max_shift = max_shift
        self.motif_len = motif_len

        # Determine window size from the data (assuming all are same length)
        self.window_size = len(all_scores[0]) if all_scores else 0
        # Define padding character (e.g., 'N')
        self.pad_char = 'N'

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        original_sequence = self.all_seq_strings[actual_idx]
        original_scores = self.all_scores[actual_idx] # This is already a numpy array

        if self.augment and self.max_shift > 0:
            # --- Find motifs first to determine valid shift range ---
            motifs = []
            i = 0
            while i <= self.window_size - self.motif_len:
                window = original_scores[i : i + self.motif_len]
                first_score = window[0]
                if first_score != 0 and np.all(window == first_score):
                    motifs.append({'start': i, 'score': first_score})
                    i += self.motif_len
                else:
                    i += 1

            # --- Determine shift amount based on motifs ---
            if motifs:
                min_motif_start = min(m['start'] for m in motifs)
                max_motif_end = max(m['start'] + self.motif_len for m in motifs)
                lower_bound = max(-self.max_shift, -min_motif_start)
                upper_bound = min(self.max_shift, self.window_size - max_motif_end)
                shift_amount = random.randint(lower_bound, upper_bound) if lower_bound <= upper_bound else 0
            else:
                shift_amount = random.randint(-self.max_shift, self.max_shift)

            # --- Apply the calculated shift ---
            # Initialize shifted sequence with padding chars
            shifted_sequence_list = list(self.pad_char * self.window_size)
            shifted_scores = np.zeros(self.window_size, dtype=np.float32)

            # Calculate copy ranges for sequence string
            src_start_seq = max(0, -shift_amount)
            src_end_seq = min(self.window_size, self.window_size - shift_amount)
            dst_start_seq = max(0, shift_amount)
            length_to_copy_seq = src_end_seq - src_start_seq

            # Copy sequence segment if valid
            if length_to_copy_seq > 0:
                shifted_sequence_list[dst_start_seq : dst_start_seq + length_to_copy_seq] = list(original_sequence[src_start_seq : src_end_seq])
            shifted_sequence = "".join(shifted_sequence_list)

            # Place motifs in the shifted score vector
            for motif in motifs:
                new_motif_start = motif['start'] + shift_amount
                motif_score = motif['score']
                fill_start = max(0, new_motif_start)
                fill_end = min(self.window_size, new_motif_start + self.motif_len)
                if fill_start < fill_end:
                    shifted_scores[fill_start:fill_end] = motif_score

            return shifted_sequence, torch.tensor(shifted_scores, dtype=torch.float32)

        else:
            # No augmentation or max_shift is 0, return original data
            return original_sequence, torch.tensor(original_scores, dtype=torch.float32)

#%% Regression Head (Copied from dna-llm-finetuning.py)
class ImprovedRegressionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, out_channels=hidden_dim,
            kernel_size=11, stride=1, padding=5, bias=True
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        x = self.conv1d(x)     # [batch_size, hidden_dim, seq_len]
        x = x.transpose(1, 2)  # [batch_size, seq_len, hidden_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc(x)         # [batch_size, seq_len, 1]
        return x.squeeze(-1)   # [batch_size, seq_len]

#%% Baseline CNN Model Definition
class BaselineCNN(nn.Module):
    def __init__(self, head_hidden_dim=128, head_dropout=0.2):
        super().__init__()
        # Initial Conv Layer matching head's parameters
        self.initial_conv = nn.Conv1d(
            in_channels=4, out_channels=head_hidden_dim,
            kernel_size=11, stride=1, padding=5, bias=True # Use bias here
        )
        self.activation = nn.GELU()
        # The regression head
        self.head = ImprovedRegressionHead(
            input_dim=head_hidden_dim,
            hidden_dim=head_hidden_dim,
            dropout_rate=head_dropout
        )
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
         for m in self.modules():
            if isinstance(m, nn.Conv1d):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 nn.init.xavier_uniform_(m.weight)
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                 nn.init.constant_(m.weight, 1.0)
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)


    def forward(self, x_one_hot):
        # x_one_hot shape: [batch_size, 4, seq_len]
        x = self.initial_conv(x_one_hot) # -> [batch_size, head_hidden_dim, seq_len]
        x = self.activation(x)
        # Transpose for the regression head
        x = x.transpose(1, 2) # -> [batch_size, seq_len, head_hidden_dim]
        # Pass through the regression head
        y = self.head(x) # -> [batch_size, seq_len]
        return y

#%% Training & Evaluation Function
def train():
    print(f"Training Baseline CNN model")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    print("Loading and preprocessing data...")
    try:
        all_seqs_df = pd.read_csv(SEQ_PATH, sep="\t", compression="gzip")
        all_scores_df = pd.read_csv(SCORE_PATH, sep="\t", compression="gzip")
    except FileNotFoundError:
        print(f"Error: Data files not found. Ensure '{SEQ_PATH}' and '{SCORE_PATH}' exist.")
        return None, None # Indicate failure (return only two values)

    # Extract sequences as list of strings
    all_seq_strings = all_seqs_df["sequence"].tolist()
    # Get all score vectors as list of numpy arrays
    all_scores = [ all_scores_df[c].values.astype(np.float32) for c in all_scores_df.columns ]

    # Basic check
    if len(all_seq_strings) != len(all_scores):
         print(f"Warning: Number of sequences ({len(all_seq_strings)}) does not match number of score vectors ({len(all_scores)}). Assuming order matches.")
         min_len = min(len(all_seq_strings), len(all_scores))
         all_seq_strings = all_seq_strings[:min_len]
         all_scores = all_scores[:min_len]


    # Create indices and shuffle
    indices = list(range(len(all_seq_strings)))
    random.seed(42) # Ensure consistent split
    random.shuffle(indices)

    # Split indices (80/20 split like fine-tuning)
    n = len(indices)
    v_count = int(0.2 * n)
    val_indices = indices[:v_count]
    train_indices = indices[v_count:]
    print(f"Total samples: {n}, Training (original): {len(train_indices)}, Validation: {len(val_indices)}")

    # Create Datasets: Original training, Augmented training, and Validation
    train_ds_original = TFBindingDataset(all_seq_strings, all_scores, train_indices, augment=False, max_shift=args.max_shift, motif_len=MOTIF_LEN)
    train_ds_augmented = TFBindingDataset(all_seq_strings, all_scores, train_indices, augment=True, max_shift=args.max_shift, motif_len=MOTIF_LEN)
    combined_train_ds = ConcatDataset([train_ds_original, train_ds_augmented])
    print(f"Combined training dataset size (original + augmented): {len(combined_train_ds)}")
    val_ds = TFBindingDataset(all_seq_strings, all_scores, val_indices, augment=False, max_shift=args.max_shift, motif_len=MOTIF_LEN)

    # Create DataLoaders
    train_dl = DataLoader(combined_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    # Initialize Model, Optimizer & Loss
    model = BaselineCNN().to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    optimr = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY) # Train all params
    criterion = nn.MSELoss()

    # Track metrics
    epoch_times = []
    train_losses = []
    val_losses = []
    pearson_scores = []
    epochs_ran = [] # Use this to track actual epochs run

    best_loss, no_improve = float('inf'), 0
    best_model_state = None
    best_epoch = 0
    best_pearson = 0.0

    print("Starting training loop...")
    for epoch in range(1, NUM_EPOCHS+1):
        epoch_start_time = time.time()
        epochs_ran.append(epoch) # Record epoch number

        # Training
        model.train()
        tloss=0
        for batch_idx, (seq_strings, y) in enumerate(train_dl):
            # One-hot encode the batch of sequences
            x_one_hot = torch.stack([one_hot_encode(s) for s in seq_strings]).to(device)
            y = y.to(device)

            optimr.zero_grad()
            pred = model(x_one_hot)
            loss = criterion(pred, y)
            loss.backward(); optimr.step()
            tloss += loss.item()
            # Optional: print batch progress
            # if (batch_idx + 1) % 100 == 0: print(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_dl)}")

        tloss /= len(train_dl)
        train_losses.append(tloss)

        # Validation
        model.eval()
        vloss, prs = 0, []
        with torch.no_grad():
            for seq_strings, y in val_dl:
                x_one_hot = torch.stack([one_hot_encode(s) for s in seq_strings]).to(device)
                y = y.to(device)
                pred = model(x_one_hot)
                vloss += criterion(pred,y).item()
                prs.append(pearson(pred,y))
        vloss /= len(val_dl); pr = np.mean(prs)
        val_losses.append(vloss)
        pearson_scores.append(pr)

        # Record metrics & Print
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch}: Train L={tloss:.4f}, Val L={vloss:.4f}, Val R={pr:.4f}, Time={epoch_time:.2f}s")

        # Track best model and early stopping
        if vloss < best_loss:
            best_loss, no_improve = vloss, 0
            best_epoch = epoch
            best_pearson = pr
            best_model_state = model.state_dict() # Save the best state
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val_loss={best_loss:.4f}, R={best_pearson:.4f}")
                break

    # Record total time
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total baseline training time: {elapsed:.2f}s")

    # Plot training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_ran, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Baseline CNN Loss Curves')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, pearson_scores, 'g-', label="Pearson's R")
    plt.xlabel('Epoch')
    plt.ylabel("Pearson's Correlation")
    plt.title("Baseline CNN Pearson's Correlation")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Highlight best epoch if training occurred
    if val_losses:
        best_epoch_idx = val_losses.index(min(val_losses))
        plt.axvline(x=epochs_ran[best_epoch_idx], color='r', linestyle='--', alpha=0.5,
                    label=f'Best Epoch ({epochs_ran[best_epoch_idx]})')
    plt.legend()

    plt.tight_layout()
    plt.savefig(CURVES_PATH)
    print(f"Saved baseline training curves to {CURVES_PATH}")
    plt.close() # Close plot

    # Save the best model at the end of training
    if best_model_state:
        print(f"Saving best baseline model (epoch {best_epoch})...")
        save_dict = {
            'model_state': best_model_state,
            'best_val_loss': best_loss,
            'epoch': best_epoch,
            'elapsed_time': elapsed,
            'epoch_times': epoch_times,
            'pearson_r': best_pearson,
            'training_history': {
                'epochs': epochs_ran,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'pearson_scores': pearson_scores
            }
        }
        torch.save(save_dict, MODEL_PATH)
        print(f"Saved baseline model to {MODEL_PATH}")
        return elapsed, best_pearson # Return key results for plotting
    else:
        print("No model state saved (likely no training epochs completed or improved).")
        return elapsed, 0.0 # Return 0 pearson if no model saved

#%% Plot Comparison Functions

def plot_runtime_comparison(baseline_time):
    """Loads fine-tuning times and plots comparison including baseline."""
    print("Generating runtime comparison plot...")
    # Dummy class is now defined at module level

    times = {}
    configs = ['full', 'vocab4', 'baseline']
    model_files = {
        'full': os.path.join(args.output_dir, 'gpt_binding_model_full.pt'),
        'vocab4': os.path.join(args.output_dir, 'gpt_binding_model_vocab4.pt'),
        'baseline': MODEL_PATH # Already defined
    }

    # Load existing times
    for config in ['full', 'vocab4']:
        path = model_files[config]
        if os.path.exists(path):
            try:
                data = torch.load(path, map_location='cpu')
                times[config] = data.get('elapsed_time', 0)
            except Exception as e:
                print(f"Warning: Could not load time from {path}: {e}")
                times[config] = 0
        else:
            print(f"Warning: Model file not found for {config}: {path}")
            times[config] = 0

    # Add baseline time
    times['baseline'] = baseline_time if baseline_time is not None else 0

    # Plotting
    valid_configs = [c for c in configs if c in times and times[c] > 0]
    valid_times = [times[c] for c in valid_configs]

    if not valid_times:
        print("No valid timing data found to plot runtime comparison.")
        return

    plt.figure(figsize=(10, 6))
    colors = {'full': 'steelblue', 'vocab4': 'darkorange', 'baseline': 'forestgreen'}
    bar_colors = [colors.get(c, 'gray') for c in valid_configs]
    bars = plt.bar(valid_configs, valid_times, color=bar_colors)

    plt.xlabel('Model Configuration')
    plt.ylabel('Training Time (seconds)')
    plt.title('DNA Model Training Time Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(valid_times)*0.02, # Adjust text position slightly
                 f'{valid_times[i]:.2f}s', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(RUNTIME_COMP_PATH)
    print(f"Saved runtime comparison to {RUNTIME_COMP_PATH}")
    plt.close()

def plot_performance_comparison(baseline_pearson):
    """Loads fine-tuning Pearson R and plots comparison including baseline."""
    print("Generating performance comparison plot...")
    # Dummy class is now defined at module level

    pearsons = {}
    configs = ['full', 'vocab4', 'baseline']
    model_files = {
        'full': os.path.join(args.output_dir, 'gpt_binding_model_full.pt'),
        'vocab4': os.path.join(args.output_dir, 'gpt_binding_model_vocab4.pt'),
        'baseline': MODEL_PATH
    }

    # Load existing Pearson scores
    for config in ['full', 'vocab4']:
        path = model_files[config]
        if os.path.exists(path):
            try:
                data = torch.load(path, map_location='cpu')
                pearsons[config] = data.get('pearson_r', 0.0)
            except Exception as e:
                print(f"Warning: Could not load Pearson R from {path}: {e}")
                pearsons[config] = 0.0
        else:
            print(f"Warning: Model file not found for {config}: {path}")
            pearsons[config] = 0.0

    # Add baseline Pearson
    pearsons['baseline'] = baseline_pearson if baseline_pearson is not None else 0.0

    # Plotting
    valid_configs = [c for c in configs if c in pearsons] # Keep all configs even if R=0
    valid_pearsons = [pearsons.get(c, 0.0) for c in valid_configs]

    if not valid_configs:
        print("No data found to plot performance comparison.")
        return

    plt.figure(figsize=(10, 6))
    colors = {'full': 'steelblue', 'vocab4': 'darkorange', 'baseline': 'forestgreen'}
    bar_colors = [colors.get(c, 'gray') for c in valid_configs]
    bars = plt.bar(valid_configs, valid_pearsons, color=bar_colors)

    plt.xlabel('Model Configuration')
    plt.ylabel("Best Validation Pearson's R")
    plt.title('DNA Model Performance Comparison (TF Binding)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=min(0, min(valid_pearsons)-0.05), top=max(valid_pearsons)+0.05) # Adjust y-axis limits

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, # Adjust text position
                 f'{valid_pearsons[i]:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(PERF_COMP_PATH)
    print(f"Saved performance comparison to {PERF_COMP_PATH}")
    plt.close()


#%% Main Execution
if __name__ == "__main__":
    if args.plot_only:
        print("Plotting only, skipping training...")
        # Load baseline results from saved file
        if os.path.exists(MODEL_PATH):
            try:
                baseline_data = torch.load(MODEL_PATH, map_location='cpu')
                baseline_elapsed_time = baseline_data.get('elapsed_time')
                baseline_best_pearson = baseline_data.get('pearson_r')
                print(f"Loaded baseline results: Time={baseline_elapsed_time:.2f}s, Pearson R={baseline_best_pearson:.4f}")

                # Generate comparison plots
                if baseline_elapsed_time is not None and baseline_best_pearson is not None:
                     plot_runtime_comparison(baseline_elapsed_time)
                     plot_performance_comparison(baseline_best_pearson)
                else:
                     print("Baseline result file missing required data for plotting.")

            except Exception as e:
                print(f"Error loading baseline model file {MODEL_PATH}: {e}")
                baseline_elapsed_time = None
                baseline_best_pearson = None
        else:
            print(f"Error: Baseline model file {MODEL_PATH} not found for plotting.")
            baseline_elapsed_time = None
            baseline_best_pearson = None
    else:
        # Train the baseline model
        baseline_elapsed_time, baseline_best_pearson = train()

        # Generate comparison plots if training was successful
        if baseline_elapsed_time is not None:
            plot_runtime_comparison(baseline_elapsed_time)
            plot_performance_comparison(baseline_best_pearson)
        else:
            print("Baseline training failed or did not return results, skipping comparison plots.")

    print("Baseline script finished.")
