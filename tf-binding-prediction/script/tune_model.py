import os
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr
import wandb

# Wandb Sweep Configuration
sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.001, 0.0001] # The 3 values to test
        },
        # Fixed parameters
        'num_channels': {
            'value': 32
        },
        'num_res_blocks': {
            'value': 3
        },
        'dropout_rate': {
            'value': 0.2
        },
        'kernel_size': {
            'value': 3
        },
        'weight_decay': {
            'value': 1e-5
        },
        'batch_size': {
            'value': 64
        },
        'num_epochs': {
            'value': 9
        },
         'optimizer_type': {
             'value': 'AdamW'
         },
         'early_stopping_patience': {
             'value': 10
         },
         'validation_split': {
             'value': 0.2
         }
    }
}

# Data Configuration
DATA_DIR = '/Users/ihlee/Desktop/DL-Genomics/tf-binding-prediction/data'
SEQ_FILE = os.path.join(DATA_DIR, 'chr22_sequences.txt.gz')
SCORE_FILE = os.path.join(DATA_DIR, 'chr22_scores.txt.gz')

# Helper Functions

def one_hot_encode(sequence):
    """Converts a DNA sequence string to a one-hot encoded tensor."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_len = len(sequence)
    encoded = torch.zeros((4, seq_len), dtype=torch.float32)
    for i, base in enumerate(sequence.upper()):
        idx = mapping.get(base, -1)
        if idx != -1:
            encoded[idx, i] = 1.0
    return encoded

def calculate_pearsonr(preds, targets):
    """Calculates Pearson correlation, handling potential NaNs."""
    # Flatten tensors and convert to numpy
    preds_flat = preds.detach().cpu().numpy().flatten()
    targets_flat = targets.detach().cpu().numpy().flatten()

    # Remove NaNs or Infs if they somehow occur
    valid_indices = np.isfinite(preds_flat) & np.isfinite(targets_flat)
    preds_flat = preds_flat[valid_indices]
    targets_flat = targets_flat[valid_indices]

    if len(preds_flat) < 2: # Need at least 2 points for correlation
        return 0.0

    try:
        r, _ = pearsonr(preds_flat, targets_flat)
        return r if np.isfinite(r) else 0.0 # Return 0 if correlation is NaN/Inf
    except ValueError:
        return 0.0

# Data Loading and Processing

class TFBindiingDataset(Dataset):
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores
        # Pre-encode sequences if memory allows, otherwise encode on-the-fly in __getitem__
        self.encoded_sequences = [one_hot_encode(seq) for seq in sequences['sequence']]
        # Convert scores DataFrame columns to a list of tensors
        self.score_tensors = [torch.tensor(scores[col].values, dtype=torch.float32) for col in scores.columns]


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Assuming sequence names in scores match sequence order/names
        window_name = self.sequences.iloc[idx]['window_name']
        # Find the corresponding score tensor - Assuming order matches
        if idx < len(self.score_tensors):
             score_vec = self.score_tensors[idx]
        else:
             raise IndexError(f"Score index {idx} out of bounds for window {window_name}")

        return self.encoded_sequences[idx], score_vec

# Model Architecture

class ResidualBlock(nn.Module):
    """Residual Block with LayerNorm."""
    def __init__(self, num_channels, kernel_size, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size,
                               stride=1, padding=kernel_size // 2, bias=False)
        # Assuming input sequence length is 300, adjust if different
        self.norm1 = nn.LayerNorm([num_channels, 300]) # Normalize across channels and length
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size,
                               stride=1, padding=kernel_size // 2, bias=False)
        self.norm2 = nn.LayerNorm([num_channels, 300])
        self.relu2 = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.relu2(out)
        return out

class BindingPredictorCNN(nn.Module):
    def __init__(self, num_channels, num_blocks, kernel_size, dropout_rate):
        super().__init__()
        self.initial_conv = nn.Conv1d(4, num_channels, kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=False)
        # Assuming input sequence length is 300, adjust if different
        self.initial_norm = nn.LayerNorm([num_channels, 300])
        self.initial_relu = nn.ReLU()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, kernel_size, dropout_rate) for _ in range(num_blocks)]
        )

        self.final_conv = nn.Conv1d(num_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
         for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # He init for initial/block convs
                # Need to access num_channels from the instance, not the global scope
                if m.out_channels == self.initial_conv.out_channels or m.in_channels == 4:
                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Default init might be okay for final 1x1 conv, or use Xavier
                elif m.out_channels == 1:
                     nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            # LayerNorm weights/biases initialized by default

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = self.initial_relu(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        x = x.squeeze(1)
        return x

# Training Loop

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # history dictionary removed, wandb handles logging

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training Phase
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        # Log epoch train loss to wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': avg_train_loss})
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
                all_preds.append(outputs)
                all_targets.append(targets)

        avg_val_loss = running_val_loss / len(val_loader)

        # Calculate Pearson correlation on validation set
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        val_pearson = calculate_pearsonr(all_preds_tensor, all_targets_tensor)

        # Log epoch validation metrics to wandb
        wandb.log({'epoch': epoch + 1, 'val_loss': avg_val_loss, 'val_pearsonr': val_pearson})
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Pearson R: {val_pearson:.4f}")

        # Early Stopping (Model saving removed)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            print(f"Validation loss improved to {best_val_loss:.4f}.")
            # Model saving logic removed
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("Training finished for this run.")
    # Return value changed, history is not needed
    return model


# Wandb Sweep Function
def train_sweep(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Device Setup
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS device (Apple Silicon GPU).")
        # elif torch.cuda.is_available():
        #     device = torch.device("cuda")
        #     print("Using CUDA device.")
        else:
            device = torch.device("cpu")
            print("Using CPU device.")

        # Load Data
        print("Loading data...")
        try:
            sequences_df = pd.read_csv(SEQ_FILE, sep="\t", compression='gzip')
            scores_df = pd.read_csv(SCORE_FILE, sep="\t", compression='gzip')
            print(f"Loaded {len(sequences_df)} sequences and {len(scores_df.columns)} score vectors.")
            if len(sequences_df) != len(scores_df.columns):
                 print(f"Warning: Sequence/score count mismatch. Assuming order matches.")

        except FileNotFoundError:
            print(f"Error: Data files not found in {DATA_DIR}.")
            exit()
        except Exception as e:
            print(f"Error loading data: {e}")
            exit()

        # Create Dataset & Dataloaders
        print("Creating dataset and dataloaders...")
        dataset = TFBindiingDataset(sequences_df, scores_df)

        # Split dataset using config
        val_split = config.validation_split
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

        # Use batch_size from config
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # Initialize Model, Loss, Optimizer using config
        print("Initializing model...")
        model = BindingPredictorCNN(
            num_channels=config.num_channels,
            num_blocks=config.num_res_blocks,
            kernel_size=config.kernel_size,
            dropout_rate=config.dropout_rate
        ).to(device)

        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        criterion = nn.MSELoss()

        # Select optimizer based on config (currently only AdamW)
        if config.optimizer_type == 'AdamW':
             optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
             # Add other optimizers here if needed
             print(f"Warning: Optimizer type {config.optimizer_type} not recognized. Defaulting to AdamW.")
             optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


        # Train using config epochs and patience
        print("Starting training for sweep run...")
        trained_model = train_model(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=config.num_epochs, patience=config.early_stopping_patience
        )

# Main Execution
if __name__ == "__main__":
    wandb.login(key="d76ec97ab2b6308d17e9bd3dfa54f33c0f6ca39a") #API key added for simplicty (no security risk in this use-case)

    # Define the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="tf-binding-tune")
    print(f"Created/found sweep with ID: {sweep_id}")
    print(f"Sweep URL: https://wandb.ai/lee-inhyeok-university-of-chicago/tf-binding-tune/sweeps/{sweep_id}")

    # Start the wandb agent
    print("Starting wandb agent...")
    # count=3 ensures we run exactly the 3 configurations defined in the grid search
    wandb.agent(sweep_id, function=train_sweep, count=3)

    print("Wandb sweep finished.")
