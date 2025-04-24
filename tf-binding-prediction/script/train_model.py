import os
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import argparse

# Data Configuration
DATA_DIR = '/Users/ihlee/Desktop/DL-Genomics/tf-binding-prediction/data'
SEQ_FILE = os.path.join(DATA_DIR, 'chr22_sequences.txt.gz')
SCORE_FILE = os.path.join(DATA_DIR, 'chr22_scores.txt.gz')

# Model Hyperparameters
NUM_CHANNELS = 32
NUM_RES_BLOCKS = 3
DROPOUT_RATE = 0.2
KERNEL_SIZE = 3

# Training Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10 # Stop if validation loss doesn't improve for 10 epochs

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
        return 0.0 # Handle cases where variance is zero

def plot_comparison(pred, obs, title="Comparison", save_path=None):
    """Plots predicted vs observed scores for a single sequence."""
    pred = pred.detach().cpu().numpy()
    obs = obs.detach().cpu().numpy()
    r_value = calculate_pearsonr(torch.tensor(pred), torch.tensor(obs)) # Use tensor version for consistency
    x = np.arange(len(pred))
    bar_width = 0.4
    plt.figure(figsize=(12, 5))
    plt.bar(x - bar_width/2, pred, width=bar_width, label="Predicted", alpha=0.7, color='b')
    plt.bar(x + bar_width/2, obs, width=bar_width, label="Observed", alpha=0.7, color='r')
    plt.xlabel("Position in sequence window")
    plt.ylabel("Homer Score")
    plt.title(f"{title}\nPearson R: {r_value:.3f}")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path)
        plt.close() 
    else:
        plt.show()


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
             # Handle potential index mismatch - raise error or return dummy
             raise IndexError(f"Score index {idx} out of bounds for window {window_name}")

        return self.encoded_sequences[idx], score_vec

# Model Architecture

class ResidualBlock(nn.Module):
    """Residual Block with LayerNorm."""
    def __init__(self, num_channels, kernel_size, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size,
                               stride=1, padding=kernel_size // 2, bias=False)
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
            # LayerNorm weights/biases are typically initialized to 1s and 0s by default

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity # Add shortcut
        out = self.relu2(out)
        return out

class BindingPredictorCNN(nn.Module):
    def __init__(self, num_channels, num_blocks, kernel_size, dropout_rate):
        super().__init__()
        self.initial_conv = nn.Conv1d(4, num_channels, kernel_size,
                                      stride=1, padding=kernel_size // 2, bias=False)
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
                if m.out_channels == NUM_CHANNELS or m.in_channels == 4:
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
        x = x.squeeze(1) # Remove channel dimension -> (Batch, 300)
        return x

# Training Loop

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_pearsonr': []}

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
            if (i + 1) % 100 == 0: # Print progress every 100 batches
                 print(f"  Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")


        avg_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        #  Validation Phase 
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
        history['val_loss'].append(avg_val_loss)

        # Calculate Pearson correlation on validation set
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        val_pearson = calculate_pearsonr(all_preds_tensor, all_targets_tensor)
        history['val_pearsonr'].append(val_pearson)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Pearson R: {val_pearson:.4f}")

        #  Early Stopping & Model Saving 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model as TorchScript
            model.eval() # Ensure model is in eval mode for scripting consistency
            try:
                print("Scripting the model for TorchScript...")
                scripted_model = torch.jit.script(model)
                torch.jit.save(scripted_model, 'script/lee-inhyeok-model.pth')
                print(f"Validation loss improved. Saved best TorchScript model.")
            except Exception as e:
                print(f"Error scripting or saving model as TorchScript: {e}. Saving state_dict instead.")
                # Fallback to saving state_dict if scripting fails
                torch.save(model.state_dict(), 'script/lee-inhyeok-model.pth')
            finally:
                 model.train() 
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("Training finished.")
    return model, history


# Main Execution
if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Train or convert TF Binding Prediction Model.')
    parser.add_argument('--convert_model', action='store_true',
                        help='Convert existing state_dict model to TorchScript and exit.')
    args = parser.parse_args()

    # Device Setup (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU).")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Conversion to fully TorchScript model
    if args.convert_model:
        print("Attempting to convert existing 'script/lee-inhyeok-model.pth' to TorchScript...")
        model_path = 'script/lee-inhyeok-model.pth'
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' not found. Cannot convert.")
            exit()

        try:
            # Instantiate model architecture
            model = BindingPredictorCNN(
                num_channels=NUM_CHANNELS,
                num_blocks=NUM_RES_BLOCKS,
                kernel_size=KERNEL_SIZE,
                dropout_rate=DROPOUT_RATE # Use hyperparameters defined above
            ) # Keep model on CPU for loading state_dict first

            # Load state dict (map_location ensures compatibility)
            print(f"Loading state_dict from {model_path}...")
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            print("State_dict loaded successfully.")

            # Set to evaluation mode
            model.eval()

            # Script the model
            print("Scripting the model...")
            scripted_model = torch.jit.script(model)
            print("Model scripted successfully.")

            # Save the TorchScript model (overwriting the original)
            torch.jit.save(scripted_model, model_path)
            print(f"Successfully converted and saved TorchScript model to '{model_path}'.")
            exit() # Exit after conversion

        except Exception as e:
            print(f"Error during model conversion: {e}")
            exit()

    # Regular Training
    # Load Data
    print("Loading data...")
    try:
        sequences_df = pd.read_csv(SEQ_FILE, sep="\t", compression='gzip')
        scores_df = pd.read_csv(SCORE_FILE, sep="\t", compression='gzip')
        print(f"Loaded {len(sequences_df)} sequences and {len(scores_df.columns)} score vectors.")

        # Basic check: Ensure number of sequences matches number of score vectors
        if len(sequences_df) != len(scores_df.columns):
             print(f"Warning: Number of sequences ({len(sequences_df)}) does not match number of score columns ({len(scores_df.columns)}). Assuming order matches.")
             # Adjust scores_df if necessary, e.g., only take first N columns
             # scores_df = scores_df.iloc[:, :len(sequences_df)]

    except FileNotFoundError:
        print(f"Error: Data files not found in {DATA_DIR}. Please ensure '{os.path.basename(SEQ_FILE)}' and '{os.path.basename(SCORE_FILE)}' exist.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # Create Dataset & Dataloaders 
    print("Creating dataset and dataloaders...")
    dataset = TFBindiingDataset(sequences_df, scores_df)

    # Split dataset
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize Model, Loss, Optimizer 
    print("Initializing model...")
    model = BindingPredictorCNN(
        num_channels=NUM_CHANNELS,
        num_blocks=NUM_RES_BLOCKS,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    print(model)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Train 
    print("Starting training...")
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, NUM_EPOCHS, EARLY_STOPPING_PATIENCE
    )

    # Plot Training History 
    print("Plotting training history...")
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], label='Train Loss', color='red', linestyle='--')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Pearson R', color=color)
    ax2.plot(history['val_pearsonr'], label='Validation Pearson R', color='blue')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Training History')
    plt.savefig('training_history.png')
    plt.show()

    # Prediction Plot 
    print("Generating prediction plot...")
    # Get 1st sample from 1st batch in the validation set
    model.eval()
    with torch.no_grad():
        example_input, example_target = next(iter(val_loader))
        example_input = example_input.to(device)
        example_target = example_target.to(device)
        example_pred = model(example_input)

        # Plot
        plot_comparison(example_pred[1], example_target[0], title="Example Validation Prediction", save_path="example_prediction2.png")

    print("Script finished.")
