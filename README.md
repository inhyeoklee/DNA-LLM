# DNA-LLM

A language model for DNA sequence modeling, with pretraining and fine-tuning capabilities.

## Pretraining

The `3_dna-llm-pretraining.py` script trains a language model on DNA sequences. It supports two configurations:
- `full`: Using the full vocabulary (40 tokens)
- `vocab4`: Using a reduced vocabulary (4 tokens - A,C,G,T)

To run pretraining:

```bash
# For the full vocabulary model
python 3_dna-llm-pretraining.py --data_dir data/full --output_dir outputs

# For the reduced vocabulary model
python 3_dna-llm-pretraining.py --data_dir data/vocab4 --output_dir outputs
```

When both models have been pretrained, a training time comparison plot will be automatically generated and saved to `outputs/training_time_comparison.png`.

### Pretraining Transformer Architecture

The core transformer model used for pretraining (design credit: Henry Raeder) follows this general data flow:

```ascii
Input Sequence (Token IDs)
       | [Batch, SeqLen]
       V
+---------------------+
| Token Embedding     | (wte: lookup V x E)
+---------------------+
       | [Batch, SeqLen, E]
       V
+---------------------+      +-----------------------+
| Position Embedding  |----->| Add Element-wise      | (wpe: lookup S x E)
+---------------------+      +-----------------------+
                                      | [Batch, SeqLen, E]
                                      V
                               +---------------------+
                               | Dropout             |
                               +---------------------+
                                      |
                                      V
+--------------------------------------------------------------------------+
| Repeat N times (N = n_layer = 4)                                         |
| +--------------------------- Transformer Block -------------------------+ |
| | Input                                                                  | |
| |   |                                                                    | |
| |   |-----------------------+                                            | |
| |   V                       |                                            | |
| | +---------------------+   |                                            | |
| | | LayerNorm (ln_1)    |   |                                            | |
| | +---------------------+   |                                            | |
| |   |                       |                                            | |
| |   V                       |                                            | |
| | +---------------------+   |                                            | |
| | | MultiHeadAttention  |   | (attn: c_attn -> QKV -> Attn -> c_proj)    | |
| | +---------------------+   |                                            | |
| |   |                       |                                            | |
| |   V                       V                                            | |
| | +---------------------+ Add (Residual 1)                               | |
| | | Add & Norm          |-----------------------+                        | |
| | +---------------------+                       |                        | |
| |   |                                           |                        | |
| |   |-----------------------+                   |                        | |
| |   V                       |                   |                        | |
| | +---------------------+   |                   |                        | |
| | | LayerNorm (ln_2)    |   |                   |                        | |
| | +---------------------+   |                   |                        | |
| |   |                       |                   |                        | |
| |   V                       |                   |                        | |
| | +---------------------+   |                   |                        | |
| | | MLP FeedForward     |   | (mlp: c_fc -> GELU -> c_proj)              | |
| | +---------------------+   |                   |                        | |
| |   |                       |                   |                        | |
| |   V                       V                   |                        | |
| | +---------------------+ Add (Residual 2)      |                        | |
| | | Add & Norm          |-----------------------+                        | |
| | +---------------------+                                                | |
| |   |                                                                    | |
| |   V Output of Block                                                    | |
| +--------------------------------------------------------------------------+
       | [Batch, SeqLen, E] (Output after N blocks)
       V
+---------------------+
| LayerNorm (ln_f)    |
+---------------------+
       | [Batch, SeqLen, E]
       V
+---------------------+
| Linear Head (lm_head)| (Project E -> V, Tied w/ wte)
+---------------------+
       | [Batch, SeqLen, V]
       V
  Output Logits
```

**Legend:**

*   `V`: `vocab_size` (40 for 'full', 4 for 'vocab4')
*   `E`: `n_embd` (embedding dimension = 384)
*   `S`: `block_size` (sequence length = 300)
*   `N`: `n_layer` (number of blocks = 4)
*   `[ ]`: Represents data shape

## Fine-tuning for TF-binding prediction

After pretraining, models can be fine-tuned for the TF-binding prediction task using the `4_dna-llm-finetuning.py` script.

```bash
# Fine-tune the full vocabulary model
python 4_dna-llm-finetuning.py --config full --checkpoint_dir outputs --output_dir tf-binding-prediction/models

# Fine-tune the reduced vocabulary model
python 4_dna-llm-finetuning.py --config vocab4 --checkpoint_dir outputs --output_dir tf-binding-prediction/models
```

This will:
1. Load the appropriate pretrained model from the checkpoint directory
2. Fine-tune it for the TF-binding prediction task 
3. Save the resulting model to the specified output directory

### Fine-tuning Architecture ("Full" and "Vocab4" Models)

During fine-tuning, the pre-trained GPT core is loaded and frozen. Only the added regression head is trained.

```ascii
Input Sequence (Token IDs, Vocab V depends on config: 40 or 4)
       | [Batch, SeqLen]
       V
+--------------------------------------------------------------------------+
| FROZEN Pre-trained GPT Core (Loaded from Checkpoint)                     |
|                                                                          |
| +---------------------+                                                  |
| | Token Embedding     | (wte: lookup V x E)                              |
| +---------------------+                                                  |
|        |                                                                 |
|        V                                                                 |
| +---------------------+      +-----------------------+                   |
| | Position Embedding  |----->| Add Element-wise      | (wpe: lookup S x E) |
| +---------------------+      +-----------------------+                   |
|                                       |                                  |
|                                       V                                  |
|                                +---------------------+                   |
|                                | Dropout             |                   |
|                                +---------------------+                   |
|                                       |                                  |
|                                       V                                  |
| +----------------------------------------------------------------------+ |
| | Repeat N times (N = n_layer = 4)                                     | |
| | +----------------------- Transformer Block -------------------------+ | |
| | | LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual | | |
| | +--------------------------------------------------------------------+ | |
| +----------------------------------------------------------------------+ |
|        |                                                                 |
|        V                                                                 |
| +---------------------+                                                  |
| | LayerNorm (ln_f)    |                                                  |
| +---------------------+                                                  |
|        | [Batch, SeqLen, E] (Output Embeddings from Frozen Core)         |
+--------------------------------------------------------------------------+
       |
       V
+--------------------------------------------------------------------------+
| TRAINABLE Regression Head (`ImprovedRegressionHead`)                     |
|                                                                          |
| +---------------------+                                                  |
| | LayerNorm           | (Input dim = E = 384)                            |
| +---------------------+                                                  |
|        |                                                                 |
|        V                                                                 |
| +---------------------+                                                  |
| | Conv1D              | (Kernel=11, Hidden=128)                          |
| +---------------------+                                                  |
|        |                                                                 |
|        V                                                                 |
| +---------------------+                                                  |
| | GELU Activation     |                                                  |
| +---------------------+                                                  |
|        |                                                                 |
|        V                                                                 |
| +---------------------+                                                  |
| | Dropout             |                                                  |
| +---------------------+                                                  |
|        |                                                                 |
|        V                                                                 |
| +---------------------+                                                  |
| | Linear FC           | (Project Hidden -> 1)                            |
| +---------------------+                                                  |
|        | [Batch, SeqLen] (Output Scores)                                 |
+--------------------------------------------------------------------------+
       V
  Final TF-Binding Scores
```

**Legend:** (Same as pretraining, E=384, S=300, N=4. V=40 for 'full', V=4 for 'vocab4')

## Baseline CNN Model

A baseline CNN model (`5_dna-baseline.py`) is provided for comparison. This model uses one-hot encoding of the input sequence (A, C, G, T) and applies the same regression head architecture and data augmentation (sequence shifting) used in the fine-tuning script, but without the pretrained transformer backbone.

To train the baseline model and generate comparison plots:

```bash
python 5_dna-baseline.py --output_dir models
```

This will:
1. Train the baseline CNN model on the TF-binding data.
2. Save the baseline model to `models/cnn_baseline_model.pt`.
3. Generate baseline training curves `models/training_curves_baseline.png`.
4. Update the runtime comparison plot `models/finetuning_time_comparison.png` to include the baseline.
5. Generate a new performance comparison plot `models/performance_comparison.png`.

To skip training and only generate the comparison plots using existing model files (assuming `gpt_binding_model_full.pt`, `gpt_binding_model_vocab4.pt`, and `cnn_baseline_model.pt` exist in the output directory):

```bash
python 5_dna-baseline.py --output_dir models --plot-only
```

## Model Architecture Comparison

This project explores three models for predicting TF-binding scores from 300bp DNA sequences.

**1. Baseline CNN (`5_dna-baseline.py`)**
*   **Architecture:** A simple CNN trained from scratch (initial Conv1D + regression head).
*   **Input Processing:** Takes 300bp sequences, converts each base ('A', 'C', 'G', 'T') into a 4-element **one-hot encoded vector**. 'N' or other characters become zero vectors. The resulting `(batch, 4, seq_len)` tensor is fed into the `initial_conv` layer.
*   **Parameters (~187k, All Trainable):**
    *   `initial_conv` (in=4, out=128, kernel=11, bias=True): `(in_channels * out_channels * kernel_size) + bias = (4 * 128 * 11) + 128 = 5,760` params
    *   `ImprovedRegressionHead` (input_dim=128, hidden_dim=128): `180,737` params (see breakdown below)
        *   `layer_norm` (dim=128): `weight + bias = 128 + 128 = 256`
        *   `conv1d` (in=128, out=128, kernel=11): `(in * out * kernel) + bias = (128 * 128 * 11) + 128 = 180,352`
        *   `fc` (in=128, out=1): `(in * out) + bias = (128 * 1) + 1 = 129`
    *   Total: `5,760 + 180,737 = **186,497**`

**2. Fine-tuned "Full" (`4_dna-llm-finetuning.py`, `config='full'`)**
*   **Architecture:** Pre-trained GPT core (frozen, `n_layer=4`, `n_embd=384`) + trainable regression head.
*   **Input Processing:** Takes 300bp sequences, **tokenizes** them into integer IDs using the 'full' vocabulary (`vocab_size=40`). Looks up **dense embedding vectors** (`n_embd=384`) from the frozen GPT's `wte` layer. The resulting `(batch, seq_len, n_embd)` tensor is fed into the trainable regression head.
*   **Parameters (~7.75 Million Total):**
    *   *Frozen GPT Core (~7.21M):* **7,211,904** params (see breakdown below)
        *   Token Embeddings `transformer.wte` (vocab_size=40, n_embd=384): `vocab_size * n_embd = 40 * 384 = 15,360`
        *   Position Embeddings `transformer.wpe` (block_size=300, n_embd=384): `block_size * n_embd = 300 * 384 = 115,200`
        *   Transformer Blocks `transformer.h` (n_layer=4, n_embd=384, bias=False): `n_layer * params_per_block = 4 * 1,770,240 = 7,080,960`
            *   *Per Block (`1,770,240` params):* `ln_1` + `attn` + `ln_2` + `mlp`
            *   `ln_1` LayerNorm (n_embd=384, bias=False): `weight = n_embd = 384`
            *   `attn` CausalSelfAttention (`589,824` params): `c_attn` + `c_proj`
                *   `c_attn` Linear (n_embd -> 3*n_embd): `n_embd * (3 * n_embd) = 384 * (3 * 384) = 442,368`
                *   `c_proj` Linear (n_embd -> n_embd): `n_embd * n_embd = 384 * 384 = 147,456`
            *   `ln_2` LayerNorm (n_embd=384, bias=False): `weight = n_embd = 384`
            *   `mlp` MLP (`1,179,648` params): `c_fc` + `c_proj`
                *   `c_fc` Linear (n_embd -> 4*n_embd): `n_embd * (4 * n_embd) = 384 * (4 * 384) = 589,824`
                *   `c_proj` Linear (4*n_embd -> n_embd): `(4 * n_embd) * n_embd = (4 * 384) * 384 = 589,824`
        *   Final LayerNorm `transformer.ln_f` (n_embd=384, bias=False): `weight = n_embd = 384`
    *   *Trainable Regression Head (~542k):* **541,697** params (see breakdown below)
        *   `layer_norm` (dim=384): `weight + bias = 384 + 384 = 768`
        *   `conv1d` (in=384, out=128, kernel=11): `(in * out * kernel) + bias = (384 * 128 * 11) + 128 = 540,800`
        *   `fc` (in=128, out=1): `(in * out) + bias = (128 * 1) + 1 = 129`
    *   Total: `7,211,904 (Frozen) + 541,697 (Trainable) = **7,753,601**`

**3. Fine-tuned "Vocab4" (`4_dna-llm-finetuning.py`, `config='vocab4'`)**
*   **Architecture:** Identical to "Full" (frozen GPT core + trainable regression head).
*   **Input Processing:** Takes 300bp sequences, **tokenizes** them using 'vocab4' vocabulary (`vocab_size=4`). Looks up **dense embedding vectors** (`n_embd=384`) from the frozen GPT's `wte` layer. The resulting `(batch, seq_len, n_embd)` tensor is fed into the trainable regression head.
*   **Parameters (~7.74 Million Total):**
    *   *Frozen GPT Core (~7.20M):* **7,198,080** params (see breakdown below)
        *   Token Embeddings `transformer.wte` (vocab_size=4, n_embd=384): `vocab_size * n_embd = 4 * 384 = 1,536`
        *   Position Embeddings `transformer.wpe` (block_size=300, n_embd=384): `block_size * n_embd = 300 * 384 = 115,200`
        *   Transformer Blocks `transformer.h` (n_layer=4, n_embd=384, bias=False): `n_layer * params_per_block = 4 * 1,770,240 = 7,080,960` (Block breakdown above)
        *   Final LayerNorm `transformer.ln_f` (n_embd=384, bias=False): `weight = n_embd = 384`
    *   *Trainable Regression Head (~542k):* Identical to "Full" model's head = **541,697** (breakdown above)
    *   Total: `7,198,080 (Frozen) + 541,697 (Trainable) = **7,739,777**`

**Key Differences & Notes:**
*   **Input:** Baseline uses sparse one-hot encoding (4 dims). Fine-tuned models use dense embeddings (384 dims) derived from tokenization.
*   **Training:** Baseline trains the whole ~187k model. Fine-tuned models freeze the large ~7.2M GPT core and only train the ~542k regression head.
*   **Parameters:** The vocabulary size difference (40 vs 4) minimally impacts total parameters (~14k difference) because the core transformer blocks dominate. Both fine-tuned models have identical trainable parameters (~542k).

## Outputs

- Pretrained models: `outputs/ckpt_full.pt` and `outputs/ckpt_vocab4.pt`
- Loss curves: `outputs/loss_full.png` and `outputs/loss_vocab4.png`
- Loss data: `outputs/losses_full.json` and `outputs/losses_vocab4.json`
- Pretraining time comparison: `outputs/training_time_comparison.png`
- Fine-tuned models: `models/gpt_binding_model_full.pt` and `models/gpt_binding_model_vocab4.pt` (Note: Default output is now `models/`)
- Fine-tuning curves: `models/training_curves_full.png` and `models/training_curves_vocab4.png`
- Baseline model: `models/cnn_baseline_model.pt`
- Baseline curves: `models/training_curves_baseline.png`
- Runtime comparison (Fine-tuning + Baseline): `models/finetuning_time_comparison.png`
- Performance comparison (Fine-tuning + Baseline): `models/performance_comparison.png`
