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
