# DNA-LLM

A language model for DNA sequence modeling, with pretraining and fine-tuning capabilities.

## Pretraining

The `dna-llm-pretraining.py` script trains a language model on DNA sequences. It supports two configurations:
- `full`: Using the full vocabulary (40 tokens)
- `vocab4`: Using a reduced vocabulary (4 tokens - A,C,G,T)

To run pretraining:

```bash
# For the full vocabulary model
python dna-llm-pretraining.py --data_dir data/full --output_dir outputs

# For the reduced vocabulary model
python dna-llm-pretraining.py --data_dir data/vocab4 --output_dir outputs
```

When both models have been pretrained, a training time comparison plot will be automatically generated and saved to `outputs/training_time_comparison.png`.

## Fine-tuning for TF-binding prediction

After pretraining, models can be fine-tuned for the TF-binding prediction task using the `dna-llm-finetuning.py` script.

```bash
# Fine-tune the full vocabulary model
python dna-llm-finetuning.py --config full --checkpoint_dir outputs --output_dir tf-binding-prediction/models

# Fine-tune the reduced vocabulary model
python dna-llm-finetuning.py --config vocab4 --checkpoint_dir outputs --output_dir tf-binding-prediction/models
```

This will:
1. Load the appropriate pretrained model from the checkpoint directory
2. Fine-tune it for the TF-binding prediction task 
3. Save the resulting model to the specified output directory

## Outputs

- Pretrained models: `outputs/ckpt_full.pt` and `outputs/ckpt_vocab4.pt`
- Loss curves: `outputs/loss_full.png` and `outputs/loss_vocab4.png`
- Loss data: `outputs/losses_full.json` and `outputs/losses_vocab4.json`
- Training time comparison: `outputs/training_time_comparison.png`
- Fine-tuned models: `tf-binding-prediction/models/gpt_binding_model_full.pt` and `tf-binding-prediction/models/gpt_binding_model_vocab4.pt`
