@echo off
echo Running fine-tuning for vocab4 configuration...
python dna-llm-finetuning.py --config vocab4 --checkpoint_dir outputs --output_dir models
echo.
echo Vocab4 fine-tuning complete.
echo.
echo Running fine-tuning for full vocabulary configuration...
python dna-llm-finetuning.py --config full --checkpoint_dir outputs --output_dir models
echo.
echo Full vocabulary fine-tuning complete.
echo.
echo Both fine-tuning runs complete.
echo Generating time comparison plot...
python -c "import torch; import matplotlib.pyplot as plt; import os; import sys; sys.path.append('.'); from importlib.util import spec_from_file_location, module_from_spec; spec = spec_from_file_location('finetuning', 'dna-llm-finetuning.py'); module = module_from_spec(spec); spec.loader.exec_module(module); module.plot_finetuning_time_comparison()"
echo.
echo Done!
