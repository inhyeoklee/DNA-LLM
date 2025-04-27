@echo off
echo Running fine-tuning for vocab4 configuration...
python 4_dna-llm-finetuning.py --config vocab4 --checkpoint_dir outputs --output_dir models
echo.
echo Vocab4 fine-tuning complete.
echo.
echo Running fine-tuning for full vocabulary configuration...
python 4_dna-llm-finetuning.py --config full --checkpoint_dir outputs --output_dir models
echo.
echo Full vocabulary fine-tuning complete.
echo.
echo Both fine-tuning runs complete.
echo Generating time comparison plot...
REM Note: The plot generation command below assumes the function is still accessible
REM       and the baseline script handles the combined plot generation now.
REM       If the function was removed or changed, this part might need adjustment.
REM       Alternatively, run 'python 5_dna-baseline.py --plot-only' separately.
REM python -c "import torch; import matplotlib.pyplot as plt; import os; import sys; sys.path.append('.'); from importlib.util import spec_from_file_location, module_from_spec; spec = spec_from_file_location('finetuning', '4_dna-llm-finetuning.py'); module = module_from_spec(spec); spec.loader.exec_module(module); module.plot_finetuning_time_comparison()"
echo.
echo Fine-tuning runs complete. Run 'python 5_dna-baseline.py --plot-only' to generate comparison plots.
echo Done!
