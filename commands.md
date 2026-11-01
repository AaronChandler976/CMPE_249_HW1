    git init
    conda env export --no-builds > environment.yml
    pip freeze > requirements.txt
    nvidia-smi > results\hardware.txt
## Stage 0 - setup (2026-11-01)
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # True NVIDIA GeForce RTX 5070 Ti
    pip install git+https://github.com/ultralytics/CLIP.git tabulate
    gh repo create CMPE_249_HW1 --private --source=. --remote=origin
