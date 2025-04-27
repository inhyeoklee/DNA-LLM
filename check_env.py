#!/usr/bin/env python3
import sys
import platform
import subprocess

def check_module(name):
    try:
        m = __import__(name)
        ver = getattr(m, "__version__", "unknown")
        return ver
    except ImportError:
        return None

def main():
    print("=== Environment Check ===")
    print(f"Python: {platform.python_implementation()} {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    # Check key libraries
    libs = ["numpy", "torch", "requests", "matplotlib"]
    for lib in libs:
        ver = check_module(lib)
        if ver:
            print(f"{lib} version: {ver}")
        else:
            print(f"{lib} not installed")
    print()

    # CUDA / GPU check via torch
    try:
        import torch
        print("Torch CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {name} (compute capability {cap[0]}.{cap[1]})")
    except Exception as e:
        print("Error checking CUDA via torch:", e)
    print()

    # CUDNN status
    try:
        import torch.backends.cudnn as cudnn
        print("cuDNN enabled:", cudnn.enabled)
    except Exception:
        pass
    print()

    # List all installed packages
    print("=== pip freeze ===")
    subprocess.run([sys.executable, "-m", "pip", "freeze"], check=False)

if __name__ == "__main__":
    main()
