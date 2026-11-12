"""Confirm the project libraries load and PyTorch can use the GPU."""
import cv2
import matplotlib
import numpy as np
import pandas
import skfuzzy
import skimage
import sklearn
import streamlit
import torch
import torchvision

for module in (torch, torchvision, np, cv2, skimage, skfuzzy, sklearn, matplotlib, pandas, streamlit):
    print(f"{module.__name__:<12}{module.__version__}")

print(f"CUDA build  {torch.version.cuda}")
print(f"GPU found   {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the GPU.")

device = torch.device("cuda")
print(f"GPU name    {torch.cuda.get_device_name(device)}")
memory_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
print(f"GPU memory  {memory_gb:.1f} GB")

# A real calculation on the GPU, because is_available() alone can't catch a broken install.
matrix = torch.rand(2000, 2000, device=device)
result = (matrix @ matrix).sum().item()
print(f"GPU test    passed ({result:.0f})")
