import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import clip

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing
model, preprocess = clip.load('ViT-B/32', device)
