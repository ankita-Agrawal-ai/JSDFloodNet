import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class Sen1Floods11Dataset(Dataset):
"""
Expects a directory with paired inputs and masks. Input image should be a stacked array
with S1 and S2 channels combined into one image file (e.g., a .npy or multi-channel TIFF) or two separate files per sample.


This implementation assumes inputs are stored as .npy arrays (H x W x C) or PNGs where last dim channels represent S1+S2.
Adjust loader depending on how your notebook saved the inputs.
"""
def __init__(self, root_dir, split='train', transform=None, input_ext='.npy', mask_ext='.png'):
self.root_dir = root_dir
self.split = split
self.transform = transform
self.input_ext = input_ext
self.mask_ext = mask_ext
self.samples = []
folder = os.path.join(root_dir, split)
if not os.path.exists(folder):
raise ValueError(f"Path {folder} does not exist")
for fname in sorted(os.listdir(folder)):
if fname.endswith(self.input_ext):
base = fname[:-len(self.input_ext)]
inp = os.path.join(folder, base + self.input_ext)
msk = os.path.join(folder, base + self.mask_ext)
if os.path.exists(msk):
self.samples.append((inp, msk))


def __len__(self):
return len(self.samples)


def _load_input(self, path):
if path.endswith('.npy'):
arr = np.load(path) # H x W x C
else:
# PIL handles multi-channel PNGs limitedly; convert to numpy
arr = np.array(Image.open(path))
# ensure shape HWC
if arr.ndim == 2:
arr = np.expand_dims(arr, -1)
# transpose to CHW
arr = arr.astype(np.float32)
arr = np.transpose(arr, (2,0,1))
return arr


def _load_mask(self, path):
m = np.array(Image.open(path).convert('L'))
# binary mask expected
m = (m > 0).astype(np.uint8)
return m


def __getitem__(self, idx):
inp_path, msk_path = self.samples[idx]
x = self._load_input(inp_path)
y = self._load_mask(msk_path)
# to tensor
x = torch.from_numpy(x)
y = torch.from_numpy(y).unsqueeze(0).float()
# optional transforms
if self.transform:
x, y = self.transform(x, y)
return x, y


# Simple augment function example (center crop or random crop)
def random_crop(x, y, size=256):
_, h, w = x.shape
if h == size and w == size:
return x, y
i = np.random.randint(0, h - size + 1)
j = np.random.randint(0, w - size + 1)
x = x[:, i:i+size, j:j+size]
y = y[:, i:i+size, j:j+size]
return x, y
