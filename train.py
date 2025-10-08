import os
import argparse
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.model import JSDFloodNet
from utils.dataset_loader import Sen1Floods11Dataset, random_crop
from utils.metrics import iou_score, dice_score




def parse_args():
p = argparse.ArgumentParser()
p.add_argument('--data-dir', default='data', help='root data directory with train/ and test/')
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--batch-size', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
p.add_argument('--output', default='results')
p.add_argument('--in-ch', type=int, default=5, help='number of input channels (S1+S2)')
return p.parse_args()




def train():
args = parse_args()
os.makedirs(args.output, exist_ok=True)
device = args.device


dataset = Sen1Floods11Dataset(args.data_dir, split='train', input_ext='.npy', mask_ext='.png')
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


model = JSDFloodNet(in_channels=args.in_ch, n_classes=1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


best_iou = 0.0


for epoch in range(args.epochs):
model.train()
running_loss = 0.0
for x,y in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
x = x.to(device).float()
y = y.to(device).float()
if x.size(2) < 64 or x.size(3) < 64:
continue
optimizer.zero_grad()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
running_loss += loss.item()
avg_loss = running_loss / len(loader)
# quick validation on small subset (here using training set first samples as proxy)
model.eval()
with torch.no_grad():
xs, ys = next(iter(loader))
xs = xs.to(device).float()
ys = ys.to(device).float()
out = model(xs)
probs = torch.sigmoid(out)
iou = iou_score(probs, ys)
dice = dice_score(probs, ys)
print(f'Epoch {epoch+1}: loss={avg_loss:.4f}, iou={iou:.4f},
