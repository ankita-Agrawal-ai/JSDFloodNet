
import os
import argparse
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.model import JSDFloodNet
from utils.dataset_loader import Sen1Floods11Dataset
from utils.metrics import iou_score, dice_score


try:
    from utils.diffusion import Diffusion
    HAS_DIFFUSION = True
except Exception:
    HAS_DIFFUSION = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data', help='root data directory with train/ and test/')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='results')
    p.add_argument('--in-ch', type=int, default=5, help='number of input channels (S1+S2)')
    # Diffusion options
    p.add_argument('--use-diffusion', action='store_true', help='Enable joint diffusion loss during training')
    p.add_argument('--diff-timesteps', type=int, default=200, help='Diffusion timesteps (if using diffusion)')
    p.add_argument('--lambda-diff', type=float, default=0.5, help='Weight for diffusion loss when joint training')
    return p.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = args.device

    # Dataset + loader
    dataset = Sen1Floods11Dataset(args.data_dir, split='train', input_ext='.npy', mask_ext='.png')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Segmentation model
    model = JSDFloodNet(in_channels=args.in_ch, n_classes=1)
    model = model.to(device)

    if args.use_diffusion:
        if not HAS_DIFFUSION:
            raise RuntimeError("Diffusion utilities not found. Add utils/diffusion.py or disable --use-diffusion.")
        # denoiser predicts per-channel noise; reuse same UNet structure but output channels = in_ch
        denoiser = JSDFloodNet(in_channels=args.in_ch, n_classes=args.in_ch)
        denoiser = denoiser.to(device)
        diff = Diffusion(timesteps=args.diff_timesteps, device=device)
    else:
        denoiser = None
        diff = None

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(list(model.parameters()) + ([] if denoiser is None else list(denoiser.parameters())),
                           lr=args.lr)

    best_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        if denoiser is not None:
            denoiser.train()

        running_loss = 0.0
        running_seg_loss = 0.0
        running_diff_loss = 0.0

        for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            x = x.to(device).float()            # x: (B, C, H, W)
            y = y.to(device).float()            # y: (B, 1, H, W)

            # Skip very small patches (optional)
            if x.size(2) < 64 or x.size(3) < 64:
                continue

            optimizer.zero_grad()

            # --- Segmentation forward + loss ---
            logits = model(x)                   # (B,1,H,W)
            seg_loss = criterion(logits, y)

            total_loss = seg_loss

            # --- Optional diffusion loss (predict noise) ---
            if denoiser is not None and diff is not None:
                # sample timesteps for each sample in batch
                t = torch.randint(0, diff.timesteps, (x.size(0),), device=device).long()
                # sample gaussian noise
                noise = torch.randn_like(x)
                # produce noisy version x_t
                x_t = diff.q_sample(x_start=x, t=t, noise=noise)
                # denoiser should predict the noise added
                pred_noise = denoiser(x_t)  # expected shape (B, in_ch, H, W)
                # If denoiser outputs n_classes channels with sigmoid, ensure shape matches noise
                # compute MSE between predicted noise and true noise
                diff_loss = nn.functional.mse_loss(pred_noise, noise)
                total_loss = seg_loss + args.lambda_diff * diff_loss
                running_diff_loss += diff_loss.item()

            # backward + step
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_seg_loss += seg_loss.item()

        # average losses
        num_batches = len(loader) if len(loader) > 0 else 1
        avg_loss = running_loss / num_batches
        avg_seg_loss = running_seg_loss / num_batches
        avg_diff_loss = (running_diff_loss / num_batches) if denoiser is not None else 0.0

        # quick validation on small subset (here using first batch from training loader as proxy)
        model.eval()
        if denoiser is not None:
            denoiser.eval()

        with torch.no_grad():
            try:
                xs, ys = next(iter(loader))
                xs = xs.to(device).float()
                ys = ys.to(device).float()
                out = model(xs)
                probs = torch.sigmoid(out)
                iou = iou_score(probs, ys)
                dice = dice_score(probs, ys)
            except StopIteration:
                iou = 0.0
                dice = 0.0

        print(f'Epoch {epoch+1}: avg_loss={avg_loss:.4f}, seg_loss={avg_seg_loss:.4f}, diff_loss={avg_diff_loss:.4f}, iou={iou:.4f}, dice={dice:.4f}')

        # save best model by IoU
        if iou > best_iou:
            best_iou = iou
            save_model_path = os.path.join(args.output, 'model.pkl')
            torch.save(model.state_dict(), save_model_path)
            print(f'Saved best segmentation model to {save_model_path} (iou={best_iou:.4f})')
            if denoiser is not None:
                save_denoiser_path = os.path.join(args.output, 'denoiser.pkl')
                torch.save(denoiser.state_dict(), save_denoiser_path)
                print(f'Saved denoiser model to {save_denoiser_path}')

    print('Training finished.')


if __name__ == '__main__':
    train()

