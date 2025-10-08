import os
p.add_argument('--in-ch', type=int, default=5, help='Number of input channels (e.g., 5 for S1+S2)')
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Computation device')
return p.parse_args()




def save_mask(mask, path):
mask_img = Image.fromarray((mask * 255).astype(np.uint8))
mask_img.save(path)




def overlay_and_save(input_arr, mask, path):
# Convert input to RGB visualization
if input_arr.ndim == 3 and input_arr.shape[0] >= 3:
rgb = input_arr[:3]
else:
rgb = np.repeat(input_arr[0:1], 3, axis=0)
rgb = np.transpose(rgb, (1, 2, 0))
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-7)
rgb = (rgb * 255).astype(np.uint8)


im = Image.fromarray(rgb)
mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')


overlay = Image.new('RGBA', im.size, (255, 0, 0, 0))
overlay_mask = mask_img.point(lambda p: 120 if p > 0 else 0)
overlay.putalpha(overlay_mask)


blended = Image.alpha_composite(im.convert('RGBA'), overlay)
blended.save(path)




def main():
args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)
device = args.device


# Load model
model = JSDFloodNet(in_channels=args.in_ch, n_classes=1)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()


# Process test inputs
inputs = [f for f in sorted(os.listdir(args.input_dir)) if f.endswith('.npy')]
if not inputs:
print(f'No .npy files found in {args.input_dir}')
return


for i, fname in enumerate(tqdm(inputs, desc='Testing')):
fpath = os.path.join(args.input_dir, fname)
arr = np.load(fpath).astype(np.float32) # HWC or CHW
if arr.ndim == 3 and arr.shape[0] not in [3, 5]:
arr = np.transpose(arr, (2, 0, 1)) # Convert to CHW


x = torch.from_numpy(arr).unsqueeze(0).to(device)
with torch.no_grad():
output = model(x)
prob = torch.sigmoid(output)[0, 0].cpu().numpy()
mask = (prob > 0.5).astype(np.uint8)


base_name = os.path.splitext(fname)[0]
mask_path = os.path.join(args.output_dir, f'{base_name}_mask.png')
overlay_path = os.path.join(args.output_dir, f'{base_name}_overlay.png')


save_mask(mask, mask_path)
overlay_and_save(arr, mask, overlay_path)
print(f'Saved: {mask_path}, {overlay_path}')


# Stop after saving two example outputs
if i >= 1:
break




if __name__ == '__main__':
main()
