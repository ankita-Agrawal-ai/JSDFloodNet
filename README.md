# JSDFloodNet


PyTorch implementation for flood segmentation using the curated and Sen1Floods11 dataset (Sentinel-1 SAR + Sentinel-2 optical). This repo contains training and testing scripts, utilities, and example output generation.


## Structure
JSDFloodNet/ ├── README.md ├── requirements.txt ├── train.py ├── test.py ├── utils/ │ ├── model.py │ ├── dataset_loader.py │ └── metrics.py ├── data/ │ ├── train/ │ └── test/ └── results/ ├── sample.png └── sample2.png


## Setup

```bash
python -m venv .venv
source .venv/bin/activate # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

Files

train.py: train model and save model.pkl.

test.py: load model.pkl and run inference to produce segmentation PNG outputs.

utils/model.py: model architecture (UNet-like, accepts combined S1+S2 channels).

utils/dataset_loader.py: dataset class for Sen1Floods11 (expects paired image and mask files).

utils/metrics.py: IoU and Dice metrics.

How to run

Prepare dataset under data/train/ and data/test/. Each sample should contain an input stack (S1+S2) and a mask image with same spatial dimensions.

Train:
python train.py --data-dir data --epochs 40 --batch-size 8 --output results

Test / inference:
python test.py --model-path results/model.pkl --input-dir data/test --output-dir results

Notes

Model is saved with torch.save(model.state_dict(), path).

The code assumes combined S1+S2 channels by default (in_channels=5). Edit in_channels in utils/model.py if your input differs.
