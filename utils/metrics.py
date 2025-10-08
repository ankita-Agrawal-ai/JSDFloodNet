import torch


def iou_score(preds, targets, threshold=0.5, eps=1e-7):
preds = (preds > threshold).float()
targets = (targets > 0.5).float()
intersection = (preds * targets).sum(dim=(1,2,3))
union = (preds + targets - preds * targets).sum(dim=(1,2,3))
iou = (intersection + eps) / (union + eps)
return iou.mean().item()


def dice_score(preds, targets, threshold=0.5, eps=1e-7):
preds = (preds > threshold).float()
targets = (targets > 0.5).float()
inter = (preds * targets).sum(dim=(1,2,3))
denom = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
dice = (2 * inter + eps) / (denom + eps)
return dice.mean().item()
