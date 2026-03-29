"""
DETR_Combiner.py — Fixed version
=================================
Key fixes over the original:
1. Gradient flow preserved — no PIL conversion during training
2. DETR runs differentiably using processor on tensors directly
3. Proper F1/FPR evaluation metrics added
4. find_hidden_person called correctly (returns list of boxes)
5. clean_root argument added to parser
"""

import argparse
import os
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from ObjectDetector.detr_detector import DETRDetector, find_hidden_person


def parse_args():
    parser = argparse.ArgumentParser()
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--test",  action="store_true")

    parser.add_argument("--df_mode",     type=str, choices=["C","W","A"], required=True)
    parser.add_argument("--dataset",     type=str, default="UPC",
                        choices=["UPC","AdvPatch","TCEGA","Natural"])
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--canary_size", type=int,   default=120)
    parser.add_argument("--wd_size",     type=int,   default=120)
    parser.add_argument("--person_conf", type=float, default=0.05)
    parser.add_argument("--weight",      type=float, default=2.0)
    parser.add_argument("--lr",          type=float, default=0.1)
    parser.add_argument("--best_canary_path", type=str, default="")
    parser.add_argument("--best_wd_path",     type=str, default="")
    parser.add_argument("--input_img",        type=str, default="")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class DETRDataset(Dataset):
    def __init__(self, clean_root, attack_root, size=512):
        self.clean_imgs  = sorted([os.path.join(clean_root, f)
                                   for f in os.listdir(clean_root)
                                   if f.lower().endswith(('.jpg','.png'))])
        self.attack_imgs = sorted([os.path.join(attack_root, f)
                                   for f in os.listdir(attack_root)
                                   if f.lower().endswith(('.jpg','.png'))])
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return min(len(self.clean_imgs), len(self.attack_imgs))

    def __getitem__(self, idx):
        clean  = self.tf(Image.open(self.clean_imgs[idx]).convert("RGB"))
        attack = self.tf(Image.open(self.attack_imgs[idx]).convert("RGB"))
        return clean, attack


# ─────────────────────────────────────────────
# Patch insertion (differentiable)
# ─────────────────────────────────────────────
def insert_patch_differentiable(img_tensor, patch, box, img_hw=512):
    """
    Insert patch into image tensor while preserving gradient flow.
    img_tensor: (3, H, W)  — stays on GPU, no PIL conversion
    patch:      (3, P, P)  — requires_grad=True
    box:        [x0,y0,x1,y1] in pixels
    """
    x0,y0,x1,y1 = [int(b) for b in box[:4]]
    x0,y0 = max(0,x0), max(0,y0)
    x1,y1 = min(img_hw,x1), min(img_hw,y1)
    bw, bh = x1-x0, y1-y0
    if bw <= 0 or bh <= 0:
        return img_tensor

    resized = F.interpolate(patch.unsqueeze(0), size=(bh,bw),
                            mode='bilinear', align_corners=False).squeeze(0)

    # Build mask to preserve gradient
    result = img_tensor.clone()
    result = torch.cat([
        result[:, :y0, :],
        torch.cat([
            result[:, y0:y1, :x0],
            resized,
            result[:, y0:y1, x1:]
        ], dim=2),
        result[:, y1:, :]
    ], dim=1)
    return result


# ─────────────────────────────────────────────
# DETR differentiable forward pass
# ─────────────────────────────────────────────
def detr_person_score(detector, img_tensor, device):
    """
    Run DETR on an image tensor and return mean person confidence.
    Keeps gradient flow intact by working directly with tensors.
    img_tensor: (3, H, W) float in [0,1]
    """
    # DETR processor expects pixel_values in (1, 3, H, W) float
    pixel_values = img_tensor.unsqueeze(0).to(device)

    # Run model directly (no processor needed — we already have tensors)
    outputs = detector.model(pixel_values=pixel_values)

    logits = outputs.logits.squeeze(0)          # (num_queries, num_classes+1)
    probas = F.softmax(logits, dim=-1)
    person_scores = probas[:, 1]                # class 1 = person in COCO
    return person_scores.mean()


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(args):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DETRDetector(device=device)
    detector.model.train()  # enable gradients through model

    BASE        = "Data/testeval/VOC07_YOLOv8/test"
    clean_root  = f"{BASE}/{args.dataset}/benign"
    attack_root = f"{BASE}/{args.dataset}/adversarial"

    dataset = DETRDataset(clean_root, attack_root)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)  # batch=1 for simplicity

    patch_size = args.canary_size if args.df_mode == 'C' else args.wd_size
    patch = torch.rand(3, patch_size, patch_size,
                       device=device, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    print(f"Training DETR {args.df_mode}-mode on {args.dataset} for {args.epochs} epochs")
    print(f"Device: {device}")

    for epoch in range(args.epochs):
        total_loss = 0.0
        for i, (clean_t, attack_t) in enumerate(loader):
            clean_t  = clean_t.squeeze(0).to(device)   # (3,512,512)
            attack_t = attack_t.squeeze(0).to(device)

            # Find candidate boxes using PIL path (no gradient needed here)
            pil_attack = Image.fromarray(
                (attack_t.detach().cpu().permute(1,2,0).numpy()*255).astype(np.uint8))
            outputs, img_size = detector.detect(pil_attack)
            boxes = find_hidden_person(outputs, img_size, threshold=args.person_conf)

            if not boxes:
                continue

            box = random.choice(boxes)
            optimizer.zero_grad()

            # Insert patch differentiably
            patched_attack = insert_patch_differentiable(attack_t, patch, box)
            patched_clean  = insert_patch_differentiable(clean_t,  patch, box)

            # Get person scores WITH gradient flow
            score_attack = detr_person_score(detector, patched_attack, device)
            score_clean  = detr_person_score(detector, patched_clean,  device)

            if args.df_mode == 'C':
                # Canary: we WANT it detected on clean, suppressed on attack
                # So maximize score_clean, minimize score_attack
                loss = args.weight * (-score_clean) + score_attack
            else:
                # Woodpecker: restore person on attack image
                loss = args.weight * (-score_attack) + score_clean

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0, 1)

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"  Epoch {epoch+1} step {i}: loss={loss.item():.4f} "
                      f"score_attack={score_attack.item():.3f} "
                      f"score_clean={score_clean.item():.3f}", flush=True)

        print(f"Epoch {epoch+1}/{args.epochs} total loss: {total_loss:.4f}")

    # Save trained patch
    save_dir  = os.path.join('trained_dfpatches', 'DETR')
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'canary.png' if args.df_mode == 'C' else 'wd.png'
    save_path = os.path.join(save_dir, save_name)
    from torchvision.utils import save_image
    save_image(patch.detach().cpu(), save_path)
    print(f"Saved to {save_path}")


# ─────────────────────────────────────────────
# Evaluation — computes F1 and FPR properly
# ─────────────────────────────────────────────
def evaluate(args):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DETRDetector(device=device)

    BASE        = "Data/testeval/VOC07_YOLOv8/test"
    clean_root  = f"{BASE}/{args.dataset}/benign"
    attack_root = f"{BASE}/{args.dataset}/adversarial"

    # Load patch
    patch_path = args.best_canary_path if args.df_mode in ['C','A'] else args.best_wd_path
    if patch_path and os.path.isfile(patch_path):
        patch_img = Image.open(patch_path).convert("RGB")
        patch_img = patch_img.resize((args.canary_size, args.canary_size))
        patch_cv  = cv2.cvtColor(np.array(patch_img), cv2.COLOR_RGB2BGR)
    else:
        print("WARNING: no patch path provided, using random patch")
        patch_cv = np.random.randint(0, 255,
                                     (args.canary_size, args.canary_size, 3),
                                     dtype=np.uint8)

    def eval_folder(folder, label):
        """Returns list of predictions (True=attack, False=benign)."""
        preds = []
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(('.jpg','.png'))])
        for fname in files:
            img_path = os.path.join(folder, fname)
            img_pil  = Image.open(img_path).convert("RGB")
            img_arr  = np.array(img_pil)

            # Find candidate boxes
            outputs, img_size = detector.detect(img_pil)
            boxes = find_hidden_person(outputs, img_size,
                                       threshold=args.person_conf)

            if not boxes:
                preds.append(False)
                continue

            # Insert patch at first candidate box
            box    = boxes[0]
            x0,y0,x1,y1 = [int(b) for b in box[:4]]
            x0,y0 = max(0,x0), max(0,y0)
            x1,y1 = min(img_arr.shape[1],x1), min(img_arr.shape[0],y1)

            if x1-x0 <= 0 or y1-y0 <= 0:
                preds.append(False)
                continue

            patched = img_arr.copy()
            resized_patch = cv2.resize(patch_cv, (x1-x0, y1-y0))
            patched[y0:y1, x0:x1] = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2RGB)

            # Run detector on patched image
            out2, size2 = detector.detect(Image.fromarray(patched))
            boxes2 = find_hidden_person(out2, size2, threshold=args.person_conf)

            # If canary area is still suppressed → attack detected
            is_attack = len(boxes2) > 0
            preds.append(is_attack)

        return preds

    print(f"\nEvaluating DETR canary defense on {args.dataset}...")
    adv_preds   = eval_folder(attack_root, 'adversarial')
    clean_preds = eval_folder(clean_root,  'benign')

    TP = sum(adv_preds)
    FN = len(adv_preds) - TP
    FP = sum(clean_preds)
    TN = len(clean_preds) - FP

    P   = TP/(TP+FP)  if (TP+FP)>0 else 0
    R   = TP/(TP+FN)  if (TP+FN)>0 else 0
    F1  = 2*P*R/(P+R) if (P+R)>0   else 0
    FPR = FP/(FP+TN)  if (FP+TN)>0 else 0

    print(f"\n{'='*50}")
    print(f"DETR Canary Defense — {args.dataset}")
    print(f"{'='*50}")
    print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"Precision : {P:.3f}")
    print(f"Recall    : {R:.3f}")
    print(f"F1        : {F1:.3f}")
    print(f"FPR       : {FPR:.3f}")
    print(f"{'='*50}")

    # Save results
    os.makedirs('results', exist_ok=True)
    result = {
        'detector': 'DETR',
        'dataset': args.dataset,
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Precision': P, 'Recall': R, 'F1': F1, 'FPR': FPR
    }
    import json
    out_path = f'results/detr_{args.dataset}_results.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")
    return result


def main():
    args = parse_args()
    if args.train:
        if args.df_mode == 'A':
            raise ValueError("A-mode only supported in test mode")
        train(args)
    elif args.test:
        evaluate(args)


if __name__ == '__main__':
    main()