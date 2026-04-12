"""Full VOC07 evaluation for the ensemble of 5 generators.

Loads existing checkpoints from mode_collapse_exp/ensemble/checkpoints/ — no retraining.
For each test image, randomly picks one of the 5 generators and samples a fresh canary.
"""
import sys, os, cv2, torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))

from YOLOv8_Combiner import Canary
from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8

# ========== GenerativeCanary (must match cell-5 definition) ==========
class GenerativeCanary(nn.Module):
    def __init__(self, z_dim=128, bbox_dim=4, canary_size=80):
        super().__init__()
        self.z_dim = z_dim
        self.canary_size = canary_size
        input_dim = z_dim + bbox_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 1024), nn.ReLU(inplace=True),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 16, 8, 8)
        x = self.deconv(x)
        x = F.interpolate(x, size=(self.canary_size, self.canary_size), mode='bilinear', align_corners=False)
        return x


def main():
    Z_DIM = 128
    N_GENERATORS = 5
    canary_cls_id = 22
    ckpt_dir = os.path.join(HERE, 'mode_collapse_exp', 'ensemble', 'checkpoints')
    test_root = os.path.abspath(os.path.join(HERE, '..', 'Data', 'testeval', 'VOC07_YOLOv8', 'test'))
    attacks = ['AdvPatch', 'UPC', 'TCEGA', 'Natural']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load ensemble from checkpoints ----
    print("Loading 5 generators from checkpoints...")
    ensemble = []
    for gen_idx in range(N_GENERATORS):
        ckpt_path = os.path.join(ckpt_dir, f'gen_{gen_idx}_epoch_050.pt')
        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
        g = GenerativeCanary(z_dim=Z_DIM).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        g.load_state_dict(ckpt['generator_state_dict'])
        g.eval()
        ensemble.append(g)
        print(f"  Loaded gen_{gen_idx} (epoch {ckpt['epoch']})")

    # ---- Load detector ----
    print("Loading YOLOv8 detector...")
    detector = YOLOv8()
    detector.model.eval()

    eval_cfg = SimpleNamespace(
        canary_cls_id=canary_cls_id, canary_size=80, img_size=640,
        person_conf=0.05, overlap_thresh=0.4, defensive_patch_location='cc',
        eval_no_overlap=True, margin_size=0, faster=False,
    )
    canary_eval = Canary(eval_cfg, detector)

    # ---- Evaluate with per-image random generator pick ----
    print("\n========== Ensemble Eval (random generator per image) ==========")
    results = {}
    rng = random.Random(301)  # deterministic for reproducibility

    for attack in attacks:
        adv_dir = os.path.join(test_root, attack, 'adversarial')
        ben_dir = os.path.join(test_root, attack, 'benign')
        TP = FP = FN = TN = 0

        # Adversarial
        adv_files = sorted([f for f in os.listdir(adv_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in tqdm(adv_files, desc=f'{attack} adv', leave=False):
            g = rng.choice(ensemble)
            eta = torch.randn(1, Z_DIM, device=device)
            bbox = torch.rand(1, 4, device=device)
            z = torch.cat([eta, bbox], dim=1)
            with torch.no_grad():
                patch = g(z)
            canary_np = (patch[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canary_eval.eval_canary = cv2.cvtColor(canary_np, cv2.COLOR_RGB2BGR)
            canary_eval.canary_cls_id = canary_cls_id

            img = cv2.imread(os.path.join(adv_dir, fname), 1)
            if canary_eval.eval_single(img):
                TP += 1
            else:
                FN += 1

        # Benign
        ben_files = sorted([f for f in os.listdir(ben_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in tqdm(ben_files, desc=f'{attack} ben', leave=False):
            g = rng.choice(ensemble)
            eta = torch.randn(1, Z_DIM, device=device)
            bbox = torch.rand(1, 4, device=device)
            z = torch.cat([eta, bbox], dim=1)
            with torch.no_grad():
                patch = g(z)
            canary_np = (patch[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canary_eval.eval_canary = cv2.cvtColor(canary_np, cv2.COLOR_RGB2BGR)
            canary_eval.canary_cls_id = canary_cls_id

            img = cv2.imread(os.path.join(ben_dir, fname), 1)
            if canary_eval.eval_single(img):
                FP += 1
            else:
                TN += 1

        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        results[attack] = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'F1': F1, 'FPR': FPR}
        print(f"  {attack:10s}  F1={F1:.3f}  FPR={FPR:.3f}  (TP={TP} FP={FP} FN={FN} TN={TN})")

    # ---- Comparison table ----
    paper = {'AdvPatch': 0.974, 'UPC': 0.936, 'TCEGA': 0.807, 'Natural': 0.871}
    print("\n" + "=" * 60)
    print(f"{'Attack':10s} | {'Paper F1':>9s} | {'Ensemble F1':>12s} | {'Ensemble FPR':>13s} | {'Delta':>8s}")
    print("-" * 60)
    for attack in attacks:
        r = results[attack]
        delta = r['F1'] - paper[attack]
        print(f"{attack:10s} | {paper[attack]:>9.3f} | {r['F1']:>12.3f} | {r['FPR']:>13.3f} | {delta:>+8.3f}")
    print("=" * 60)

    # Save JSON for later
    import json
    out_path = os.path.join(HERE, 'ensemble_eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
