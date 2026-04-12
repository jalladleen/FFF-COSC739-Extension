"""Standalone evaluation script for the generative canary ensemble.

Loads all generator checkpoints from a directory, runs the full VOC07 evaluation
(AdvPatch, UPC, TCEGA, Natural) with a fresh randomly-picked generator per test
image, and prints F1/FPR plus a comparison table against published baselines.

Usage:
    python evaluate_ensemble.py
    python evaluate_ensemble.py --checkpoint_dir ./ensemble_1000/checkpoints
    python evaluate_ensemble.py --max_generators 100  # eval with a subset

Runs locally on RTX 3070 Ti (8 GB). 1000 generators * ~2.7 MB = ~2.7 GB GPU RAM,
fits alongside YOLOv8's ~500 MB. Falls back to CPU-resident ensemble if OOM.
"""
import argparse
import glob
import json
import os
import random
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, default='./ensemble_1000/checkpoints')
    p.add_argument('--test_root', type=str,
                   default='../Data/testeval/VOC07_YOLOv8/test')
    p.add_argument('--output_json', type=str, default='ensemble_1000_eval_results.json')
    p.add_argument('--z_dim', type=int, default=128)
    p.add_argument('--canary_cls_id', type=int, default=22)
    p.add_argument('--max_generators', type=int, default=0,
                   help='0 = load all')
    p.add_argument('--seed', type=int, default=301)
    return p.parse_args()


def main():
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(here, '..')))
    sys.path.insert(0, here)

    import cv2
    import numpy as np
    import torch
    from types import SimpleNamespace
    from tqdm import tqdm

    from YOLOv8_Combiner import Canary
    from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8
    from generative_canary import GenerativeCanary

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Discover and load checkpoints ----
    ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'gen_*.pt')))
    if args.max_generators > 0:
        ckpt_paths = ckpt_paths[:args.max_generators]

    if not ckpt_paths:
        print(f"ERROR: no checkpoints found in {args.checkpoint_dir}")
        sys.exit(1)

    print(f"Loading {len(ckpt_paths)} generators from {args.checkpoint_dir} ...")

    # Try GPU first, fall back to CPU if OOM
    ensemble_on_gpu = True
    ensemble = []
    try:
        for i, cp in enumerate(ckpt_paths):
            g = GenerativeCanary(z_dim=args.z_dim).to(device)
            ckpt = torch.load(cp, map_location=device)
            g.load_state_dict(ckpt['generator_state_dict'])
            g.eval()
            ensemble.append(g)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(ckpt_paths)}")
    except torch.cuda.OutOfMemoryError:
        print("  GPU OOM — falling back to CPU-resident ensemble (lazy transfer)")
        for g in ensemble:
            g.cpu()
        ensemble_on_gpu = False
        for i, cp in enumerate(ckpt_paths[len(ensemble):], start=len(ensemble)):
            g = GenerativeCanary(z_dim=args.z_dim)
            ckpt = torch.load(cp, map_location='cpu')
            g.load_state_dict(ckpt['generator_state_dict'])
            g.eval()
            ensemble.append(g)

    print(f"Ensemble ready: {len(ensemble)} generators "
          f"(on {'GPU' if ensemble_on_gpu else 'CPU'})")

    # ---- Load detector ----
    print("Loading YOLOv8 detector...")
    detector = YOLOv8()
    detector.model.eval()

    eval_cfg = SimpleNamespace(
        canary_cls_id=args.canary_cls_id, canary_size=80, img_size=640,
        person_conf=0.05, overlap_thresh=0.4, defensive_patch_location='cc',
        eval_no_overlap=True, margin_size=0, faster=False,
    )
    canary_eval = Canary(eval_cfg, detector)

    # ---- Evaluate ----
    attacks = ['AdvPatch', 'UPC', 'TCEGA', 'Natural']
    results = {}
    rng = random.Random(args.seed)

    print(f"\n========== Ensemble Eval ({len(ensemble)} generators, random pick per image) ==========")

    def sample_canary():
        g = rng.choice(ensemble)
        eta = torch.randn(1, args.z_dim, device=device)
        bbox = torch.rand(1, 4, device=device)
        z = torch.cat([eta, bbox], dim=1)
        if not ensemble_on_gpu:
            g_gpu = g.to(device)
            with torch.no_grad():
                patch = g_gpu(z)
            g.cpu()
        else:
            with torch.no_grad():
                patch = g(z)
        return patch

    for attack in attacks:
        adv_dir = os.path.join(args.test_root, attack, 'adversarial')
        ben_dir = os.path.join(args.test_root, attack, 'benign')

        if not os.path.isdir(adv_dir) or not os.path.isdir(ben_dir):
            print(f"  SKIP {attack}: directory missing")
            continue

        TP = FP = FN = TN = 0

        adv_files = sorted([f for f in os.listdir(adv_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in tqdm(adv_files, desc=f'{attack} adv', leave=False):
            patch = sample_canary()
            canary_np = (patch[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canary_eval.eval_canary = cv2.cvtColor(canary_np, cv2.COLOR_RGB2BGR)
            canary_eval.canary_cls_id = args.canary_cls_id
            img = cv2.imread(os.path.join(adv_dir, fname), 1)
            if canary_eval.eval_single(img):
                TP += 1
            else:
                FN += 1

        ben_files = sorted([f for f in os.listdir(ben_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in tqdm(ben_files, desc=f'{attack} ben', leave=False):
            patch = sample_canary()
            canary_np = (patch[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            canary_eval.eval_canary = cv2.cvtColor(canary_np, cv2.COLOR_RGB2BGR)
            canary_eval.canary_cls_id = args.canary_cls_id
            img = cv2.imread(os.path.join(ben_dir, fname), 1)
            if canary_eval.eval_single(img):
                FP += 1
            else:
                TN += 1

        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        results[attack] = {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'Precision': P, 'Recall': R, 'F1': F1, 'FPR': FPR,
        }
        print(f"  {attack:10s}  F1={F1:.3f}  FPR={FPR:.3f}  "
              f"(TP={TP} FP={FP} FN={FN} TN={TN})")

    # ---- Comparison table ----
    paper = {'AdvPatch': 0.974, 'UPC': 0.936, 'TCEGA': 0.807, 'Natural': 0.871}
    fixed17 = {'AdvPatch': 0.935, 'UPC': 0.877, 'TCEGA': 0.896, 'Natural': 0.900}
    ens5 = {'AdvPatch': 0.935, 'UPC': 0.908, 'TCEGA': 0.905, 'Natural': 0.929}

    print("\n" + "=" * 82)
    print(f"{'Attack':10s} | {'Paper':>7s} | {'Fixed17':>7s} | {'5-gen':>7s} | "
          f"{'K-gen F1':>8s} | {'K-gen FPR':>9s} | {'Δ vs Paper':>10s}")
    print("-" * 82)
    for attack in attacks:
        if attack not in results:
            continue
        r = results[attack]
        delta = r['F1'] - paper[attack]
        print(f"{attack:10s} | {paper[attack]:>7.3f} | {fixed17[attack]:>7.3f} | "
              f"{ens5[attack]:>7.3f} | {r['F1']:>8.3f} | {r['FPR']:>9.3f} | "
              f"{delta:>+10.3f}")
    print("=" * 82)

    # ---- Save JSON ----
    out_path = os.path.abspath(args.output_json)
    with open(out_path, 'w') as f:
        json.dump({
            'n_generators': len(ensemble),
            'attacks': results,
            'baselines': {'paper': paper, 'fixed17': fixed17, 'ensemble_5': ens5},
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
