# Fight Fire with Fire — COSC 739 Extension
**Khalifa University | COSC 739 Security of Machine Learning Systems**  
Leen Al-Jallad (100067697) · Mohamed Thabet (100066980)

---

## Overview

This repo extends the **Fight Fire with Fire (FFF)** adversarial patch detection framework ([Feng et al., IEEE S&P 2025](https://github.com/1j4j1230/Fight_Fire_with_Fire)). FFF defends person detectors against adversarial patches by inserting learned *canary patches* near candidate bounding boxes. If the canary is suppressed, an attack is flagged.

Our contributions:
1. **Baseline reproduction** — YOLOv8 on VOC07, 4 attack scenarios
2. **Canary seed ablation** — 5 seed images (zebra, elephant, gaussian, checkerboard, gradient)
3. **Failure mode analysis** — manual categorization of FN/FP across all 3 attacks
4. **Randomised deployment** — evaluates FFF's intended randomisation strategy with mean ± std
5. **Generative canary module** — proposed architecture (implementation pending)

---

## Environment Setup

**Requirements:** Ubuntu 22.04, Python 3.10, CUDA 12.1, GPU with ≥8GB VRAM

```bash
# Create and activate virtualenv
python3.10 -m venv ~/ffwf_env310
source ~/ffwf_env310/bin/activate

# Install dependencies
pip install torch==2.5.1+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics==8.0.145
pip install opencv-python numpy matplotlib

# CRITICAL: patch the ultralytics loss file (required every time after reinstall)
sed -i 's/self\.hyp\.box/self.hyp["box"]/g' ~/ffwf_env310/lib/python3.10/site-packages/ultralytics/utils/loss.py
sed -i 's/self\.hyp\.cls/self.hyp["cls"]/g' ~/ffwf_env310/lib/python3.10/site-packages/ultralytics/utils/loss.py
sed -i 's/self\.hyp\.dfl/self.hyp["dfl"]/g' ~/ffwf_env310/lib/python3.10/site-packages/ultralytics/utils/loss.py
```

**Activate environment (every session):**
```bash
source ~/ffwf_env310/bin/activate
```

---

## Dataset

VOC07 split provided by the FFF authors. Place under:
```
Data/testeval/VOC07_YOLOv8/test/
├── AdvPatch/adversarial/    (376 images)
├── AdvPatch/benign/         (376 images)
├── UPC/adversarial/         (73 images)
├── UPC/benign/              (73 images)
├── TCEGA/adversarial/       (95 images)
├── TCEGA/benign/            (95 images)
├── Natural/adversarial/     (205 images)
└── Natural/benign/          (205 images)
```

Training data: 120 images under `Data/` (see original FFF repo for download).

---

## Trained Canaries

All 5 trained canaries are in `FJNTraining/`:

| Folder | Seed | Avg F1 | TCEGA F1 |
|--------|------|--------|----------|
| `canary_zebra/` | Zebra (COCO class 22) | 0.914 | 0.836 |
| `canary_elephant/` | Elephant (COCO class 20) | **0.942** | **0.931** |
| `canary_gaussian/` | Gaussian noise | 0.933 | 0.922 |
| `canary_checkerboard_correct/` | Checkerboard | 0.932 | 0.922 |
| `canary_gradient_correct/` | Solid gradient | 0.935 | 0.895 |

Each canary is at: `FJNTraining/canary_NAME/exp_VOC07_120_22_80_50/canary_050.png`

---

## Training a New Canary

```bash
# 1. Set your seed image as InitImages/22.jpg
cp InitImages/20.jpg InitImages/22.jpg  # example: elephant seed

# 2. Protect any existing canary_zebra folder first
mv FJNTraining/canary_zebra FJNTraining/canary_zebra_backup

# 3. Train
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python YOLOv8_Combiner.py \
  --train --df_mode C --defensive_patch_location cc \
  --canary_cls_id 22 --canary_size 80 --person_conf 0.05 \
  --weight 2.0 --batch_size 5

# 4. Rename output folder
mv FJNTraining/canary_zebra FJNTraining/canary_YOURNAME

# Takes ~5 minutes on RTX 4060
```

**Key parameters:**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--canary_cls_id` | 22 | Zebra class in COCO (what detector looks for) |
| `--canary_size` | 80 | 80×80 pixel canary patch |
| `--person_conf` | 0.05 | τ threshold — suppress score below this = attack flagged |
| `--weight` | 2.0 | λ_c in loss (canary weight vs woodpecker weight) |
| `--batch_size` | 5 | Reduce if OOM, increase if you have more VRAM |

---

## Evaluation Scripts

All scripts are in `our_scripts/`. Run from repo root.

### 1. Full Ablation (all 5 canaries × 4 attacks)
```bash
python our_scripts/evaluate_fast.py
```
Outputs a table of F1/FPR/TP/FP/FN/TN for every canary × attack combination.
Uses checkpoint files (`checkpoints/fast_progress_*.json`) — safe to interrupt and resume.

**To add a new canary**, edit the `CANARIES` dict at the top of `evaluate_fast.py`:
```python
CANARIES = {
    ...
    'your_canary': './FJNTraining/canary_YOURNAME/exp_VOC07_120_22_80_50/canary_050.png',
}
```

### 2. Randomised Deployment (FFF intended strategy)
```bash
python our_scripts/evaluate_random.py
```
Runs 3 independent trials. Each image gets a randomly chosen canary from the pool
and a randomly chosen placement position (cc/uc/bc/cl/cr — centre and ±30px offsets).
Reports mean ± std across trials. Results saved to `results/random_trial_results.json`.

### 3. Failure Analysis (which specific images fail)
```bash
python our_scripts/eval_failures.py
```
Runs zebra canary on AdvPatch and UPC, saves FN/FP filenames to:
- `results/failures_advpatch.json`
- `results/failures_upc.json`

### 4. Placement Grid Figure
```bash
python our_scripts/make_placement_grid.py
```
Generates `figures/placement_grid.png` — a 5×5 grid showing each canary at each placement position.

---

## Key Results Summary

### Phase 1: Baseline Reproduction
| Attack | Paper F1 | Our F1 | Our FPR |
|--------|----------|--------|---------|
| AdvPatch | 0.974 | 0.966 | 0.051 |
| UPC | 0.936 | 0.951 | 0.027 |
| TCEGA | 0.807 | 0.836 | 0.084 |
| Natural | 0.871 | 0.905 | 0.063 |

### Phase 2: Best Canary per Attack
| Attack | Best seed | F1 |
|--------|-----------|-----|
| AdvPatch | Zebra | 0.966 |
| UPC | Gradient | 0.946 |
| TCEGA | **Elephant** | **0.931** |
| Natural | Elephant / Gradient | 0.940 |

### Phase 3: Randomised Deployment (3 trials)
| Attack | F1 (mean±std) | FPR (mean±std) |
|--------|---------------|----------------|
| AdvPatch | 0.947±0.006 | 0.087±0.011 |
| UPC | 0.916±0.017 | 0.087±0.034 |
| TCEGA | 0.916±0.027 | 0.070±0.018 |
| Natural | 0.923±0.001 | 0.060±0.010 |

---

## Failure Mode Analysis

Manual review of all FNs across 3 attacks (33 total):

| Category | AdvPatch | UPC | TCEGA | Total |
|----------|----------|-----|-------|-------|
| Vehicle/animal occlusion | 4 | 1 | 14 | **19 (58%)** |
| Non-standard pose | 2 | 2 | 3 | 7 (21%) |
| Group/partial occlusion | 1 | 0 | 3 | 4 (12%) |
| Other | 0 | 2 | 1 | 3 (9%) |

**Root cause:** Canary placement assumes an upright pedestrian bounding box.
When a person rides a horse or motorcycle, the bounding box merges rider + vehicle,
causing wrong canary placement.

---

## Next Steps (Level A)

### Option A: Generative Canary Module (proposed, not yet implemented)

Replace the fixed canary with a learned generator that produces a unique canary per image:

```
z = [η(32-dim noise) || b(4-dim bbox)] ∈ R^36
→ FC(256) → FC(512) → FC(1024) → reshape(16×8×8)
→ ConvT(16×20×20) → ConvT(3×80×80) → sigmoid
→ canary patch 80×80×3
```

**To implement:**
1. Write `our_scripts/generative_canary.py` with the generator class
2. Modify `YOLOv8_Combiner.py` to call `G(z, bbox)` instead of loading a fixed canary image
3. Train end-to-end with same loss (λ_c=2.0, λ_w=1.0)
4. Evaluate: run `evaluate_fast.py` adapted for the generator
5. Adaptive attack: optimize a patch against fixed zebra, test it against the generator

**Expected result:** Generator F1 similar to fixed canary, but adaptive attacks
that were trained against zebra should fail against the generator.

### Option B: Pose-guided Canary Placement (lower risk, directly fixes identified failure)

Use MediaPipe pose estimation to place canary on torso keypoint instead of bbox centre:

```bash
pip install mediapipe --break-system-packages
```

Modify `add_defensivepatch_into_img()` in `YOLOv8_Combiner.py` to use
torso keypoint (shoulder midpoint) as anchor instead of bbox centre.
Re-evaluate on the 19 vehicle/animal occlusion FN images — expect significant improvement.

---

## File Structure

```
Fight_Fire_with_Fire/
├── YOLOv8_Combiner.py          # Main FFF training/eval code (original)
├── ObjectDetector/             # YOLOv8 wrapper (original)
├── InitImages/                 # Canary seed images
│   ├── 20.jpg                  # Elephant seed
│   ├── 22.jpg                  # Current seed (check what's in here)
│   ├── gaussian_noise.jpg
│   ├── checkerboard.jpg
│   └── gradient.jpg
├── FJNTraining/                # Trained canary checkpoints
│   ├── canary_zebra/
│   ├── canary_elephant/
│   ├── canary_gaussian/
│   ├── canary_checkerboard_correct/
│   └── canary_gradient_correct/
├── Data/                       # VOC07 dataset
├── figures/                    # Generated figures for paper
├── checkpoints/                # Evaluation progress (resumable)
├── results/                    # Failure analysis JSON outputs
└── our_scripts/                # Our evaluation scripts
    ├── evaluate_fast.py        # Main ablation evaluator
    ├── evaluate_random.py      # Randomised deployment evaluator
    ├── eval_failures.py        # FN/FP failure analysis
    └── make_placement_grid.py  # Placement grid figure generator
```

---

## Common Issues

**OOM during training:**
```bash
# Reduce batch size
--batch_size 3
```

**ultralytics loss error (`hyp.box` AttributeError):**
```bash
# Re-apply the patch
sed -i 's/self\.hyp\.box/self.hyp["box"]/g' ~/ffwf_env310/lib/python3.10/site-packages/ultralytics/utils/loss.py
sed -i 's/self\.hyp\.cls/self.hyp["cls"]/g' ~/ffwf_env310/lib/python3.10/site-packages/ultralytics/utils/loss.py
sed -i 's/self\.hyp\.dfl/self.hyp["dfl"]/g' ~/ffwf_env310/lib/python3.10/site-packages/ultralytics/utils/loss.py
```

**Checkpoint files out of sync:**
```bash
# Delete checkpoints for the affected canary/attack and re-run
rm checkpoints/fast_progress_CANARYNAME_ATTACK.json
```

**22.jpg is corrupted (it was accidentally overwritten with gradient):**
```bash
# Use elephant as semantic seed instead
cp InitImages/20.jpg InitImages/22.jpg
```
