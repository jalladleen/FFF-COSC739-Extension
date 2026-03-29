"""
evaluate_random.py
Randomised canary deployment — matches FFF paper's intended inference strategy.

For each candidate box per image:
  - Canary drawn uniformly at random from pool of 5 trained canaries
  - Placement drawn uniformly from the 5 paper positions:
    cc (centre), uc (up), bc (down), cl (left), cr (right)
    each offset ±30px from candidate box centre, plus ±10px jitter (built-in)

N_TRIALS independent trials, report mean ± std.
"""
import os, sys, json, cv2, random
import numpy as np
from copy import deepcopy

sys.argv = ['eval', '--test', '--df_mode', 'C', '--defensive_patch_location', 'cc',
            '--canary_cls_id', '22', '--canary_size', '80', '--person_conf', '0.05',
            '--weight', '2.0', '--best_canary_path', 'dummy', '--input_img', 'dummy']

from YOLOv8_Combiner import get_args, Canary, freeze_seed
from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8

CANARY_POOL = {
    'zebra':        './FJNTraining/canary_zebra/exp_VOC07_120_22_80_50/canary_050.png',
    'elephant':     './FJNTraining/canary_elephant/exp_VOC07_120_22_80_50/canary_050.png',
    'gaussian':     './FJNTraining/canary_gaussian/exp_VOC07_120_22_80_50/canary_050.png',
    'checkerboard': './FJNTraining/canary_checkerboard_correct/exp_VOC07_120_22_80_50/canary_050.png',
    'gradient':     './FJNTraining/canary_gradient_correct/exp_VOC07_120_22_80_50/canary_050.png',
}

# 5 positions: centre + 4 cardinal offsets of ±30px from box centre
PLACEMENT_POOL = ['cc', 'uc', 'bc', 'cl', 'cr']

ATTACKS = {
    'AdvPatch': ('Data/testeval/VOC07_YOLOv8/test/AdvPatch/adversarial',
                 'Data/testeval/VOC07_YOLOv8/test/AdvPatch/benign'),
    'UPC':      ('Data/testeval/VOC07_YOLOv8/test/UPC/adversarial',
                 'Data/testeval/VOC07_YOLOv8/test/UPC/benign'),
    'TCEGA':    ('Data/testeval/VOC07_YOLOv8/test/TCEGA/adversarial',
                 'Data/testeval/VOC07_YOLOv8/test/TCEGA/benign'),
    'Natural':  ('Data/testeval/VOC07_YOLOv8/test/Natural/adversarial',
                 'Data/testeval/VOC07_YOLOv8/test/Natural/benign'),
}

N_TRIALS = 3
RANDOM_SEED_BASE = 42

def compute_metrics(TP, FP, FN, TN):
    p   = TP/(TP+FP)  if (TP+FP)>0 else 0
    r   = TP/(TP+FN)  if (TP+FN)>0 else 0
    f1  = 2*p*r/(p+r) if (p+r)>0   else 0
    fpr = FP/(FP+TN)  if (FP+TN)>0 else 0
    return p, r, f1, fpr

def run_trial(trial_idx, canary_obj, canary_images):
    """
    For each image, randomly pick canary image AND placement position,
    swap both onto the canary object, then call eval_single normally.
    """
    rng = random.Random(RANDOM_SEED_BASE + trial_idx)
    canary_names = list(canary_images.keys())
    results = {atk: {'TP':0,'FP':0,'FN':0,'TN':0} for atk in ATTACKS}

    for attack_name, (adv_dir, ben_dir) in ATTACKS.items():
        adv_imgs = sorted([f for f in os.listdir(adv_dir) if f.endswith(('.jpg','.png'))])
        ben_imgs = sorted([f for f in os.listdir(ben_dir) if f.endswith(('.jpg','.png'))])
        print(f'  Trial {trial_idx+1} [{attack_name}] {len(adv_imgs)+len(ben_imgs)} images...', flush=True)

        for fname in adv_imgs:
            img = cv2.imread(os.path.join(adv_dir, fname), 1)
            if img is None:
                continue
            canary_obj.eval_canary = canary_images[rng.choice(canary_names)]
            canary_obj.cfg.defensive_patch_location = rng.choice(PLACEMENT_POOL)
            try:
                is_attack = canary_obj.eval_single(img)
                results[attack_name]['TP' if is_attack else 'FN'] += 1
            except Exception as e:
                print(f'    ERROR {fname}: {e}', flush=True)

        for fname in ben_imgs:
            img = cv2.imread(os.path.join(ben_dir, fname), 1)
            if img is None:
                continue
            canary_obj.eval_canary = canary_images[rng.choice(canary_names)]
            canary_obj.cfg.defensive_patch_location = rng.choice(PLACEMENT_POOL)
            try:
                is_attack = canary_obj.eval_single(img)
                results[attack_name]['FP' if is_attack else 'TN'] += 1
            except Exception as e:
                print(f'    ERROR {fname}: {e}', flush=True)

        TP,FP,FN,TN = (results[attack_name][k] for k in ('TP','FP','FN','TN'))
        p,r,f1,fpr = compute_metrics(TP,FP,FN,TN)
        print(f'    TP:{TP} FP:{FP} FN:{FN} TN:{TN} '
              f'P:{p:.3f} R:{r:.3f} F1:{f1:.3f} FPR:{fpr:.3f}', flush=True)

    return results

if __name__ == '__main__':
    print('Loading model and canaries...', flush=True)
    cfg = get_args()
    freeze_seed(cfg.seed)
    detector = YOLOv8()

    canary_obj = Canary(cfg, detector)
    canary_obj.eval_load_canary(
        canary_path=list(CANARY_POOL.values())[0],
        canary_cls_id=22
    )

    canary_images = {}
    for name, path in CANARY_POOL.items():
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (80, 80))
        canary_images[name] = img
        print(f'  Loaded: {name}', flush=True)

    trial_results = []
    for t in range(N_TRIALS):
        print(f'\n=== Trial {t+1}/{N_TRIALS} ===', flush=True)
        trial_results.append(run_trial(t, canary_obj, canary_images))

    with open('random_trial_results.json', 'w') as f:
        json.dump(trial_results, f, indent=2)

    print('\n\n========== RANDOMISED DEPLOYMENT RESULTS (N=3 trials) ==========')
    print(f"{'Attack':<12} {'Prec':>12} {'Rec':>12} {'F1':>12} {'FPR':>12}")
    print('-'*60)
    for attack_name in ATTACKS:
        ps, rs, f1s, fprs = [], [], [], []
        for t in range(N_TRIALS):
            d = trial_results[t][attack_name]
            p,r,f1,fpr = compute_metrics(d['TP'],d['FP'],d['FN'],d['TN'])
            ps.append(p); rs.append(r); f1s.append(f1); fprs.append(fpr)
        print(f"{attack_name:<12} "
              f"{np.mean(ps):.3f}±{np.std(ps):.3f}  "
              f"{np.mean(rs):.3f}±{np.std(rs):.3f}  "
              f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
              f"{np.mean(fprs):.3f}±{np.std(fprs):.3f}")

    print('\nRaw per-trial data saved to random_trial_results.json')
