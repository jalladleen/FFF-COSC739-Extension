"""
evaluate_tau_sweep.py
Evaluates detection performance across different values of tau (person_conf).

The FFF paper sets tau=0.05 and treats it as a fixed hyperparameter.
This script tests whether tau=0.05 is actually optimal across all attack types,
or whether different attacks benefit from different thresholds.

tau controls what counts as a "candidate box":
- Lower tau = more candidate boxes found (higher recall, more FP)
- Higher tau = fewer candidate boxes (lower recall, fewer FP)

We test: tau in {0.01, 0.025, 0.05, 0.075, 0.10}
Using the zebra canary (baseline) to isolate the effect of tau alone.
"""
import os, sys, json, cv2, copy
import numpy as np
from copy import deepcopy

CANARY_PATH = './FJNTraining/canary_zebra/exp_VOC07_120_22_80_50/canary_050.png'

TAU_VALUES = [0.01, 0.025, 0.05, 0.075, 0.10]

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

def compute_metrics(TP, FP, FN, TN):
    p   = TP/(TP+FP)  if (TP+FP)>0 else 0
    r   = TP/(TP+FN)  if (TP+FN)>0 else 0
    f1  = 2*p*r/(p+r) if (p+r)>0   else 0
    fpr = FP/(FP+TN)  if (FP+TN)>0 else 0
    return p, r, f1, fpr

def ckpt_file(tau, attack):
    return f'checkpoints/tau_progress_{tau}_{attack}.json'

def load_ckpt(tau, attack):
    f = ckpt_file(tau, attack)
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return {'TP':0,'FP':0,'FN':0,'TN':0,'done_adv':[],'done_benign':[]}

def save_ckpt(tau, attack, prog):
    with open(ckpt_file(tau, attack), 'w') as fp:
        json.dump(prog, fp)

def eval_with_tau(tau, canary_obj):
    """Evaluate all attacks with a specific tau value."""
    # Override person_conf on the canary object
    canary_obj.person_conf = tau
    print(f'\n=== tau = {tau} ===', flush=True)

    results = {}
    for attack_name, (adv_dir, ben_dir) in ATTACKS.items():
        prog = load_ckpt(tau, attack_name)
        adv_imgs = sorted([f for f in os.listdir(adv_dir) if f.endswith(('.jpg','.png'))])
        ben_imgs = sorted([f for f in os.listdir(ben_dir) if f.endswith(('.jpg','.png'))])
        done = len(prog['done_adv']) + len(prog['done_benign'])
        total = len(adv_imgs) + len(ben_imgs)
        print(f'  [{attack_name}] {done}/{total} already done', flush=True)

        for fname in adv_imgs:
            if fname in prog['done_adv']:
                continue
            img = cv2.imread(os.path.join(adv_dir, fname), 1)
            if img is None:
                continue
            try:
                is_attack = canary_obj.eval_single(img)
                prog['TP' if is_attack else 'FN'] += 1
            except Exception as e:
                print(f'    ERROR {fname}: {e}', flush=True)
            prog['done_adv'].append(fname)
            save_ckpt(tau, attack_name, prog)

        for fname in ben_imgs:
            if fname in prog['done_benign']:
                continue
            img = cv2.imread(os.path.join(ben_dir, fname), 1)
            if img is None:
                continue
            try:
                is_attack = canary_obj.eval_single(img)
                prog['FP' if is_attack else 'TN'] += 1
            except Exception as e:
                print(f'    ERROR {fname}: {e}', flush=True)
            prog['done_benign'].append(fname)
            save_ckpt(tau, attack_name, prog)

        TP,FP,FN,TN = prog['TP'],prog['FP'],prog['FN'],prog['TN']
        p,r,f1,fpr = compute_metrics(TP,FP,FN,TN)
        print(f'  [{attack_name}] TP:{TP} FP:{FP} FN:{FN} TN:{TN} '
              f'P:{p:.3f} R:{r:.3f} F1:{f1:.3f} FPR:{fpr:.3f}', flush=True)
        results[attack_name] = {'TP':TP,'FP':FP,'FN':FN,'TN':TN}

    return results

if __name__ == '__main__':
    # Import here so sys.argv override works
    sys.argv = ['eval', '--test', '--df_mode', 'C', '--defensive_patch_location', 'cc',
                '--canary_cls_id', '22', '--canary_size', '80', '--person_conf', '0.05',
                '--weight', '2.0', '--best_canary_path', 'dummy', '--input_img', 'dummy']
    from YOLOv8_Combiner import get_args, Canary, freeze_seed
    from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8

    print('Loading model (once for all tau values)...', flush=True)
    cfg = get_args()
    freeze_seed(cfg.seed)
    detector = YOLOv8()
    canary_obj = Canary(cfg, detector)
    canary_obj.eval_load_canary(canary_path=CANARY_PATH, canary_cls_id=22)

    # Run sweep
    all_results = {}
    for tau in TAU_VALUES:
        all_results[tau] = eval_with_tau(tau, canary_obj)

    # Save
    with open('results/tau_sweep_results.json', 'w') as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    # Print summary table
    attacks = list(ATTACKS.keys())
    print('\n\n========== TAU SWEEP RESULTS (Zebra Canary) ==========')
    print(f"{'tau':<8}", end='')
    for atk in attacks:
        print(f"  {atk+' F1':>12} {atk+' FPR':>10}", end='')
    print(f"  {'Avg F1':>8} {'Avg FPR':>8}")
    print('-' * (8 + len(attacks)*24 + 18))

    paper_tau = 0.05
    for tau in TAU_VALUES:
        marker = ' ← paper' if tau == paper_tau else ''
        f1s, fprs = [], []
        print(f"{tau:<8}", end='')
        for atk in attacks:
            d = all_results[tau][atk]
            p,r,f1,fpr = compute_metrics(d['TP'],d['FP'],d['FN'],d['TN'])
            f1s.append(f1); fprs.append(fpr)
            print(f"  {f1:>12.3f} {fpr:>10.3f}", end='')
        print(f"  {sum(f1s)/4:>8.3f} {sum(fprs)/4:>8.3f}{marker}")

    print('\nKey question: Is tau=0.05 optimal for ALL attacks, or does each attack')
    print('benefit from a different threshold?')
    print('\nResults saved to results/tau_sweep_results.json')