import os, sys, json, cv2, copy
import numpy as np

sys.argv = ['eval', '--test', '--df_mode', 'C', '--defensive_patch_location', 'cc',
            '--canary_cls_id', '22', '--canary_size', '80', '--person_conf', '0.05',
            '--weight', '2.0', '--best_canary_path', 'dummy', '--input_img', 'dummy']

from YOLOv8_Combiner import get_args, Canary, freeze_seed
from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8

CANARIES = {
    'gaussian':     './FJNTraining/canary_gaussian/exp_VOC07_120_22_80_50/canary_050.png',
    'checkerboard': './FJNTraining/canary_checkerboard_correct/exp_VOC07_120_22_80_50/canary_050.png',
    'gradient':     './FJNTraining/canary_gradient_correct/exp_VOC07_120_22_80_50/canary_050.png',
    'elephant':    './FJNTraining/canary_elephant/exp_VOC07_120_22_80_50/canary_050.png',
    'elephant':    './FJNTraining/canary_elephant/exp_VOC07_120_22_80_50/canary_050.png',
}

ZEBRA_RESULTS = {
    'AdvPatch': {'TP': 369, 'FP': 19, 'FN': 7,  'TN': 357},
    'UPC':      {'TP': 68,  'FP': 2,  'FN': 5,  'TN': 71},
    'TCEGA':    {'TP': 74,  'FP': 8,  'FN': 21, 'TN': 87},
    'Natural':  {'TP': 180, 'FP': 13, 'FN': 25, 'TN': 192},
}

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
    p  = TP/(TP+FP) if (TP+FP)>0 else 0
    r  = TP/(TP+FN) if (TP+FN)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    fpr= FP/(FP+TN) if (FP+TN)>0 else 0
    return f1, fpr

def ckpt_file(canary_name, attack_name):
    return f'fast_progress_{canary_name}_{attack_name}.json'

def load_ckpt(canary_name, attack_name):
    f = ckpt_file(canary_name, attack_name)
    if os.path.exists(f):
        with open(f) as fp:
            return json.load(fp)
    return {'TP':0,'FP':0,'FN':0,'TN':0,'done_adv':[],'done_benign':[]}

def save_ckpt(canary_name, attack_name, prog):
    with open(ckpt_file(canary_name, attack_name), 'w') as fp:
        json.dump(prog, fp)

def eval_canary_on_attacks(canary_name, canary_path):
    print(f'\n=== Canary: {canary_name} ===', flush=True)
    cfg = get_args()
    freeze_seed(cfg.seed)
    detector = YOLOv8()
    canary = Canary(cfg, detector)
    canary.eval_load_canary(canary_path=canary_path, canary_cls_id=22)
    print(f'  Model loaded, evaluating all attacks...', flush=True)

    results = {}
    for attack_name, (adv_dir, ben_dir) in ATTACKS.items():
        prog = load_ckpt(canary_name, attack_name)
        adv_imgs = sorted([f for f in os.listdir(adv_dir) if f.endswith(('.jpg','.png'))])
        ben_imgs = sorted([f for f in os.listdir(ben_dir) if f.endswith(('.jpg','.png'))])
        total = len(adv_imgs) + len(ben_imgs)
        done = len(prog['done_adv']) + len(prog['done_benign'])
        print(f'  [{attack_name}] {done}/{total} already done', flush=True)

        for fname in adv_imgs:
            if fname in prog['done_adv']:
                continue
            img = cv2.imread(os.path.join(adv_dir, fname), 1)
            if img is None:
                continue
            try:
                is_attack = canary.eval_single(img)
                prog['TP' if is_attack else 'FN'] += 1
            except:
                pass
            prog['done_adv'].append(fname)
            save_ckpt(canary_name, attack_name, prog)

        for fname in ben_imgs:
            if fname in prog['done_benign']:
                continue
            img = cv2.imread(os.path.join(ben_dir, fname), 1)
            if img is None:
                continue
            try:
                is_attack = canary.eval_single(img)
                prog['FP' if is_attack else 'TN'] += 1
            except:
                pass
            prog['done_benign'].append(fname)
            save_ckpt(canary_name, attack_name, prog)

        TP,FP,FN,TN = prog['TP'],prog['FP'],prog['FN'],prog['TN']
        f1,fpr = compute_metrics(TP,FP,FN,TN)
        print(f'  [{attack_name}] TP:{TP} FP:{FP} FN:{FN} TN:{TN} → F1:{f1:.3f} FPR:{fpr:.3f}', flush=True)
        results[attack_name] = {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'F1':f1,'FPR':fpr}

    return results

if __name__ == '__main__':
    all_results = {'zebra': {}}
    for atk, counts in ZEBRA_RESULTS.items():
        TP,FP,FN,TN = counts['TP'],counts['FP'],counts['FN'],counts['TN']
        f1,fpr = compute_metrics(TP,FP,FN,TN)
        all_results['zebra'][atk] = {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'F1':f1,'FPR':fpr}

    for canary_name, canary_path in CANARIES.items():
        all_results[canary_name] = eval_canary_on_attacks(canary_name, canary_path)

    print('\n\n========== ABLATION RESULTS ==========')
    print(f"{'Canary':<14} {'Attack':<12} {'F1':>6} {'FPR':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print('-'*65)
    for cname, attacks in all_results.items():
        for aname, r in attacks.items():
            print(f"{cname:<14} {aname:<12} {r['F1']:>6.3f} {r['FPR']:>6.3f} {r['TP']:>5} {r['FP']:>5} {r['FN']:>5} {r['TN']:>5}")
