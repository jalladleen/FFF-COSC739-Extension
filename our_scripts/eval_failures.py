"""
eval_failures.py
Find false negative (missed attack) and false positive images
for AdvPatch and UPC with the zebra canary.
Saves results to failures_advpatch.json and failures_upc.json
"""
import os, sys, json, cv2, copy
from copy import deepcopy

sys.argv = ['eval', '--test', '--df_mode', 'C', '--defensive_patch_location', 'cc',
            '--canary_cls_id', '22', '--canary_size', '80', '--person_conf', '0.05',
            '--weight', '2.0', '--best_canary_path', 'dummy', '--input_img', 'dummy']

from YOLOv8_Combiner import get_args, Canary, freeze_seed
from ObjectDetector.fjn_yolov8 import FJN_YOLOv8 as YOLOv8

ZEBRA_PATH = './FJNTraining/canary_zebra/exp_VOC07_120_22_80_50/canary_050.png'

ATTACKS = {
    'AdvPatch': ('Data/testeval/VOC07_YOLOv8/test/AdvPatch/adversarial',
                 'Data/testeval/VOC07_YOLOv8/test/AdvPatch/benign'),
    'UPC':      ('Data/testeval/VOC07_YOLOv8/test/UPC/adversarial',
                 'Data/testeval/VOC07_YOLOv8/test/UPC/benign'),
}

if __name__ == '__main__':
    cfg = get_args()
    freeze_seed(cfg.seed)
    detector = YOLOv8()
    canary = Canary(cfg, detector)
    canary.eval_load_canary(canary_path=ZEBRA_PATH, canary_cls_id=22)

    for attack_name, (adv_dir, ben_dir) in ATTACKS.items():
        fn_files = []
        fp_files = []

        adv_imgs = sorted([f for f in os.listdir(adv_dir) if f.endswith(('.jpg','.png'))])
        ben_imgs = sorted([f for f in os.listdir(ben_dir) if f.endswith(('.jpg','.png'))])

        print(f'\n[{attack_name}] Scanning {len(adv_imgs)} adversarial images...', flush=True)
        for fname in adv_imgs:
            img = cv2.imread(os.path.join(adv_dir, fname), 1)
            if img is None:
                continue
            try:
                is_attack = canary.eval_single(img)
                if not is_attack:
                    fn_files.append(fname)
                    print(f'  FN: {fname}', flush=True)
            except Exception as e:
                print(f'  ERROR {fname}: {e}', flush=True)

        print(f'[{attack_name}] Scanning {len(ben_imgs)} benign images...', flush=True)
        for fname in ben_imgs:
            img = cv2.imread(os.path.join(ben_dir, fname), 1)
            if img is None:
                continue
            try:
                is_attack = canary.eval_single(img)
                if is_attack:
                    fp_files.append(fname)
                    print(f'  FP: {fname}', flush=True)
            except Exception as e:
                print(f'  ERROR {fname}: {e}', flush=True)

        result = {'FN': fn_files, 'FP': fp_files}
        outfile = f'failures_{attack_name.lower()}.json'
        with open(outfile, 'w') as f:
            json.dump(result, f, indent=2)

        print(f'\n[{attack_name}] FN={len(fn_files)} FP={len(fp_files)} → saved to {outfile}')
