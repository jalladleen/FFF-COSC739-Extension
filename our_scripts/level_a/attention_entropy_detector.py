import torch, os, json
import torch.nn.functional as F
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np

device = torch.device('cuda')
print('Loading DETR...', flush=True)
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained(
    'facebook/detr-resnet-50', attn_implementation='eager'
).to(device)
model.eval()
print('Done.\n', flush=True)

ATTACKS = {
    'AdvPatch': 'Data/testeval/VOC07_YOLOv8/test/AdvPatch',
    'UPC':      'Data/testeval/VOC07_YOLOv8/test/UPC',
    'TCEGA':    'Data/testeval/VOC07_YOLOv8/test/TCEGA',
    'Natural':  'Data/testeval/VOC07_YOLOv8/test/Natural',
}

THRESHOLD = 6.5

def get_min_entropy(img_path):
    img = Image.open(img_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.cross_attentions[-1].mean(dim=1).squeeze(0)
    attn = F.softmax(attn, dim=-1)
    entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
    return entropy.min().item()

def compute_metrics(TP, FP, FN, TN):
    P   = TP/(TP+FP)  if (TP+FP)>0 else 0
    R   = TP/(TP+FN)  if (TP+FN)>0 else 0
    F1  = 2*P*R/(P+R) if (P+R)>0   else 0
    FPR = FP/(FP+TN)  if (FP+TN)>0 else 0
    return P, R, F1, FPR

def eval_attack(attack_name, base_path, threshold):
    adv_dir = base_path + '/adversarial'
    ben_dir = base_path + '/benign'
    adv_imgs = sorted([f for f in os.listdir(adv_dir) if f.endswith(('.jpg','.png'))])
    ben_imgs = sorted([f for f in os.listdir(ben_dir) if f.endswith(('.jpg','.png'))])
    print('  [%s] %d adv + %d benign images...' % (attack_name, len(adv_imgs), len(ben_imgs)), flush=True)
    TP = FN = FP = TN = 0
    adv_entropies = []
    ben_entropies = []
    for fname in adv_imgs:
        e = get_min_entropy(adv_dir + '/' + fname)
        adv_entropies.append(e)
        if e < threshold: TP += 1
        else: FN += 1
    for fname in ben_imgs:
        e = get_min_entropy(ben_dir + '/' + fname)
        ben_entropies.append(e)
        if e < threshold: FP += 1
        else: TN += 1
    P, R, F1, FPR = compute_metrics(TP, FP, FN, TN)
    print('    TP=%d FP=%d FN=%d TN=%d' % (TP, FP, FN, TN))
    print('    Adv entropy: mean=%.3f std=%.3f range=[%.3f, %.3f]' % (
        np.mean(adv_entropies), np.std(adv_entropies), min(adv_entropies), max(adv_entropies)))
    print('    Ben entropy: mean=%.3f std=%.3f range=[%.3f, %.3f]' % (
        np.mean(ben_entropies), np.std(ben_entropies), min(ben_entropies), max(ben_entropies)))
    print('    P=%.3f R=%.3f F1=%.3f FPR=%.3f' % (P, R, F1, FPR))
    return {'TP':TP,'FP':FP,'FN':FN,'TN':TN,'P':P,'R':R,'F1':F1,'FPR':FPR,
            'adv_entropy_mean':float(np.mean(adv_entropies)),
            'ben_entropy_mean':float(np.mean(ben_entropies))}

print('='*60)
print('ATTENTION ENTROPY DETECTOR — Full Evaluation')
print('Threshold: %.1f' % THRESHOLD)
print('='*60)

all_results = {}
for attack_name, base_path in ATTACKS.items():
    print('\n--- %s ---' % attack_name)
    all_results[attack_name] = eval_attack(attack_name, base_path, THRESHOLD)

os.makedirs('results', exist_ok=True)
with open('results/attention_entropy_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

yolov8     = {'AdvPatch':{'F1':0.966,'FPR':0.051},'UPC':{'F1':0.951,'FPR':0.027},'TCEGA':{'F1':0.836,'FPR':0.084},'Natural':{'F1':0.905,'FPR':0.063}}
detr_canary= {'AdvPatch':{'F1':0.795,'FPR':0.386},'UPC':{'F1':0.555,'FPR':0.356},'TCEGA':{'F1':0.512,'FPR':0.316},'Natural':{'F1':0.537,'FPR':0.263}}

print('\n\n' + '='*75)
print('COMPARISON: YOLOv8 Canary vs DETR Canary vs Attention Entropy Detector')
print('='*75)
print('%-12s %8s %8s | %8s %8s | %8s %8s' % ('Attack','YOLOv8F1','YOLOv8FPR','DETRcnF1','DETRcnFPR','AttnF1','AttnFPR'))
print('-'*75)
for atk in ATTACKS:
    r = all_results[atk]
    print('%-12s %8.3f %8.3f | %8.3f %8.3f | %8.3f %8.3f' % (
        atk, yolov8[atk]['F1'], yolov8[atk]['FPR'],
        detr_canary[atk]['F1'], detr_canary[atk]['FPR'],
        r['F1'], r['FPR']))

print('\nResults saved to results/attention_entropy_results.json')
