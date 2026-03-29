import torch, os
import torch.nn.functional as F
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import numpy as np

device = torch.device('cuda')
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
model = DetrForObjectDetection.from_pretrained(
    'facebook/detr-resnet-50', attn_implementation='eager'
).to(device)
model.eval()

def get_entropy(img_path):
    img = Image.open(img_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attn = outputs.cross_attentions[-1].mean(dim=1).squeeze(0)
    attn = F.softmax(attn, dim=-1)
    entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
    return entropy.min().item(), entropy.mean().item()

base = 'Data/testeval/VOC07_YOLOv8/test/TCEGA'
clean_files  = sorted(os.listdir(base + '/benign'))[:15]
attack_files = sorted(os.listdir(base + '/adversarial'))[:15]

clean_mins, attack_mins = [], []
print('%-25s %8s %8s  %s' % ('Image', 'min_H', 'mean_H', 'label'))
print('-' * 55)

for f in clean_files:
    mn, me = get_entropy(base + '/benign/' + f)
    clean_mins.append(mn)
    print('%-25s %8.3f %8.3f  CLEAN' % (f, mn, me))

for f in attack_files:
    mn, me = get_entropy(base + '/adversarial/' + f)
    attack_mins.append(mn)
    print('%-25s %8.3f %8.3f  ATTACK' % (f, mn, me))

print()
print('CLEAN  min_entropy: mean=%.3f  std=%.3f  range=[%.3f, %.3f]' % (
    np.mean(clean_mins), np.std(clean_mins), min(clean_mins), max(clean_mins)))
print('ATTACK min_entropy: mean=%.3f  std=%.3f  range=[%.3f, %.3f]' % (
    np.mean(attack_mins), np.std(attack_mins), min(attack_mins), max(attack_mins)))

gap = np.mean(clean_mins) - np.mean(attack_mins)
print()
print('Gap (clean - attack): %.3f' % gap)
if gap > 0.1:
    print('VERDICT: HYPOTHESIS CONFIRMED -- clean images have higher entropy')
elif gap < -0.1:
    print('VERDICT: REVERSED -- attacked images have higher entropy (unexpected)')
else:
    print('VERDICT: NO CLEAR SEPARATION -- entropy does not distinguish clean vs attack')

