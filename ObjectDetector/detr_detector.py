import torch
import torch.nn.functional as F
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


class DETRDetector:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cpu")

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        self.model.to(self.device)
        self.model.eval()

    def detect(self, image):
        # 👇 support BOTH path and PIL image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs, image.size


def find_hidden_person(outputs, image_size, threshold=0.05, person_class_id=1):
    logits = outputs.logits.squeeze(0)
    probas = F.softmax(logits, -1)

    person_scores = probas[:, person_class_id]
    boxes = outputs.pred_boxes.squeeze(0)

    width, height = image_size

    candidate_boxes = []

    for score, box in zip(person_scores, boxes):
        if threshold < score < 0.5:  # 🔥 IMPORTANT FILTER
            cx, cy, w, h = box

            x0 = (cx - w/2) * width
            y0 = (cy - h/2) * height
            x1 = (cx + w/2) * width
            y1 = (cy + h/2) * height

            candidate_boxes.append([
                x0.item(),
                y0.item(),
                x1.item(),
                y1.item(),
                score.item()
            ])

    return candidate_boxes