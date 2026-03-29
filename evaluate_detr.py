import os
from ObjectDetector.detr_detector import DETRDetector, find_hidden_person

detector = DETRDetector()

BASE_PATH = "Data/testeval/VOC07_YOLOv8/test"

datasets = ["AdvPatch", "TCEGA", "UPC"]

def evaluate_folder(folder_path, name):
    print(f"\n===== {name} =====")

    total = 0
    total_boxes = 0

    for img in os.listdir(folder_path):
        path = os.path.join(folder_path, img)

        outputs, image_size = detector.detect(path)
        boxes = find_hidden_person(outputs, image_size)

        total += 1
        total_boxes += len(boxes)

    avg_boxes = total_boxes / total if total > 0 else 0

    print(f"Images: {total}")
    print(f"Total candidate boxes: {total_boxes}")
    print(f"Average boxes per image: {avg_boxes:.2f}")


# 🔴 Adversarial
for d in datasets:
    adv_path = os.path.join(BASE_PATH, d, "adversarial")
    evaluate_folder(adv_path, f"{d} (Adversarial)")


# 🟢 Benign
for d in datasets:
    benign_path = os.path.join(BASE_PATH, d, "benign")
    evaluate_folder(benign_path, f"{d} (Benign)")
