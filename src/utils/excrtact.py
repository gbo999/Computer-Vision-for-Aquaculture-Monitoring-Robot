import os
import pytesseract
from PIL import Image

# List of confusion matrix image paths
image_paths = [
    "vals2/val_confusion_matrix.png",
    "vals2/val2_confusion_matrix.png",
    "vals2/val3_confusion_matrix.png",
    "vals2/val4_confusion_matrix.png",
    "vals2/val5_confusion_matrix.png",
    "vals2/val6_confusion_matrix.png",
    "vals2/val7_confusion_matrix.png",
    "vals2/val8_confusion_matrix.png",
    "vals2/val9_confusion_matrix.png",
    "vals2/val10_confusion_matrix.png"
]

def extract_confusion_matrix_values(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    lines = text.strip().split('\n')
    values = []
    for line in lines:
        nums = [int(s) for s in line.split() if s.isdigit()]
        values.extend(nums)
    
    if len(values) >= 4:
        tp, fp, fn, tn = values[:4]
        return tp, fp, fn, tn
    else:
        return None

def calculate_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score

# Analyze each confusion matrix
results = {}
for path in image_paths:
    values = extract_confusion_matrix_values(path)
    if values:
        print(values)
        tp, fp, fn, tn = values
        f1_score = calculate_f1_score(tp, fp, fn)
        results[path] = f1_score

# Find the best confusion matrix by F1 score
best_matrix = max(results, key=results.get)
best_f1_score = results[best_matrix]

print(f"The best confusion matrix is {best_matrix} with an F1 score of {best_f1_score:.2f}")