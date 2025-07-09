import time
from ultralytics import YOLO

# Load your model
model = YOLO('path/to/your/model.pt')

# Load your image
image = 'path/to/your/image.jpg'

# Timing inference
start_time = time.time()
results = model.predict(image)
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference time: {inference_time} seconds")