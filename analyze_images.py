import pathlib
from PIL import Image
import sys

# Find all image files
img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
files = [p for p in pathlib.Path('.').rglob('*') if p.suffix.lower() in img_exts]

sizes = []
for f in files:
    try:
        with Image.open(f) as img:
            sizes.append(img.size)  # (width, height)
    except Exception as e:
        print(f"Could not open {f}: {e}", file=sys.stderr)

if not sizes:
    print("No images found or could be opened.")
    sys.exit(0)

count = len(sizes)
avg_w = sum(w for w, h in sizes) / count
avg_h = sum(h for w, h in sizes) / count
min_w = min(w for w, h in sizes)
min_h = min(h for w, h in sizes)
max_w = max(w for w, h in sizes)
max_h = max(h for w, h in sizes)

print(f"Total images: {count}")
print(f"Average size: {int(avg_w)}x{int(avg_h)}")
print(f"Min size: {min_w}x{min_h}")
print(f"Max size: {max_w}x{max_h}") 