import os
import csv
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image

class DetectionReviewer:
    def __init__(self, visualized_dir, labels_dir, output_csv):
        """
        Initialization:
        1. Sets up paths for visualized images and labels
        2. Loads existing review results if any
        3. Creates the user interface
        """
        self.visualized_dir = Path(visualized_dir)
        self.labels_dir = Path(labels_dir)
        self.output_csv = Path(output_csv)
        
        # Get all visualized images
        self.images = sorted(list(self.visualized_dir.glob("viz_*.jpg")))
        self.current_idx = 0
        self.total_images = len(self.images)
        
        print(f"Found {self.total_images} images to review")
        
        # Initialize results dictionary
        self.results = {}
        
        # Load existing results if the CSV file exists
        if self.output_csv.exists():
            with open(self.output_csv, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.results[row['image_name']] = {
                        'is_good': row['is_good'] == 'True',
                        'has_big': row['has_big'] == 'True',
                        'has_small': row['has_small'] == 'True'
                    }
            print(f"Loaded {len(self.results)} existing reviews")
    
    def get_detection_info(self, image_path):
        """
        Analyzes label files to determine:
        1. If image has big exuviae (175-220mm)
        2. If image has small exuviae (116-174mm)
        Uses same measurement logic as visualization
        """
        # Get the base image name without 'viz_' prefix
        base_name = image_path.name[4:] if image_path.name.startswith("viz_") else image_path.name
        label_file = self.labels_dir / f"{Path(base_name).stem}.txt"
        
        has_big = False
        has_small = False
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                detections = f.readlines()
                
                # Read each detection and check the calculated size
                for detection in detections:
                    # If we already found both big and small, no need to continue
                    if has_big and has_small:
                        break
                    
                    values = list(map(float, detection.strip().split()))
                    
                    # Extract keypoints
                    keypoints = []
                    for i in range(5, len(values)-1, 3):
                        x = values[i]
                        y = values[i + 1]
                        keypoints.append([x, y])
                    
                    if len(keypoints) >= 4:
                        # Calculate total length for size determination
                        # Constants for size calculation (original image dimensions)
                        calc_width = 5312
                        calc_height = 2988
                        
                        # Determine image type and height
                        is_circle2 = "GX010191" in base_name
                        height_mm = 700 if is_circle2 else 410
                        
                        # Calculate total length
                        keypoints_calc = []
                        for kp in keypoints:
                            x = kp[0] * calc_width
                            y = kp[1] * calc_height
                            keypoints_calc.append([x, y])
                        
                        import math
                        # Calculate Euclidean distance
                        def calculate_euclidean_distance(point1, point2):
                            return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                            
                        # Calculate total length in original image pixels
                        total_length_pixels = calculate_euclidean_distance(keypoints_calc[3], keypoints_calc[2])
                        
                        # Convert to mm using original image dimensions
                        diagonal_image_size = math.sqrt(calc_width ** 2 + calc_height ** 2)
                        total_length_mm = (total_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(84.6/2)))
                        
                        # Check if detection is big or small using the same criteria as visualization
                        if 175 <= total_length_mm <= 220:
                            has_big = True
                        elif 116 <= total_length_mm <= 174:
                            has_small = True
        
        return has_big, has_small

    def show_current_image(self):
        """Display the current image and update status information"""
        if self.current_idx < 0 or self.current_idx >= self.total_images:
            return
        
        # Get current image path
        image_path = self.images[self.current_idx]
        image_name = image_path.name
        print(f"Loading image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"ERROR: Image file does not exist: {image_path}")
            plt.clf()
            plt.text(0.5, 0.5, f"ERROR: File not found: {image_name}", 
                     ha='center', va='center', fontsize=12)
            plt.draw()
            return
            
        try:
            # Load image with PIL
            img = Image.open(image_path)
            print(f"Image loaded successfully. Size: {img.size}")
            
            # Get detection info
            has_big, has_small = self.get_detection_info(image_path)
            print(f"Detections - Big: {has_big}, Small: {has_small}")
            
            # Get or initialize results for this image
            if image_name in self.results:
                is_good = self.results[image_name]['is_good']
            else:
                is_good = False
                self.results[image_name] = {
                    'is_good': is_good,
                    'has_big': has_big,
                    'has_small': has_small
                }
            
            # Clear the figure
            plt.clf()
            
            # Display the image
            plt.imshow(np.array(img))
            
            # Update status
            status_text = f"Image {self.current_idx + 1}/{self.total_images}: "
            status_text += f"{'GOOD' if is_good else 'BAD'} | "
            status_text += f"Big: {'YES' if has_big else 'NO'} | "
            status_text += f"Small: {'YES' if has_small else 'NO'}"
            plt.title(f"{image_name}\n{status_text}", fontsize=14)
            
            # Hide the axes for a cleaner look
            plt.axis('off')
            
            # Update button colors - highlight the current status
            self.good_button.color = 'green' if is_good else 'lightgreen'
            self.bad_button.color = 'red' if not is_good else 'salmon'
            
            # Redraw the buttons
            self.good_button.ax.figure.canvas.draw_idle()
            self.bad_button.ax.figure.canvas.draw_idle()
            
            plt.tight_layout()
            plt.draw()
            
        except Exception as e:
            print(f"Error displaying image: {e}")
            plt.clf()
            plt.text(0.5, 0.5, f"ERROR loading {image_name}: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            plt.draw()

    def mark_good(self, event=None):
        """Mark the current image as good"""
        if self.current_idx < 0 or self.current_idx >= self.total_images:
            return
        
        image_name = self.images[self.current_idx].name
        self.results[image_name]['is_good'] = True
        self.next_image()
    
    def mark_bad(self, event=None):
        """Mark the current image as bad"""
        if self.current_idx < 0 or self.current_idx >= self.total_images:
            return
        
        image_name = self.images[self.current_idx].name
        self.results[image_name]['is_good'] = False
        self.next_image()
    
    def prev_image(self, event=None):
        """Navigate to the previous image"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current_image()
    
    def next_image(self, event=None):
        """Navigate to the next image"""
        if self.current_idx < self.total_images - 1:
            self.current_idx += 1
            self.show_current_image()
    
    def save_results(self, event=None):
        """Save the results to a CSV file"""
        with open(self.output_csv, 'w', newline='') as f:
            fieldnames = ['image_name', 'is_good', 'has_big', 'has_small']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for image_name, data in self.results.items():
                writer.writerow({
                    'image_name': image_name,
                    'is_good': data['is_good'],
                    'has_big': data['has_big'],
                    'has_small': data['has_small']
                })
        
        print(f"Results saved to {self.output_csv}")
        plt.close()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'left':
            self.prev_image()
        elif event.key == 'right':
            self.next_image()
        elif event.key == 'g':
            self.mark_good()
        elif event.key == 'b':
            self.mark_bad()
        elif event.key == 's':
            self.save_results()
    
    def run(self):
        """Run the reviewer with matplotlib interface"""
        plt.figure(figsize=(12, 9))
        
        # Create a grid for the buttons
        plt.subplots_adjust(bottom=0.2)
        
        # Create the buttons with larger size and better colors
        ax_prev = plt.axes([0.1, 0.05, 0.15, 0.1])
        ax_good = plt.axes([0.3, 0.05, 0.15, 0.1])
        ax_bad = plt.axes([0.5, 0.05, 0.15, 0.1])
        ax_next = plt.axes([0.7, 0.05, 0.15, 0.1])
        ax_save = plt.axes([0.85, 0.05, 0.1, 0.1])
        
        self.prev_button = Button(ax_prev, 'Previous', color='lightblue', hovercolor='skyblue')
        self.good_button = Button(ax_good, 'Mark as Good', color='lightgreen', hovercolor='green')
        self.bad_button = Button(ax_bad, 'Mark as Bad', color='salmon', hovercolor='red')
        self.next_button = Button(ax_next, 'Next', color='lightblue', hovercolor='skyblue')
        self.save_button = Button(ax_save, 'Save & Exit', color='yellow', hovercolor='gold')
        
        # Make button labels larger
        for button in [self.prev_button, self.good_button, self.bad_button, self.next_button, self.save_button]:
            button.label.set_fontsize(14)
        
        self.prev_button.on_clicked(self.prev_image)
        self.good_button.on_clicked(self.mark_good)
        self.bad_button.on_clicked(self.mark_bad)
        self.next_button.on_clicked(self.next_image)
        self.save_button.on_clicked(self.save_results)
        
        # Connect keyboard shortcuts
        plt.gcf().canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add a text box at the top to show controls
        plt.figtext(0.5, 0.95, "Controls: Left/Right arrows to navigate, 'g' for good, 'b' for bad, 's' to save",
                   ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Show the first image
        self.show_current_image()
        
        plt.show()


if __name__ == "__main__":
    visualized_dir = "runs/pose/predict57/viz_real"  # Directory with visualized images
    labels_dir = "runs/pose/predict57/filtered_labels"
    output_csv = "runs/pose/predict57/review_results.csv"
    
    # Print debug info about directories and check if they exist
    visualized_path = Path(visualized_dir).absolute()
    labels_path = Path(labels_dir).absolute()
    print(f"Looking for images in: {visualized_path}")
    print(f"Looking for labels in: {labels_path}")
    
    # Check if directories exist
    if not visualized_path.exists():
        print(f"ERROR: Image directory does not exist: {visualized_path}")
        print("Creating the directory...")
        os.makedirs(visualized_path, exist_ok=True)
    
    if not labels_path.exists():
        print(f"ERROR: Labels directory does not exist: {labels_path}")
        print("Creating the directory...")
        os.makedirs(labels_path, exist_ok=True)
    
    # List all files in visualized_dir to see what's there
    print("\nFiles in image directory:")
    if visualized_path.exists():
        files = list(visualized_path.glob("*"))
        for file in files[:10]:  # Show first 10 files
            print(f"  - {file.name}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        if not files:
            print("  No files found!")
    
    reviewer = DetectionReviewer(visualized_dir, labels_dir, output_csv)
    reviewer.run() 