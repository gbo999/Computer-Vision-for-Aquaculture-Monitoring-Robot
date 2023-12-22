
# Load your image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import math

class GridOverlayApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Grid Overlay App")

        # Create a canvas for image display
        self.canvas = tk.Canvas(self.master, bg='white', width=800, height=600)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # Bind events for grid manipulation
        self.canvas.bind('<ButtonPress-1>', self.start_move)  # Move grid
        self.canvas.bind('<B1-Motion>', self.move_grid)
        self.canvas.bind('<MouseWheel>', self.scale_grid)  # Windows/Linux, Zoom in/out
        self.canvas.bind('<Button-4>', self.scale_grid)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.scale_grid)  # Linux scroll down
        self.canvas.bind('<Shift-MouseWheel>', self.rotate_grid)  # Rotate grid

        # Menu for additional actions
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open Image', command=self.load_image)
        file_menu.add_command(label='Save Grid Data', command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.master.quit)

        # Image related attributes
        self.image = None
        self.photo_image = None

        # Grid related attributes
        self.grid_lines = []
        self.grid_size = 50
        self.grid_color = 'blue'
        self.rotation_angle = 0

        # Store the start position for moving the grid
        self.start_x = None
        self.start_y = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Load the image with PIL
        self.image = Image.open(file_path)

        # Resize the image to fit the canvas, maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.image.size

        # Calculate the new size to fit the canvas
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        scale_factor = min(scale_width, scale_height)

        # Avoid upscaling the image if it's smaller than the canvas
        scale_factor = min(scale_factor, 1)

        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        # Resize and display the image
        self.image = self.image.resize((new_width, new_height), Image.LANCZOS)
        self.display_image()


    def display_image(self):
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.draw_grid()

    def draw_grid(self):
        img_width, img_height = self.image.size
        for i in range(0, img_width, self.grid_size):
            line = self.canvas.create_line(i, 0, i, img_height, fill=self.grid_color)
            self.grid_lines.append(line)
        for i in range(0, img_height, self.grid_size):
            line = self.canvas.create_line(0, i, img_width, i, fill=self.grid_color)
            self.grid_lines.append(line)

    def clear_grid(self):
        for line in self.grid_lines:
            self.canvas.delete(line)
        self.grid_lines.clear()


    def start_move(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def move_grid(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        for line in self.grid_lines:
            self.canvas.move(line, dx, dy)
        self.start_x = event.x
        self.start_y = event.y

    def scale_grid(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        self.grid_size = int(self.grid_size * scale_factor)
        self.draw_grid()

    def rotate_grid(self, event=None, angle_degrees=5):
    # Placeholder function to demonstrate rotation
        self.rotation_angle = (self.rotation_angle + angle_degrees) % 360
        angle_radians = math.radians(self.rotation_angle)
        cos_val = math.cos(angle_radians)
        sin_val = math.sin(angle_radians)

        img_width, img_height = self.image.size
        for line in self.grid_lines:
            self.canvas.delete(line)
        self.grid_lines.clear()

        for i in range(0, img_width, self.grid_size):
            x1, y1 = self.rotate_point((i, 0), (img_width / 2, img_height / 2), cos_val, sin_val)
            x2, y2 = self.rotate_point((i, img_height), (img_width / 2, img_height / 2), cos_val, sin_val)
            line = self.canvas.create_line(x1, y1, x2, y2, fill=self.grid_color)
            self.grid_lines.append(line)
        for i in range(0, img_height, self.grid_size):
            x1, y1 = self.rotate_point((0, i), (img_width / 2, img_height / 2), cos_val, sin_val)
            x2, y2 = self.rotate_point((img_width, i), (img_width / 2, img_height / 2), cos_val, sin_val)
            line = self.canvas.create_line(x1, y1, x2, y2, fill=self.grid_color)
            self.grid_lines.append(line)

    def rotate_point(self, point, origin, cos_val, sin_val):
        # Rotate a point counterclockwise by a given angle around a given origin.
        ox, oy = origin
        px, py = point

        qx = ox + cos_val * (px - ox) - sin_val * (py - oy)
        qy = oy + sin_val * (px - ox) + cos_val * (py - oy)
        return qx, qy
    def save_data(self):
        # Code to save grid data to file
        pass

if __name__ == '__main__':
    root = tk.Tk()
    app = GridOverlayApp(root)
    root.mainloop()
