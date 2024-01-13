
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
        self.canvas.bind('<Button-4>', self.scale_grid)  # Linux scroll up
        self.canvas.bind('<Button-5>', self.scale_grid)  # Linux scroll down
        self.canvas.bind('<Shift-MouseWheel>', self.rotate_grid)
        self.canvas.bind('<Control-MouseWheel>', self.zoom_image)  # For zoom and rotation
        self.pan_start_x = None
        self.pan_start_y = None
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_image)
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
        self.zoom_level = 1
        self.image_on_canvas = None

        # Grid related attributes
        self.grid_lines = []
        self.grid_size = 50
        self.grid_color = 'blue'
        self.rotation_angle = 0
        self.pan_start_x = None
        self.pan_start_y = None
        # Store the start position for moving the grid
        self.start_x = None
        self.start_y = None

        self.grid_size_slider = tk.Scale(master, from_=10, to=200, orient='horizontal', label='Grid Size', command=self.update_grid_size)
        self.grid_size_slider.pack(side=tk.TOP, fill=tk.X)

        # Set initial grid size
        self.grid_size = self.grid_size_slider.get()
    
    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def on_mouse_wheel(self, event):
        if event.state == 1:  # Shift key pressed
            self.rotate_grid(angle_degrees=event.delta / 120)
        elif event.state == 4:  # Ctrl key pressed
            self.zoom_image(zoom_in=event.delta > 0)
        
    def update_grid_size(self, event=None):
        new_grid_size = self.grid_size_slider.get()
        if new_grid_size != self.grid_size:
            self.grid_size = new_grid_size
            self.draw_grid()

    def pan_image(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.canvas.scan_dragto(event.x, event.y, gain=1)

        # Update the start point for the next pan
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Load the image with PIL
        self.image = Image.open(file_path)
        self.photo_image = ImageTk.PhotoImage(self.image)

        # Create an image item on the canvas and keep a reference to it
        if self.image_on_canvas is not None:
            self.canvas.delete(self.image_on_canvas)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Set the initial zoom level
        self.zoom_level = 1

        # Update the scroll region to the size of the image
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # Draw the grid on the image
        self.draw_grid()



    def display_image(self):
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.draw_grid()

    def draw_grid(self):
        self.clear_grid()
        img_width, img_height = self.image.size

        # Calculate the center of the image
        center_x, center_y = img_width // 2, img_height // 2

        for i in range(0, img_width, self.grid_size):
            x1, y1 = self.rotate_point((i, 0), (center_x, center_y))
            x2, y2 = self.rotate_point((i, img_height), (center_x, center_y))
            line = self.canvas.create_line(x1, y1, x2, y2, fill=self.grid_color)
            self.grid_lines.append(line)
        for i in range(0, img_height, self.grid_size):
            x1, y1 = self.rotate_point((0, i), (center_x, center_y))
            x2, y2 = self.rotate_point((img_width, i), (center_x, center_y))
            line = self.canvas.create_line(x1, y1, x2, y2, fill=self.grid_color)
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

    
    def rotate_grid(self, event):
            # Adjust rotation amount based on OS
            rotation_amount = event.delta
            if os.name == 'nt':  # Windows
                rotation_amount /= 120
            self.rotation_angle = (self.rotation_angle + rotation_amount) % 360
            self.draw_grid()


    def rotate_point(self, point, origin):
        angle_radians = math.radians(self.rotation_angle)
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle_radians) * (px - ox) - math.sin(angle_radians) * (py - oy)
        qy = oy + math.sin(angle_radians) * (px - ox) + math.cos(angle_radians) * (py - oy)
        return qx, qy
    def save_data(self):
        # Code to save grid data to file
        pass

    def scale_image(self, zoom_in):
        scale_factor = 1.1 if zoom_in else 0.9
        self.canvas.scale("all", 0, 0, scale_factor, scale_factor)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
    
    def zoom_image(self, event):
        # Zoom in or out
        zoom_factor = 1.1
        if event.delta > 0:  # Zoom in
            self.zoom_level *= zoom_factor
        else:  # Zoom out
            self.zoom_level /= zoom_factor

        # Resize the image
        new_width = int(self.image.size[0] * self.zoom_level)
        new_height = int(self.image.size[1] * self.zoom_level)
        resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)

        # Update the image on the canvas
        self.photo_image = ImageTk.PhotoImage(resized_image)
        self.canvas.itemconfig(self.image_on_canvas, image=self.photo_image)

        # Update the scroll region
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

import os

if __name__ == '__main__':
    root = tk.Tk()
    app = GridOverlayApp(root)
    root.mainloop()
