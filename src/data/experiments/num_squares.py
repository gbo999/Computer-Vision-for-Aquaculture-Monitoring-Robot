
# Load your image
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

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

        self.image = Image.open(file_path)
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)
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

    def rotate_grid(self, event):
        # Code to handle rotating the grid
        pass

    def save_data(self):
        # Code to save grid data to file
        pass

if __name__ == '__main__':
    root = tk.Tk()
    app = GridOverlayApp(root)
    root.mainloop()
