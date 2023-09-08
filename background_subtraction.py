import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class BGsubtraction:
    def __init__(self, root):
        self.root = root
        self.root.title("Background subtraction")
        
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None  
        self.rect = None
        self.rect_start_x = None
        self.rect_start_y = None
        self.rect_end_x = None
        self.rect_end_y = None

        self.annotation_status_label = tk.Label(self.root, text="Upload an image first!")
        self.annotation_status_label.pack()

        self.load_image_button = ttk.Button(self.root, text="Upload Image", command=self.load_image)
        self.load_image_button.pack()

        self.bg_removal_button = ttk.Button(self.root, text="Remove Background", state=tk.DISABLED, command=self.execute)
        self.bg_removal_button.pack()


        self.canvas.bind("<ButtonPress-1>", self.start_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_rectangle)
        self.canvas.bind("<Double-Button-1>", self.discard_rectangle)
        self.canvas.bind("<B1-Motion>", self.draw_temp_rectangle)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")])
        if file_path:
            image = cv2.imread(file_path)
            self.image = self.resize_image(image, 1000)
            if self.image is not None:
                self.display_image()
                self.annotation_status_label.config(text="Draw a Rectangle!")
                self.bg_removal_button.config(state=tk.DISABLED)

    def display_image(self):
        self.annotation_status_label.config(text="Image Uploaded!")
        if self.image is not None:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
            self.canvas.config(width=width, height=height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)


    def start_rectangle(self, event):
        self.rect_start_x, self.rect_start_y = event.x, event.y

    def end_rectangle(self, event):
        self.rect_end_x, self.rect_end_y = event.x, event.y
        if self.rect_start_x and self.rect_start_y:
            self.draw_rectangle()
            x0, y0, x1, y1 = self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y
            self.annotation_status_label.config(text=f"Rectangle Coordinates: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
            self.bg_removal_button.config(state=tk.NORMAL)

    def draw_temp_rectangle(self, event):
        if self.rect_start_x and self.rect_start_y:
            x0, y0, x1, y1 = self.rect_start_x, self.rect_start_y, event.x, event.y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="red", width=2)

    def draw_rectangle(self):
        if self.rect_start_x and self.rect_start_y and self.rect_end_x and self.rect_end_y:
            x0, y0, x1, y1 = self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(x0, y0, x1, y1, outline="green", width=2)

    def discard_rectangle(self, _):
        self.rect_start_x, self.rect_start_y = None, None
        self.rect_end_x, self.rect_end_y = None, None
        if self.rect:
            self.canvas.delete(self.rect)
        # self.annotation_status_label.config(text="No Rectangle")
        self.bg_removal_button.config(state=tk.DISABLED)

    def execute(self):
        if self.rect_start_x and self.rect_start_y and self.rect_end_x and self.rect_end_y:
            x0, y0, x1, y1 = self.rect_start_x, self.rect_start_y, self.rect_end_x, self.rect_end_y

            rect = (min(x0, x1), min(y0,y1), abs(x1-x0), abs(y1-y0))
            mask = np.zeros(self.image.shape[:2], np.uint8)

            fgModel = np.zeros((1, 65), dtype="float")
            bgModel = np.zeros((1, 65), dtype="float")
            (mask, bgModel, fgModel) = cv2.grabCut(self.image, mask, rect, bgModel, fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_RECT)
            
            outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
            outputMask = (outputMask * 255).astype("uint8")
            output = cv2.bitwise_and(self.image, self.image, mask=outputMask)


            canvas = cv2.hconcat(
                [self.resize_image(self.image), 
                 self.resize_image(cv2.cvtColor(outputMask, cv2.COLOR_GRAY2BGR)), 
                 self.resize_image(output)
                 ]
                 )

            cv2.imshow('Input, foreground mask and foreground', canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    
    def resize_image(self, image, target_size=500):
        height, width = image.shape[:2]
        aspect_ratio = width / float(height)
        if width > 500 or height > 500:
            if width > height:
                new_width = target_size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(new_height * aspect_ratio)
            if width != new_width or height != new_height:
                resized_image = cv2.resize(image, (new_width, new_height))
                return resized_image
            else:
                return image
        else:
                return image


def main():
    root = tk.Tk()
    app = BGsubtraction(root)
    root.mainloop()

if __name__ == "__main__":
    main()