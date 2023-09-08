import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import random

class SegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation")
        
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image = None  # To store the loaded image
        self.results = None
        self.names = None
        self.executed = False
        self.mask_colors = ["red", "green", "blue", "black", "white"]
        self.text_colors = ["black", "white"]
        self.rect = None
        self.text = None
        self.text2 = None

        self.annotation_status_label = tk.Label(self.root, text="Upload an image first!")
        self.annotation_status_label.pack()

        self.load_image_button = ttk.Button(self.root, text="Upload Image", command=self.load_image)
        self.load_image_button.pack()

        self.segment_button = ttk.Button(self.root, text="View Segments", state=tk.DISABLED, command=self.segment)
        self.segment_button.pack()

        self.canvas.bind("<Motion>", self.on_mouse_motion)



    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp *.jpeg")])
        if file_path:
            self.annotation_status_label.config(text="Loading image...")
            image = cv2.imread(file_path)
            self.image = self.resize_image(image, 1000)
            if self.image is not None:
                self.annotation_status_label.config(text="Processing started...")
                self.display_image()

                self.executed = True
                model = YOLO('yolov8n-seg.pt')
                results = model(self.image)
                self.annotation_status_label.config(text="Processing done! Hover over the image to find objects!")
                self.results = results
                self.names = self.results[0].names
                self.segment_button.config(state=tk.NORMAL)

    def display_image(self):
        if self.image is not None:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))

            self.canvas.config(width=width, height=height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            

    def segment(self):
        if self.results is not None:
            img  = self.results[0].plot(pil=False, img=self.image, boxes=False, masks=True) 
            # im = Image.fromarray(im_array[..., ::-1])  
            cv2.imshow("Segments", img)
            # im.show()
            
    def is_inside_box(self, point, box):
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def remove_rectangle(self, rect_id, text_id, text_id2):
        self.canvas.delete(rect_id) 
        self.canvas.delete(text_id)
        self.canvas.delete(text_id2)

    def on_mouse_motion(self, event):
        
        if self.executed is True:

            x, y = event.x, event.y
            delay_ms = 100

            def after_delay():
                if x == event.x and y == event.y:
                    found_one = False
                    for mask_index in range(len(self.results[0].boxes.xyxy)):
                        box = self.results[0].boxes.xyxy[mask_index]
                        if self.is_inside_box((x, y), box):
                            x1, y1, x2, y2 = box
                            x1 = int(x1)
                            y1 = int(y1)
                            x2 = int(x2)
                            y2 = int(y2)
                            self.rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=random.choice(self.mask_colors), width=2)
                            self.text = self.canvas.create_text(x1, y1, 
                                                                text = str(self.names[self.results[0].boxes.cls[mask_index].item()]), 
                                                                font=("Arial", 18), anchor="nw", 
                                                                fill=random.choice(self.text_colors),
                                                                )
                            
                            self.text2 = self.canvas.create_text(x1+1, y1+1, 
                                                                text = str(self.names[self.results[0].boxes.cls[mask_index].item()]), 
                                                                font=("Arial", 18), anchor="nw", 
                                                                fill=random.choice(self.text_colors),
                                                                )
                            self.canvas.tag_bind(self.rect, "<Leave>", lambda e, rect=self.rect: self.remove_rectangle(rect, self.text, self.text2))
                            # self.canvas.tag_bind(self.text, "<Leave>", lambda e, text_item=self.text: self.remove_text(text_item))
                            found_one = True
                        else:
                            if found_one == False:
                                all_items = self.canvas.find_all()
                                for item in all_items:
                                    item_type = self.canvas.type(item)
                                    if item_type == "rectangle" or item_type == 'text':
                                        self.canvas.delete(item)
                            
            self.root.after(delay_ms, after_delay)

    
    def resize_image(self, image, target_size=500):

        # Get the dimensions of the original image
        height, width = image.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = width / float(height)

        # Determine the new dimensions based on the target size
        if width > 500 or height > 500:
            if width > height:
                new_width = target_size
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = target_size
                new_width = int(new_height * aspect_ratio)

            # Check if resizing is needed
            if width != new_width or height != new_height:
                # Resize the image while preserving the aspect ratio
                resized_image = cv2.resize(image, (new_width, new_height))
                return resized_image
                
            else:
                # No resizing needed, save the original image
                return image
        else:
                # No resizing needed, save the original image
                return image


def main():
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()