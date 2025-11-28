import customtkinter as ctk
from PIL import Image, ImageTk
import os

class CropGallery(ctk.CTkScrollableFrame):
    def __init__(self, master, retry_callback):
        super().__init__(master, label_text="Znalezione zdjęcia")
        self.retry_cb = retry_callback
        self.images_shown = 0
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def add_images(self, image_paths, source_file):
        for path in image_paths:
            self.images_shown += 1
            idx = self.images_shown
            card = ctk.CTkFrame(self)
            card.grid(row=(idx-1)//2, column=(idx-1)%2, padx=5, pady=5, sticky="ew")
            try:
                pil_img = Image.open(path)
                pil_img.thumbnail((150, 150))
                ctk_img = ImageTk.PhotoImage(pil_img)
                lbl_img = ctk.CTkLabel(card, image=ctk_img, text="")
                lbl_img.image = ctk_img
                lbl_img.pack(pady=5)
            except: pass
            ctk.CTkLabel(card, text=f"Foto #{idx}", font=("Arial", 10)).pack()
            ctk.CTkButton(card, text="Inna Metoda ↻", width=100, height=20,
                          font=("Arial", 10), fg_color="#555555",
                          command=lambda s=source_file: self.cycle_method(s)).pack(pady=5)

    def cycle_method(self, source_file):
        import random
        next_strat = random.choice([1, 2]) 
        self.retry_cb(source_file, next_strat)

class ComparisonSlider(ctk.CTkFrame):
    def __init__(self, master, width=800, height=600):
        super().__init__(master, width=width, height=height)
        self.w, self.h = width, height
        self.canvas = ctk.CTkCanvas(self, width=self.w, height=self.h, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.img_b = self.img_a = None
        self.pos = width // 2
        self.canvas.bind("<B1-Motion>", self.move); self.canvas.bind("<Button-1>", self.move)
    def move(self, e):
        self.pos = max(0, min(e.x, self.w)); self.draw()
    def draw(self):
        if not self.img_b or not self.img_a: return
        self.canvas.delete("all")
        self.p_b = ImageTk.PhotoImage(self.img_b.crop((0, 0, self.pos, self.h)))
        self.canvas.create_image(0, 0, image=self.p_b, anchor="nw")
        self.p_a = ImageTk.PhotoImage(self.img_a.crop((self.pos, 0, self.w, self.h)))
        self.canvas.create_image(self.pos, 0, image=self.p_a, anchor="nw")
        self.canvas.create_line(self.pos, 0, self.pos, self.h, fill="white", width=2)
