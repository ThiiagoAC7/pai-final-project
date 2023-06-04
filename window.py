import tkinter as tk
from PIL import Image, ImageTk
import glob
import cv2
import numpy as np

IMGS = glob.glob('./mamografias/**/*.png')


class Window(tk.Frame):
    def __init__(self, master):
        self.master = master
        # self.label = tk.Label(self.master, text="P.A.I", bg="#191919", fg="#d4d4d4")
        # self.label.pack(fill=tk.X)

        self.SliderFrame = tk.Frame(self.master, bg="#191919")
        self.SliderFrame.pack(side=tk.LEFT, fill=tk.Y)
        self.contrast_button = tk.Button(self.SliderFrame, text="Contraste",
                                         bg="#191919", fg="#d4d4d4", command=self.apply_contrast)
        self.contrast_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.min_val = tk.DoubleVar()
        self.min_contrast = tk.Scale(self.SliderFrame, label="min", from_=0, to=255, variable=self.min_val,
                                     orient=tk.VERTICAL, showvalue=1, bg="#191919", fg="#d4d4d4",
                                     highlightbackground="#191919", highlightcolor="#d4d4d4")
        self.min_contrast.pack(side=tk.LEFT, fill=tk.Y)

        self.max_val = tk.DoubleVar()
        self.max_contrast = tk.Scale(self.SliderFrame, label="max", from_=0, to=255, variable=self.max_val,
                                     orient=tk.VERTICAL, showvalue=1, bg="#191919", fg="#d4d4d4",
                                     highlightbackground="#191919", highlightcolor="#d4d4d4")
        self.max_contrast.set(255)
        self.max_contrast.pack(side=tk.LEFT, fill=tk.Y)

        self.ImgFrame = tk.Frame(self.master, bg="#181824")
        self.ImgFrame.pack(side=tk.RIGHT, fill=tk.Y)
        self.img_label = tk.Label(self.master, bg="#191919", fg="#d4d4d4")
        self.img_label.config(width=self.ImgFrame.winfo_width(),
                              height=self.ImgFrame.winfo_height())
        self.img_label.pack(fill=tk.BOTH, expand=1)

        self.master.bind("n", self.next_prev_handler)
        self.master.bind("p", self.next_prev_handler)
        self.index = 0
        self.curr_img = cv2.imread(IMGS[self.index], cv2.IMREAD_GRAYSCALE)

    def update_image(self):
        """
        Atualiza a Imagem atual na tela
        """
        w = self.img_label.winfo_width()
        h = self.img_label.winfo_height()
        tkimg = ImageTk.PhotoImage(Image.fromarray(self.curr_img))
        if self.curr_img.shape[0] > h or self.curr_img.shape[1] > w:
            resized_image = cv2.resize(self.curr_img, (w, h))
            tkimg = ImageTk.PhotoImage(Image.fromarray(resized_image))
        self.img_label.config(image=tkimg)
        self.img_label.image = tkimg

    def next_prev_handler(self, event):
        """
        Handler para passar as imagens
        """
        if event.char == "n":
            self.index = (self.index + 1) % len(IMGS)
        elif event.char == "p":
            self.index = (self.index - 1) % len(IMGS)

        self.curr_img = cv2.imread(IMGS[self.index], cv2.IMREAD_GRAYSCALE)
        self.update_image()

    def apply_contrast(self):
        """
        Aplica contraste à imagem atual, considerando valores min e max do slider.
        """
        print(self.min_val.get(), self.max_val.get())
        min_val = self.min_val.get()
        max_val = self.max_val.get()
        if self.min_val.get() < self.max_val.get():
            np.clip(self.curr_img, min_val, max_val, out=self.curr_img)
            self.update_image()
