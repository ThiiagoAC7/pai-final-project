import tkinter as tk
from PIL import Image, ImageTk
import glob
import cv2
import numpy as np

from tkinter import filedialog
from predict import predict

IMGS = glob.glob('./datasets/dataset_validation/**/*.png')


class Window(tk.Frame):
    def __init__(self, master):
        self.master = master
        # self.label = tk.Label(self.master, text="P.A.I", bg="#191919", fg="#d4d4d4")
        # self.label.pack(fill=tk.X)

        self.SliderFrame = tk.Frame(self.master, bg="#191919")
        self.SliderFrame.pack(side=tk.LEFT, fill=tk.Y)
        self.select_image_button = tk.Button(self.SliderFrame, bg="#191919", fg="#d4d4d4",
                                             text="Selecionar Imagem", command=self.select_image)
        self.select_image_button.pack(side=tk.TOP, fill=tk.X)

        self.contrast_button = tk.Button(self.SliderFrame, text="Reset Image",
                                         bg="#191919", fg="#d4d4d4", command=self.apply_contrast)
        self.contrast_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.min_val = tk.DoubleVar()
        self.min_contrast = tk.Scale(self.SliderFrame, label="min", from_=0, to=255, variable=self.min_val,
                                     orient=tk.VERTICAL, showvalue=1, bg="#191919", fg="#d4d4d4",
                                     highlightbackground="#191919", highlightcolor="#d4d4d4",
                                     command=self.slider_handler_min)
        self.min_contrast.set(0)
        self.min_contrast.pack(side=tk.LEFT, fill=tk.Y)

        self.max_val = tk.DoubleVar()
        self.max_contrast = tk.Scale(self.SliderFrame, label="max", from_=0, to=255, variable=self.max_val,
                                     orient=tk.VERTICAL, showvalue=1, bg="#191919", fg="#d4d4d4",
                                     highlightbackground="#191919", highlightcolor="#d4d4d4",
                                     command=self.slider_handler_max)
        self.max_contrast.set(255)
        self.max_contrast.pack(side=tk.LEFT, fill=tk.Y)

        self.ClassificationFrame = tk.Frame(self.master, bg="#191919")
        self.ClassificationFrame.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_button = tk.Button(self.ClassificationFrame, bg="#191919", fg="#d4d4d4",
                                     text="INFO CLASSIFICADOR BINARIO", command=self.info_callback)
        self.info_button.pack(side=tk.TOP, fill=tk.X)

        self.info_button_quat = tk.Button(self.ClassificationFrame, bg="#191919", fg="#d4d4d4",
                                     text="INFO CLASSIFICADOR 4 CLASSES", command=self.info_callback_quat)
        self.info_button_quat.pack(side=tk.TOP, fill=tk.X)

        self.classf_bin_button = tk.Button(self.ClassificationFrame, bg="#191919", fg="#d4d4d4",
                                           text="Classificador Binário", command=self.classify_bin)
        self.classf_bin_button.pack(side=tk.BOTTOM, fill=tk.X)
        self.classf_four_button = tk.Button(self.ClassificationFrame, bg="#191919", fg="#d4d4d4",
                                            text="Classificador 4 Classes", command=self.classify_four)
        self.classf_four_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.ImgFrame = tk.Frame(self.master, bg="#181824")
        self.ImgFrame.pack(side=tk.RIGHT, fill=tk.Y)
        self.img_label = tk.Label(self.master, bg="#191919", fg="#d4d4d4")
        self.img_label.config(width=self.ImgFrame.winfo_width(),
                              height=self.ImgFrame.winfo_height())
        self.img_label.pack(fill=tk.BOTH, expand=1)

        # self.master.bind("n", self.next_prev_handler)
        # self.master.bind("p", self.next_prev_handler)
        self.index = 0
        self.curr_img_path = ''
        self.curr_img = cv2.imread(IMGS[self.index], cv2.IMREAD_GRAYSCALE)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=(
            ("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            self.curr_img_path = file_path
            self.curr_img = cv2.imread(self.curr_img_path, cv2.IMREAD_GRAYSCALE)
            self.update_image()

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

    def slider_handler_min(self, event):
        print(f'Min Contrast -> {event}')
        if self.curr_img_path != '':
            min_val = int(event)
            max_val = self.max_val.get()
            np.clip(self.curr_img, min_val, max_val, out=self.curr_img)
            self.curr_img = cv2.equalizeHist(self.curr_img)
            self.update_image()

    def slider_handler_max(self, event):
        print(f'Max Contrast -> {event}')
        if self.curr_img_path != '':
            max_val = int(event)
            min_val = self.min_val.get()
            np.clip(self.curr_img, min_val, max_val, out=self.curr_img)
            self.curr_img = cv2.equalizeHist(self.curr_img)
            self.update_image()

    def apply_contrast(self):
        """
        Reseta Imagem
        """
        self.curr_img = cv2.imread(self.curr_img_path, cv2.IMREAD_GRAYSCALE)
        self.update_image()

    def classify_bin(self):
        print(f'callback classificador binário ')

        classes = {"1_2": "BIRADS I+II",
                   "3_4": "BIRADS III+IV"}

        curr_img_path = f'.{self.curr_img_path[24:]}'

        print(curr_img_path)
        predicted_label, true_label, time, pred_probs = predict(curr_img_path)

        print(f'predicted label -> {predicted_label}')
        print(f'true label -> {true_label}')
        print(f'pred_probs -> {pred_probs}')

        fontfamily = ("Arial", 15)

        popup_window = tk.Toplevel(self.master)
        popup_window.title("Resultados")
        popup_window.geometry("500x500")

        time_label = tk.Label(
                popup_window, text=f"Classificou em: {time:.4f} segundos",  font=fontfamily)
        time_label.pack()

        predicted_label_label = tk.Label(popup_window, text=f"Classe predita: {classes[predicted_label]}",
                                         font=fontfamily)
        predicted_label_label.pack(expand=1)

        true_label_label = tk.Label(
            popup_window, text=f"Classe real: {classes[true_label]}",  font=fontfamily)
        true_label_label.pack(expand=1)


        pred_probs_label = tk.Label(popup_window, text="Certeza (probabilidades):", font=fontfamily)
        pred_probs_label.pack(expand=1)
        for key, value in pred_probs.items():
            pred_prob_label = tk.Label(popup_window, text=f"{classes[key]}: {value}", font=fontfamily)
            pred_prob_label.pack()


    def classify_four(self):
        print(f'callback classificador 4 classes ')
        # curr_img_path = IMGS[self.index]

        classes = {"1": "BIRADS I",
                   "2": "BIRADS II",
                   "3": "BIRADS III",
                   "4": "BIRADS IV"}

        curr_img_path = f'.{self.curr_img_path[24:]}'
        predicted_label, true_label, time, pred_probs = predict(
            curr_img_path, True)

        print(f'pred_probs -> {pred_probs}')
        print(f'predicted label -> {predicted_label}')
        print(f'true label -> {true_label}')

        fontfamily = ("Arial", 15)

        popup_window = tk.Toplevel(self.master)
        popup_window.title("Resultados")
        popup_window.geometry("500x500")

        time_label = tk.Label(
                popup_window, text=f"Classificou em: {time:.4f} segundos",  font=fontfamily)
        time_label.pack()

        predicted_label_label = tk.Label(popup_window, text=f"Classe predita: {classes[predicted_label]}",
                                         font=fontfamily)
        predicted_label_label.pack(expand=1)

        true_label_label = tk.Label(
            popup_window, text=f"Classe real: {classes[true_label]}",  font=fontfamily)
        true_label_label.pack(expand=1)


        pred_probs_label = tk.Label(popup_window, text="Certeza (probabilidades):", font=fontfamily)
        pred_probs_label.pack(expand=1)
        for key, value in pred_probs.items():
            pred_prob_label = tk.Label(popup_window, text=f"{classes[key]}: {value}", font=fontfamily)
            pred_prob_label.pack()

    def info_callback(self):
        print(f'callback info ')
        info = {
            'accuracy': 0.8333333333333334,
            'precision': 0.8954372623574145,
            'recall': 0.7548076923076923,
            'specificity': 0.9118589743589743,
            'fone_score': 0.8191304347826087,
            'time': 1014.721773147583
        }

        # Create a popup window
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Resultados Classificador Binário")
        popup_window.geometry("500x500")

        fontfamily = ("Arial", 15)

        title_label = tk.Label(popup_window, text="Resultados Classificador Binário", font=fontfamily)
        title_label.pack(expand=1)

        # Create and configure labels for each information
        accuracy_label = tk.Label(popup_window, text=f"Acurácia: {info['accuracy']:.4f}", font=fontfamily)
        accuracy_label.pack()

        precision_label = tk.Label(popup_window, text=f"Precisão: {info['precision']:.4f}", font=fontfamily)
        precision_label.pack()

        recall_label = tk.Label(popup_window, text=f"Recall: {info['recall']:.4f}", font=fontfamily)
        recall_label.pack()

        specificity_label = tk.Label(popup_window, text=f"Especificidade: {info['specificity']:.4f}", font=fontfamily)
        specificity_label.pack()

        fone_score_label = tk.Label(popup_window, text=f"F1 Score: {info['fone_score']:.4f}", font=fontfamily)
        fone_score_label.pack()

        time_label = tk.Label(popup_window, text=f"Tempo classificando dataset de validação\n: {info['time']:.4f} segundos", font=fontfamily)
        time_label.pack()

    def info_callback_quat(self):
        file_path = "./_graficos/confusion_matrix_4class.png"

        # Create a popup window
        popup_window = tk.Toplevel(self.master)
        popup_window.title("Resultados Classificador Quaternário")
        popup_window.geometry("640x480")
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(popup_window, image=photo)
        label.image = photo
        label.pack()
