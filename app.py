import tkinter as tk
from window import Window

WINNAME = "Processamento e Análise de Imagens"

def window():
    root = tk.Tk()
    root.title(WINNAME)
    root.config(bg="#191919")
    app = Window(root)
    app.update_image()
    root.state('zoomed')
    root.mainloop()


if __name__ == "__main__":
    window()
