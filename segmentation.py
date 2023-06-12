import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt


IMGS = glob.glob('./mamografias/**/*.png')


def plot_histogram(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(10, 6))
    plt.hist(img.flatten(), bins=256, range=[0, 256], color='red', alpha=0.7)
    plt.title(f'Histograma {img_path}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


def segmentation(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # binariza a img
    _, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
    print(_)
    cv2.imwrite('./test/test_bin.png', threshold)
    # encontra os contornos da img binarizada
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # encontra o maior contorno, baseado na area
    largest_contour = max(contours, key=cv2.contourArea)

    # cria a mascara, utilizando o maior contorno
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    cv2.imwrite('./test/test_mask.png', mask)

    # aplica a mascara na img
    segmented_image = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite('./test/test_segmented.png', segmented_image)


# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (66).png")]
# IMAGEM = IMGS[111]
IMAGEM = IMGS[3]
print(IMAGEM)  
# plot_histogram(IMAGEM)
segmentation(IMAGEM)
