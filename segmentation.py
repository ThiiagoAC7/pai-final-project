import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

IMGS = glob.glob('./mamografias/**/*.png')

def segmentation(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    # binariza a img
    cv2.imwrite('./test/test_img.png', img)
    _, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imwrite('./test/bin.png', thresh_img)

    # encontra os contornos da img binarizada
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # encontra o maior contorno, baseado na area
    largest_contour = max(contours, key=cv2.contourArea)

    # cria a mascara, utilizando o maior contorno
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    cv2.imwrite('./test/mask.png', mask)

    # aplica a mascara na img
    segmented_image = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite('./test/segmented.png', segmented_image)
    return segmented_image



def split_classes(segment=False):
    """
    Splitando as imagens em suas respectivas classes 
    As imagens com numeração múltiplo de 4 serão separadas para teste e as 
    demais para treino. As classes de BIRADS I, II, III e IV começam com a letra D, E, F 
    e G respectivamente
    """
    # create ./dataset/test and ./dataset/train folders

    dataset = 'datasets/dataset'
    if segment:
        dataset = 'datasets/dataset_segmented'

    if not os.path.exists(f'./{dataset}/'):
        os.makedirs(f'./{dataset}/train')
        os.makedirs(f'./{dataset}/train/1')
        os.makedirs(f'./{dataset}/train/2')
        os.makedirs(f'./{dataset}/train/3')
        os.makedirs(f'./{dataset}/train/4')
        os.makedirs(f'./{dataset}/test')
        os.makedirs(f'./{dataset}/test/1')
        os.makedirs(f'./{dataset}/test/2')
        os.makedirs(f'./{dataset}/test/3')
        os.makedirs(f'./{dataset}/test/4')

    # train = []
    # test = []

    for i in IMGS:
        imclass, number = i.split(' ')
        number = int(number[1:-5])
        imclass = imclass[14:].split('\\')[0]
        
        print(f'Salvando e segmentando ->{imclass}-{number}')
        img = i
        if segment:
            seg_img = segmentation(i)
            cv2.imwrite(f'./test/segmented_{imclass} ({number}).png', seg_img)
            img = f'./test/segmented_{imclass} ({number}).png'

        if number % 4 == 0:
            if imclass[0] == 'D':
                shutil.copy(img, f'./{dataset}/test/1/')
            if imclass[0] == 'E':
                shutil.copy(img, f'./{dataset}/test/2/')
            if imclass[0] == 'F':
                shutil.copy(img, f'./{dataset}/test/3/')
            if imclass[0] == 'G':
                shutil.copy(img, f'./{dataset}/test/4/')
        else:
            if imclass[0] == 'D':
                shutil.copy(img, f'./{dataset}/train/1/')
            if imclass[0] == 'E':
                shutil.copy(img, f'./{dataset}/train/2/')
            if imclass[0] == 'F':
                shutil.copy(img, f'./{dataset}/train/3/')
            if imclass[0] == 'G':
                shutil.copy(i, f'./{dataset}/train/4/')


def data_augmentation(segment=False):
    """
    Aumento de dados de teste.
    Cada imagem gera:
        - imagem espelhada
        - imagem equalizada
        - imagem espelhada e equalizada
    """

    dataset = 'datasets/dataset'
    if segment:
        dataset = 'datasets/dataset_segmented'

    imagens = glob.glob(f'./{dataset}/train/**/*.png')
    
    # original = cv2.imread(imagens[0],0)
    # img_path = imagens[0][:18].replace('\\','/') # path to img 
    # img_name = imagens[0][18:][:-4] # img name (without .png)
    # img_mirror = cv2.flip(original, 1)
    # cv2.imwrite(f'{img_path}{img_name}(flipped).png', img_mirror)
    # ori_equalized = cv2.equalizeHist(original)
    # mirror_equalized = cv2.equalizeHist(img_mirror)
    # fig, axs = plt.subplots(2, 2)
    # axs[0][0].imshow(original, cmap='gray')
    # axs[0][0].set_title('Original')
    # axs[1][0].imshow(img_mirror, cmap='gray')
    # axs[1][0].set_title('Mirror Horizontal')
    # axs[0][1].imshow(ori_equalized, cmap='gray')
    # axs[0][1].set_title('Original Equalized')
    # axs[1][1].imshow(mirror_equalized, cmap='gray')
    # axs[1][1].set_title('Mirror Equalized')
    # plt.show()

    for i in imagens:
        sep = 18 # separar nome do diretório do nome da imagem
        if segment:
            sep = 28

        img_path = i[:sep].replace('\\','/') # path to img 
        img_name = i[sep:][:-4] # img name (without .png)
        
        print(f'Aumentando dados -> {img_name}')

        original = cv2.imread(i,0)
        img_mirror = cv2.flip(original, 1)
        cv2.imwrite(f'{img_path}{img_name}(flipped).png', img_mirror)
        ori_equalized = cv2.equalizeHist(original)
        cv2.imwrite(f'{img_path}{img_name}(equalized).png', ori_equalized)
        mirror_equalized = cv2.equalizeHist(img_mirror)
        cv2.imwrite(f'{img_path}{img_name}(flipped_equalized).png', mirror_equalized)


# split_classes(True)
data_augmentation(True)

