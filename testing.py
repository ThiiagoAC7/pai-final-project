"""
TESTING FUNCTIONS
- can be ignored
- functions used to test segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def transform_histogram(img, threshold):
    transformed_img = np.where(img <= threshold, 0, img - threshold)
    plt.imshow(transformed_img, cmap='gray')
    plt.show()
    # return transformed_histogram

def gamma_correction(img):
    gamma = 1.9
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    img_gamma_corrected = cv2.LUT(img, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_compare = cv2.hconcat([img, img_gamma_corrected])
    return img_compare

def plot_histogram(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    
    # bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # t, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    # histogram, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    # transform_histogram(img, 102.0)
    
    h,s,v = cv2.split(img)
    
    # fig,axs = plt.subplots(1,3, figsize=(20,10))
    # axs[0].set_title("H")
    # axs[0].imshow(h)
    # axs[0].axis('off')
    # axs[1].imshow(s)
    # axs[1].set_title("S")
    # axs[1].axis('off')
    # axs[2].set_title("V")
    # axs[2].imshow(v)
    # axs[2].axis('off')
    
    plt.imshow(v)
    plt.colorbar()
    plt.figure(figsize=(10, 6))
    plt.hist(img.flatten(), bins=256, range=[0, 256], color='red')
    plt.title(f'Histograma {img_path}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()



def aspect_ratio():
    imgs = glob.glob('./dataset_segmented/train/**/*.png')

    for i in imgs:
        img = cv2.imread(i,0)
        # print(f'aspect_ratio : {str(float(img.shape[0]) / float(img.shape[1]))}, {i}')
        print(f'aspect_ratio : {str(float(img.shape[0]))}  {str(float(img.shape[1]))}, {i}')



# aspect_ratio()
# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (66).png")]
# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (135).png")]
# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (152).png")]
# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (204).png")]
# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (141).png")]
# IMAGEM = IMGS[IMGS.index("./mamografias\DleftCC\d_left_cc (242).png")]
# IMAGEM = IMGS[111]
# IMAGEM = IMGS[3]
# IMAGEM = IMGS[IMGS.index("./mamografias\DrightCC\d_right_cc (1).png")]
# plot_histogram(IMAGEM)
