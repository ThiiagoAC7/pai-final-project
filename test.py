import cv2
import numpy as np
import glob

IMGS = glob.glob('./mamografias/**/*.png')
WINNAME = "Processamento e Análise de Imagens"

index = 0
contrast = 1.0
zoom = 1.0

def update_contrast(val):
    global contrast
    contrast = val / 100
    curr_img = cv2.imread(IMGS[index])
    cv2.imshow(WINNAME, cv2.convertScaleAbs(curr_img, alpha=contrast))

def update_zoom(val):
    pass  


def main():
    global index

    cv2.namedWindow(WINNAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Contraste", WINNAME, 100, 200, update_contrast)
    cv2.createTrackbar("Zoom", WINNAME, 100, 200, update_zoom)

    while True:
        curr_img = cv2.imread(IMGS[index])
        curr_img = cv2.convertScaleAbs(curr_img, alpha=contrast)

        cv2.imshow(WINNAME, curr_img)

        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:
            break
        elif key == ord('n') or key == ord('d'):
            index = (index + 1) % len(IMGS)
        elif key == ord('p') or key == ord('a'):
            index = (index - 1) % len(IMGS)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
