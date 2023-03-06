import cv2 as cv
import glob
import sys

'''
TODO:
    - Diminuir resolução das imagens para display
    - opencv buttons (?)
        - next/previous (?)
        - pause (?)
    - zoom (trackbar)
        - https://docs.opencv.org/4.x/da/d6a/tutorial_trackbar.html
    - https://github.com/Chetan-Goyal/Slideshow-using-Opencv-python/blob/master/main.py
    - https://analyticsindiamag.com/real-time-gui-interactions-with-opencv-in-python/
'''


def showImages(winname, path):
    for i in path:
        img = cv.imread(i)
        img = cv.resize(img, (256, 256))
        print(img.shape[0], img.shape[1])
        cv.imshow(winname, img)
        key = cv.waitKey(5000)
        if key == ord('q'):
            sys.exit(0)
    cv.destroyAllWindows()


def main():
    winname = 'window'
    cv.namedWindow(winname, cv.WINDOW_NORMAL)
    cv.resizeWindow(winname, 500, 500)
    path = glob.glob('./mamografias/**/*.png')
    # print(path[1])  # ./mamografias\DleftCC\d_left_cc (10).png
    # print(path[len(path)-1])  # ./mamografias\GrightMLO\g_right_mlo (99).png
    showImages(winname, path)


if __name__ == '__main__':
    main()
