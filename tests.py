import numpy as np
import cv2
import matplotlib.pyplot as plt

def mshow(im, isgray=True):
    if(isgray):
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)
    plt.show()

im = cv2.cvtColor(cv2.imread('asdf03.jpg'), cv2.COLOR_BGR2HSV)
mshow(im)
