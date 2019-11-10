import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np

BG_HUE = (90,105)
BG_SAT = (0,255)
BG_VAL = (0,255)

TARGET_RADIUS = 150 #in pixel

def show(im):
    cv2.imshow('', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mshow(im, isgray=True):
    if(isgray):
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)
    plt.show()


def get_target_frame(frame):
        
    blur = cv2.medianBlur(frame, (2*int(0.05*frame.shape[0]/2)) + 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    def get_largest_contour(contours):
        if len(contours)==0:
            return None
        l = contours[0]
        l_area = cv2.contourArea(l)
        for c in contours[1:]:
            a = cv2.contourArea(c)
            if a > l_area:
                l = c
                l_area = a
        return l

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = np.array(((hsv[:,:,0]>BG_HUE[0])&(hsv[:,:,0]<BG_HUE[1])&\
                     (hsv[:,:,1]>BG_SAT[0])&(hsv[:,:,1]<BG_SAT[1])&\
                     (hsv[:,:,2]>BG_VAL[0])&(hsv[:,:,2]<BG_VAL[1]))*255,\
                    dtype=np.uint8)

    kernel_size = int(round(0.03*max(mask.shape)))
    kernel = np.ones( (kernel_size, kernel_size) , mask.dtype)
    mask = cv2.medianBlur(mask, (2*int(0.05*mask.shape[0]/2)) + 1)

    _, mask_cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask_cont = get_largest_contour(mask_cont)
    (x,y), r = cv2.minEnclosingCircle(mask_cont)

    x, y, r = (int(round(x)), int(round(y)), int(round(r)))

    targetFrame = frame[y-r:y+r, x-r:x+r, :]

    return cv2.resize(targetFrame, (2*TARGET_RADIUS, 2*TARGET_RADIUS))

def apply_mask(targetFrame):
    bgcolour_frame = np.zeros((2*TARGET_RADIUS, 2*TARGET_RADIUS), dtype=targetFrame.dtype)
    mask = np.zeros((2*TARGET_RADIUS, 2*TARGET_RADIUS), dtype=targetFrame.dtype)
    cv2.circle(mask, (TARGET_RADIUS,TARGET_RADIUS), TARGET_RADIUS, 255, -1)
    cv2.circle(mask, (TARGET_RADIUS,TARGET_RADIUS), TARGET_RADIUS/3, 0, -1)
    masked = cv2.bitwise_and(targetFrame, targetFrame, mask = mask)
    return masked


frame = cv2.imread('asdf02.jpg')
targetFrame = get_target_frame(frame)
masked = apply_mask(targetFrame)
masked = cv2.medianBlur(masked, 11)
hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
hue_space = hsv[:,:,0]



    
