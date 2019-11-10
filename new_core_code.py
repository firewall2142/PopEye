import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from random import randint
from math import atan, acos, asin, sin, cos, pi, tan
from pprint import pprint

DEBUG = True

UPPER_HUE_THRESH = 105
LOWER_HUE_THRESH = 95

BG_HUE = (LOWER_HUE_THRESH, UPPER_HUE_THRESH)
BG_SAT = (0,150)
BG_VAL = (0,255)
BLUE_SAT_THRESHOLD = 70

'''
BALLOON_COLOUR = {'blue': ((90, 90, 80), (105, 255, 255)),
                  'red': ((155, 0, 80),(15, 255, 255)), 'green':((50, 0, 80),(70,255,255)),
                  'yellow':((20,0,80),(48,255,255)), 'black':((0,0,0),(255,255,80))}
BALLOON_BLUE = ((80, 90, 80), (180, 255, 255))
'''

#Actual game
BALLOON_COLOUR = {'blue': ((85, 160, 120), (180, 255, 160)),
                  'red': ((155, 30, 120),(10, 255, 255)), 'green':((65, 30, 90),(82,255,255)),
                  'yellow':((40,30,120),(60,255,255)), 'black':((0,0,0),(255,255,90))}

BALLOON_BLUE = ((80, 90, 80), (180, 255, 255))

#Arm parameters in cm
ARM_A = 30          #base arm length
ARM_B = 30          #end arm length
ARM_OFF_X = 5       #distance of base arm from centre
ARM_PLANE_Z = -3     #z coordinate of plane

DEFAULT_BETA = 100

TARGET_RADIUS = 300 #in pixel

FRAME_PADDING = TARGET_RADIUS/3
MORPH_KERNEL_SIZE = 8
MINIMUM_CIRC_TO_RECT_AREA_THRESHOLD = (3.14159/16)*2.5
THRESH_OBJECT_AREA = (10)*(2*TARGET_RADIUS/90.)**2
SAMPLE_SQUARE_SIZE = 10



HUE_SPACE_SHIFT = 0

multiplot_plots = []

def classify_colour(colour):
    '''
    black- value < 80
    blue 90, 110
    red- 155-10
    yellow-35-45
    green-45-70
    '''

    BLACK_THRESH_VAL = 80
    blue=(90,110)
    red =(155,15)
    yellow=(35,48)
    green=(50,70)

    hue, sat, val = colour
    if val < BLACK_THRESH_VAL:
        return 'black'
    elif blue[0] < hue <= blue[1]:
        return 'blue'
    elif hue>red[0] or hue<= red[1]:
        return 'red'
    elif yellow[0] < hue <= yellow[1]:
        return 'yellow'
    elif green[0] < hue <= green[1]:
        return 'green'
    else:
        print 'BAD COLOUR: ', colour
        return None


def show(im, winname = ''):
    cv2.imshow(winname, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mshow(im, title = '', cvtColor=False, isgray=False):
    if cvtColor:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if(isgray):
        plt.imshow(im, cmap = 'gray')
    else:
        plt.imshow(im)
    plt.title(title)
    plt.show()

def cvtCord_im2cart(ix, iy):
    x = (ix - TARGET_RADIUS)*(45.0/TARGET_RADIUS)
    y = (TARGET_RADIUS - iy)*(45.0/TARGET_RADIUS)

    return x, y

def cvtCord_cart2arm(x, y, z):

    a, b = ARM_A, ARM_B

    x, y, z = float(x), float(y), float(z)
    
    if x == 0:
        if y > 0:
            theta = pi/2
        else:
            theta = 3*pi/2
    else:
        theta = atan(y/x)
        if x < 0:
            theta += pi
        elif y < 0:
            theta += 2*pi

    d = pow(x**2 + y**2, 0.5) - ARM_OFF_X
    c = pow(z**2 + d**2, 0.5)

    alpha = (asin((a**2 + c**2 - b**2)/(2*a*c)) - atan(z/d))
    beta = acos((a**2 + b**2 - c**2)/(2*a*b))

    return (theta)*180/pi, alpha*180/pi, beta*180/pi

def cvtCord_im2arm(ix, iy):
    x, y = cvtCord_im2cart(ix, iy)
    z = ARM_PLANE_Z
    return cvtCord_cart2arm(x, y, z)

def classify_targets(targets):

    classified = {}

    for target in targets:
        colour = classify_colour(target[1])
        if colour == None:
            continue
        else:
            if colour in classified.keys():
                classified[colour].append(target[0])
            else:
                classified[colour] = [target[0]]

    return classified

def multiplot_add(im, title=''):
    global multiplot_plots
    multiplot_plots.append((np.array(im), title))

def multiplot_show():
    n = len(multiplot_plots)
    row = int(n**0.5)
    col = n/row
    if row*col != n:
        col += 1
    for i in xrange(n):
        plt.subplot(row, col, i+1)
        plt.imshow(multiplot_plots[i][0])
        plt.title(multiplot_plots[i][1])

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

    _, mask_cont, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,\
                                               cv2.CHAIN_APPROX_NONE)

    mask_cont = get_largest_contour(mask_cont)

    #cv2.drawContours(frame, [mask_cont], 0, (0,0,0))
    #show(frame)
    
    (x,y), r = cv2.minEnclosingCircle(mask_cont)
    
    x, y, r = (int(round(x)), int(round(y)), int(round(r)))

    #cv2.circle(frame, (x,y), r, (0,0,0))
    #cv2.circle(frame, (x,y), 10, (0,0,255), -1)
    #show(frame)

    h, w = frame.shape[0], frame.shape[1]

    frame_padded = np.zeros((frame.shape[0] + 2*FRAME_PADDING, frame.shape[1] + 2*FRAME_PADDING, 3), dtype=np.uint8)
    bg_color = (sum(BG_HUE)/2, sum(BG_SAT)/2, sum(BG_VAL)/1.5)
    frame_padded[:,:] = bg_color
    frame_padded = cv2.cvtColor(frame_padded, cv2.COLOR_HSV2BGR)
    frame_padded[FRAME_PADDING:FRAME_PADDING+h, FRAME_PADDING:FRAME_PADDING+w] = frame
    #show(frame_padded)
        
    targetFrame = frame_padded[FRAME_PADDING+y-r:FRAME_PADDING+y+r, FRAME_PADDING+x-r:FRAME_PADDING+x+r ,:]
    #show(targetFrame)
    return cv2.resize(targetFrame, (2*TARGET_RADIUS, 2*TARGET_RADIUS))

def get_mask(complement = False):
    mask = np.zeros((2*TARGET_RADIUS, 2*TARGET_RADIUS), dtype=np.uint8)
    cv2.circle(mask, (TARGET_RADIUS,TARGET_RADIUS), TARGET_RADIUS, 255, -1)
    cv2.circle(mask, (TARGET_RADIUS,TARGET_RADIUS), TARGET_RADIUS/3, 0, -1)
    if complement:
        mask = ~mask
    return mask

def apply_mask(targetFrame):
    bgcolour_frame = np.zeros((2*TARGET_RADIUS, 2*TARGET_RADIUS),\
                              dtype=np.uint8)
    mask = get_mask()
    masked = cv2.bitwise_and(targetFrame, targetFrame, mask = mask)
    return masked

def fillbg(im):
    mask = get_mask()
    bg_hue = sum(BG_HUE)/2
    bg_image = np.zeros((2*TARGET_RADIUS, 2*TARGET_RADIUS), dtype=np.uint8)
    bg_image[:,:] = bg_hue
    im = cv2.bitwise_and(im, im, mask=mask)
    im = cv2.add(im, cv2.bitwise_or(im, bg_image, mask=~mask))
    return im

def centre_hue_to_bg(h_chnl):
    #H val is b/w 0 and 180 in opencv
    global HUE_SPACE_SHIFT
    bg_hue = sum(BG_HUE)/2
    shift = 90- bg_hue #(old hue + shift = new hue)
    if shift < 0:
        shift += 180
    HUE_SPACE_SHIFT = shift

    h_chnl = np.array((np.array(h_chnl, dtype=np.uint32) + shift)%180, dtype=np.uint8)

    return h_chnl

def is_contour_valid(cont): #tests if the contour is okay to be a balloon/square
    minCircArea = 3.14159*cv2.minEnclosingCircle(cont)[1]**2
    contArea = cv2.contourArea(cont)

    if DEBUG:
        print 'contourArea, contAr/minCircArea :', contArea, contArea/minCircArea,
        print 'THRESHOLDS', THRESH_OBJECT_AREA, MINIMUM_CIRC_TO_RECT_AREA_THRESHOLD

    
    if contArea > THRESH_OBJECT_AREA and contArea/minCircArea > MINIMUM_CIRC_TO_RECT_AREA_THRESHOLD:
        return True
    else:
        return False

def get_object_points_from_raw(frame, giveCart=False):
    final_objects = {}

    for colour in BALLOON_COLOUR:
        final_objects[colour] = []
    
    fg_conts = []

    target_frame = get_target_frame(frame)
    #target_frame = cv2.medianBlur(target_frame, 21) #try bilateral filtering
    target_frame = cv2.bilateralFilter(target_frame, 10, 30, 30)

    multiplot_add(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB), 'original')
    
    target_frame_hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)

    for colour in BALLOON_COLOUR:
        fore = cv2.inRange(target_frame_hsv, BALLOON_COLOUR[colour][0], BALLOON_COLOUR[colour][1])

        if colour == 'red':
            print BALLOON_COLOUR['red'][0][0]
            fore = np.array(((target_frame_hsv[:,:,0]>BALLOON_COLOUR['red'][0][0])|(target_frame_hsv[:,:,0]<BALLOON_COLOUR['red'][1][0]))&\
                             (target_frame_hsv[:,:,1]>BALLOON_COLOUR['red'][0][1])&(target_frame_hsv[:,:,1]<BALLOON_COLOUR['red'][1][1])&\
                            (target_frame_hsv[:,:,2]>BALLOON_COLOUR['red'][0][2])&(target_frame_hsv[:,:,2]<BALLOON_COLOUR['red'][1][2]), dtype=np.uint8)*255
        
        #fore = cv2.medianBlur(fore, 5)
        fore = cv2.bitwise_and(fore, fore, mask=get_mask())
        fore = cv2.morphologyEx(fore, cv2.MORPH_OPEN, np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), dtype=np.uint8))
        multiplot_add(fore, colour+' new fore')
        conts = cv2.findContours(fore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
        result = cv2.cvtColor(fore, cv2.COLOR_GRAY2RGB)
        for cont in conts:
            if is_contour_valid(cont):
                M = cv2.moments(cont)
                if giveCart:
                    final_objects[colour].append((M['m10']/M['m00'], M['m01']/M['m00']))
                else:
                    final_objects[colour].append(cvtCord_im2arm(M['m10']/M['m00'], M['m01']/M['m00']))
        
    multiplot_show()
    
    return final_objects

    multiplot_add(target_frame_hsv, 'original hsv')
    
    hue_channel = target_frame_hsv[:,:,0]
    hue_channel = fillbg(hue_channel)

    multiplot_add(hue_channel, 'hue channel')

    #hue_channel = cv2.morphologyEx(hue_channel,cv2.MORPH_OPEN, np.ones((MORPH_KERNEL_SIZE,MORPH_KERNEL_SIZE)))
    #hue_channel = cv2.morphologyEx(hue_channel,cv2.MORPH_CLOSE, np.ones((MORPH_KERNEL_SIZE,MORPH_KERNEL_SIZE)))

    foreground = np.array((hue_channel>((UPPER_HUE_THRESH + HUE_SPACE_SHIFT)%180))|(hue_channel<((LOWER_HUE_THRESH + HUE_SPACE_SHIFT)%180)), dtype=np.uint8)*255

    multiplot_add(foreground, 'fore orig')    

    
    foreground = cv2.morphologyEx(foreground,cv2.MORPH_OPEN, np.ones((MORPH_KERNEL_SIZE,MORPH_KERNEL_SIZE)))

    multiplot_add(foreground, 'fore morphed')

    #finding blue balloons


    
    #finalising contours from foreground
    fg_conts.extend(cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1])
    result = np.array(target_frame_hsv, dtype=np.uint8)

    for colour in final_objects:
        
        
        if DEBUG:
            cv2.circle(result, (cx,cy), 10, (hue,sat,val), -1)
            cv2.circle(result, (cx,cy), 10, (0,0,0), 3)

    if DEBUG:
        cv2.circle(result, (TARGET_RADIUS, TARGET_RADIUS), 3, (0,0,0), -1)
        mshow(cv2.cvtColor(result, cv2.COLOR_HSV2RGB))
        multiplot_show()
    
    final_objects = classify_targets(final_objects)
    
    return final_objects


if __name__ == '__main__':
    frame = cv2.imread('round_image.jpg')
    objs = get_object_points_from_raw(frame)
    print objs
    

