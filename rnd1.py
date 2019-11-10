import armHandle
from new_core_code import *
import time

HOLD_TIME = 3
DEBUG = False

def do_first_round(targets):

    def get_nearest_target(initial_cord, targets):
        '''coordinates must be arm coordinates'''
        nearest = targets[0]
        
        min_dist = sum([abs(initial_cord[i] - targets[0][i]) for i in xrange(3)])

        for target in targets:
            dist = sum([abs(initial_cord[i] - target[i]) for i in xrange(3)])
            if dist < min_dist:
                nearest = target
                min_dist = dist
        
        return nearest
        
    def pop(pos, holdTime):
        armHandle.moveTo(beta=DEFAULT_BETA)
        armHandle.moveTo(pos[0], pos[1], pos[2])
        time.sleep(holdTime)
        armHandle.moveTo(beta=DEFAULT_BETA)
    
    if DEBUG:
        pprint(targets)
    
    iniPos = targets['black'][0]
    armHandle.initialise(theta=iniPos[0], alpha=0)
    armHandle.moveTo(theta=iniPos[0]+2, alpha=2)
    
    pop(iniPos, HOLD_TIME)

    nxt_target = get_nearest_target(iniPos, targets['red'])
    targets['red'].remove(nxt_target)

    pop(nxt_target, HOLD_TIME)
    
    nxt_target = get_nearest_target(iniPos, targets['red'])
    targets['red'].remove(nxt_target)

    pop(nxt_target, HOLD_TIME)

    pop(iniPos, HOLD_TIME)


frame = cv2.imread('round_image.jpg')
targets = get_object_points_from_raw(frame)
print 'targets: ', targets
do_first_round(targets)

'''try:
    doRound = True
    print 'press c to cancel & q to start the round'
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            cv2.imshow('camera output', frame)
            wtkey = cv2.waitKey(1)&0xFF
            if wtkey == ord('q'):
                break
            if wtkey == ord('c'):
                doRound = False
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if doRound:
        frame=cv2.imread('fdsa.jpg') #REMOVE THIS
        targets = get_object_points_from_raw(frame)
        print 'targets: ', targets
        do_first_round(targets)
finally:
    armHandle.close()


'''
