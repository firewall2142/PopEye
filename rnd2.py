import armHandle
from new_core_code import *
import time

NUMBER_OF_TARGETS = 5
HOLD_TIME = 1.5

def do_second_round(targets):

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

    def get_dia_opp(iniCrd, targets):
        opp_theta = (iniCrd[0]+180)%360
        nearest = targets[0]
        minDiff = abs(opp_theta - nearest[0])
        if minDiff > 180:
            minDiff = 360 - minDiff
        for target in targets:
            diff = abs(opp_theta - target[0])
            if diff > 180:
                diff = 360 - diff
            if diff < minDiff:
                nearest = target
                minDiff = diff

        return nearest
        
    def pop(pos, holdTime):
        armHandle.moveTo(beta=DEFAULT_BETA)
        armHandle.moveTo(pos[0], pos[1], pos[2])
        time.sleep(holdTime)
        armHandle.moveTo(beta=DEFAULT_BETA)

    
    iniPos = targets['black'][0]
    armHandle.initialise(theta=iniPos[0], alpha=0)

    armHandle.moveTo(iniPos[0]+2,2)
    
    armHandle.moveTo(iniPos[0], iniPos[1], iniPos[2])

    lastPos = iniPos

    target_counter = 0    

    #Red-Blue-green-yellow
    for target_counter in xrange(NUMBER_OF_TARGETS):
        
        colour = ('red', 'blue', 'green', 'yellow')[(target_counter)%4]
        
        if len(targets[colour]) == 0:
            print 'COMPLETED', target_counter, 'targets'
            break
        
        newTarget = get_nearest_target(lastPos, targets[colour])
        print 'popping', colour, newTarget[0]
        pop(newTarget, HOLD_TIME)

        if len(targets[colour]) == 0:
            print 'COMPLETED', target_counter, 'targets'
            break
        
        targets[colour].remove(newTarget)
        lastPos = newTarget
            
        newTarget = get_dia_opp(lastPos, targets[colour])
        print 'popping', colour, newTarget[0]
        pop(newTarget, HOLD_TIME)
        
        targets[colour].remove(newTarget)
        lastPos = newTarget


if __name__ == '__main__':
    frame = cv2.imread('round_image.jpg')
    targets = get_object_points_from_raw(frame)
    pprint(targets)
    do_second_round(targets)
            

