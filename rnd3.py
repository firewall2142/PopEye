import armHandle
from new_core_code import *
import time

NUMBER_OF_TARGETS = 12
HOLD_TIME = 1.5

def do_third_round(targets):

    def get_dia_opp(iniCrd, targets):
        opp_theta = (iniCrd[0]+180)%360

        nearest = targets[0]
        minDiff = abs(opp_theta - nearest[0])
        for target in targets:
            diff = abs(opp_theta - target[0])
            if diff < minDiff:
                nearest = target
                minDiff = diff

        return nearest

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

    def get_combined_targets(targets, leaveBlack=True, removeLess=True):
        combined = []
        
        for colour in targets:
            if colour == 'black' and leaveBlack:
                continue
            if removeLess and len(targets[colour]) < 2:
                continue
            combined.extend(targets[colour])

        return combined

    def get_target_colour(targetList, target):
        for colour in targetList:
            if target in targetList[colour]:
                return colour

        raise ValueError("can't find target colour from target")

    iniPos = targets['black'][0]
    armHandle.initialise(theta=iniPos[0], alpha=0)
    pop(iniPos, HOLD_TIME)

    lastPos = iniPos
    combined = get_combined_targets(targets)

    while len(combined) >= 2:
        newTarget = get_nearest_target(lastPos, combined)
        colour = get_target_colour(targets, newTarget)
        print '\npopping', colour
        pop(newTarget, HOLD_TIME)
        lastPos = newTarget

        targets[colour].remove(newTarget)
        combined = get_combined_targets(targets)
        
        newTarget = get_dia_opp(newTarget, targets[colour])
        print 'popping', colour
        pop(newTarget, HOLD_TIME)
        lastPos = newTarget
        combined = get_combined_targets(targets)

        print '\ngoing back to black'
        pop(iniPos, HOLD_TIME)

    pop(iniPos, HOLD_TIME+3)

if __name__ == '__main__':
    frame = cv2.imread('round_image.jpg')
    targets = get_object_points_from_raw(frame)
    pprint(targets)
    do_third_round(targets)
                
