import serial
import time

DEBUG_NOSERIAL = False

#SPR = Steps Per Revolution
BASE_SPR = 1080
ARM_SPR = 1080

#All Coordinates theta, alpha, beta
currentPos = [None, None, None]
if not DEBUG_NOSERIAL:
    ser = serial.Serial('COM5', 9600)
first_sent = False

def sendCommand(command):

    global first_sent
    
    if command != '':
        print 'sending:', command

    numberOfCommands=0
    for c in command:
        if c in 'abcde':
            numberOfCommands+=1

    if not first_sent:
        time.sleep(3)
        first_sent = True
    if not DEBUG_NOSERIAL:
        ser.write(command)
        while(numberOfCommands>0):
            c = ser.read(size=1)
            if c=='k':
                numberOfCommands -= 1
            

def theta2steps(theta):
    steps = theta*BASE_SPR/360.0
    steps = int(steps)
    steps = steps + ((steps%7)%2)
    return steps

def alpha2steps(alpha):
    steps = alpha*ARM_SPR/360.0
    steps = int(steps)
    steps = steps + ((steps%7)%2)
    return steps

def initialise(theta=None, alpha=None):
    command = ''
    if theta != None:
        command += str(theta2steps(theta)) + 'd'
        currentPos[0] = theta
    if alpha != None:
        command += str(alpha2steps(alpha)) + 'e'
        currentPos[1] = alpha
        
    sendCommand(command)

def moveTo(theta=None, alpha=None, beta=None, hold=0):
    
    command = ''
    if theta != None:
        thstep = theta2steps(theta)
        if currentPos[0] != thstep:
            currentPos[0] = thstep
            command += str(thstep) + 'a'
    if alpha != None:
        alstep = alpha2steps(alpha)
        if currentPos[1] != alstep:
            currentPos[1] = alstep
            command += str(alstep) + 'b'
    if beta != None:
        beta = int(round(beta))
        if currentPos[2] != beta:
            currentPos[2] = beta
            command += str(beta) + 'c'

    sendCommand(command)
    time.sleep(hold)
    
def close():
    if not DEBUG_NOSERIAL:
        ser.close()
