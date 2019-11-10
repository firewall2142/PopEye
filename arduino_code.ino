/* a - Stepper 1 set to val
 * b - Stepper 2 set to val
 * c - Servo set to val
 * d - set stepper1_position to val
 * e - set stepper2_position to val
 */


#include<Servo.h>

#define servoPin 6
#define stepper1Delay 5
#define stepper2Delay 5

int stepper1_pins[] = {7,9,8,10};
int stepper1_stepsPerRev = 400;
int stepper1_position = 0;

 int stepper2_pins[] = {4,2,5,3};
int stepper2_stepsPerRev = 400;
int stepper2_position = 0;


Servo servo1;

char commandTo = 'z';
int ang = 0;
int curpos = 0;
int pos;


const byte step_cycle[] = {B1000, B1100, B0100, B0110, B0010, B0011, B0001, B1001};
//const byte step_cycle[] = {B1000, B1000, B0100, B0100, B0010, B0010, B0001, B0001};

void setup(){

  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
 
  for(int i=0;i<4;i++){
    pinMode(stepper1_pins[i], OUTPUT);
    digitalWrite(stepper1_pins[i] , LOW);
  }
  digitalWrite(stepper1_pins[0], HIGH);

  
  for(int i=0;i<4;i++){
    pinMode(stepper2_pins[i], OUTPUT);
    digitalWrite(stepper2_pins[i] , LOW);
  }
  digitalWrite(stepper2_pins[0], HIGH);
  
  servo1.attach(servoPin);
  servo1.write(170);
  Serial.begin(9600); 
}

int getCommand(){
  char inChar = '$';
  int angle = 0;
  while(1){
    if(Serial.available()>0){
      digitalWrite(13, HIGH);
      inChar = (char)Serial.read();
      if(isAlpha(inChar)){
        commandTo = inChar;
        break;
      } else if (isDigit(inChar)){
        angle = angle*10 + (int)inChar - (int)'0';
      }
    }
  }
  return angle;
}

void doCommand(){
  int val = getCommand();
  
  if(commandTo == 'a'){
    step_motor(stepper1_pins, val, stepper1_position, stepper1Delay);
    stepper1_position = val;
    Serial.print('k');
  } else if(commandTo == 'b'){
    step_motor(stepper2_pins, val, stepper2_position, stepper2Delay);
    stepper2_position = val;
    Serial.print('k');
  } else if(commandTo == 'c'){
    servo1.write(val);
    Serial.print('k');
  } else if(commandTo == 'd'){
    stepper1_position = val;
    Serial.print('k');
  } else if(commandTo == 'e'){
    stepper2_position = val;
    Serial.print('k');
  }
  commandTo = 'z';
}

/*
void loop(){
  if(Serial.available() > 0){
    ang = getCommand();
    Serial.print("Moving to ");
    Serial.println(ang);
    Serial.print("Steps final step count ");
    Serial.println(pos=getFinalPosition(stepper1_stepsPerRev, ang, 0));
    step_motor(stepper1_pins, curpos, pos, 5);
    curpos = pos;
  }
}
*/

void loop(){
  doCommand();
}

/*pins[0], pins[1], pins[2] ... are inputs pins 1, 2, 3 ... of h-bridge
stepTime is in milliseconds
make sure stepTo and currentPos are positive*/
void step_motor(int pins[], int stepTo, int currentPos, int stepTime){  
  int i;
  short del;

  if(stepTo > currentPos){
    del = 1;
  } else {
    del = -1;
  }

  while(currentPos != stepTo){
    byte mod = currentPos%8;
    byte b = step_cycle[mod];
    for(i=0;i<4;i++){
      if(bitRead(b, i)){
        digitalWrite(pins[i], HIGH);
      } else {
        digitalWrite(pins[i], LOW);
      }
    }
    delay(stepTime);
    currentPos += del;
  }
}
