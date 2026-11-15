#include <Arduino.h>


void setup() {
  Serial.begin(115200);
}

void loop() {
  int FSR = analogRead(33);
  Serial.print(">fsr:");
  Serial.println(FSR);
  delay(20);
}