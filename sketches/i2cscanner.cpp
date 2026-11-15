#include <Wire.h> // The I2C library; the 2 wire protocol (SDA & SDL) your ESP32 uses to talk to the BNO085.
#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    Wire.begin(21, 22);
}

void Scanner() {
    Serial.println ();
    Serial.println("I2C scanner. Scanning ....");
    byte count = 0;

    for (byte i = 8; i < 120; i++) {
        Wire.beginTransmission(i);
        if (Wire.endTransmission() == 0)
        {
            Serial.print("Found address: ");
            Serial.print(i, DEC);
            Serial.print(" (0x");
            Serial.print(i, HEX);
            Serial.println (")");
            count++;
        }

    }
    Serial.print("Found ");
    Serial.print(count, DEC);
    Serial.println(" device(s).");                                                                                       
}

void loop() {
    Scanner();
    delay(1000);
}
