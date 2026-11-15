# for testing if fsr works
import serial

ser = serial.Serial("COM15", 115200, timeout = 1)

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line.startswith(">fsr:"):
        value = int(line.split(":")[1])
        touching = value > 1000
        print(value, "TOUCHING" if touching else "")