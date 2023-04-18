# arduino_LED_user.py

import serial
import time

# Define the serial port and baud rate.
# Ensure the 'COM#' corresponds to what was seen in the Windows Device Manager
ser = serial.Serial('COM3', 9600)

def led_on_off():
    user_input = input("\n Type S,L,R,T,N, quit : ")
    if user_input =="S":
        print("STOP")
        time.sleep(0.1)
        ser.write(b'S')
        led_on_off()
    elif user_input == "L":
        print("LEFT")
        time.sleep(0.1)
        ser.write(b'L')
        led_on_off()
    elif user_input == "R":
        print("RIGHT")
        time.sleep(0.1)
        ser.write(b'R')
        led_on_off()
    elif user_input == "N":
        print("NEUTRAL")
        time.sleep(0.1)
        ser.write(b'N')
        led_on_off()
    elif user_input == "T":
        print("THANKS")
        time.sleep(0.1)
        ser.write(b'T')
        led_on_off()
    elif user_input =="quit" or user_input == "q":
        print("Program Exiting")
        time.sleep(0.1)
        ser.write(b'L')
        ser.close()
    else:
        print("Invalid input. Type on / off / quit.")
        led_on_off()

    # if user_input =="on":
    #     print("LED is on...")
    #     time.sleep(0.1)
    #     ser.write(b'H')
    #     led_on_off()
    # elif user_input =="off":
    #     print("LED is off...")
    #     time.sleep(0.1)
    #     ser.write(b'L')
    #     led_on_off()
    # elif user_input =="quit" or user_input == "q":
    #     print("Program Exiting")
    #     time.sleep(0.1)
    #     ser.write(b'L')
    #     ser.close()
    # else:
    #     print("Invalid input. Type on / off / quit.")
    #     led_on_off()

time.sleep(2) # wait for the serial connection to initialize

led_on_off()