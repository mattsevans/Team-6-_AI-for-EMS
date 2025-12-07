#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

BUTTON_PIN = 18  # Change to another GPIO if you decide not to use 10

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print(f"Monitoring GPIO {BUTTON_PIN} (pull-up enabled).")
print("Expected behavior:")
print("  - Not pressed  -> HIGH (1)")
print("  - Pressed      -> LOW  (0)")
print("Press Ctrl+C to exit.\n")

try:
    while True:
        state = GPIO.input(BUTTON_PIN)
        if state == GPIO.HIGH:
            print("GPIO state: HIGH (not pressed)")
        else:
            print("GPIO state: LOW  (BUTTON PRESSED)")
        time.sleep(2.0)

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    GPIO.cleanup()