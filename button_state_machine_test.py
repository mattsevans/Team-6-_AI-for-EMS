#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time

BUTTON_PIN = 18  # BCM 18 = physical pin 12
LONG_PRESS_THRESHOLD = 5.0  # seconds

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print(f"Monitoring button on GPIO {BUTTON_PIN} (physical pin 12, pull-up enabled).")
print("Behavior:")
print("  - First button press -> Starting code")
print("  - Short press (< 5s) -> Toggle mode on/off")
print("  - Long press (>= 5s) -> Stopping code\n")
print("Press Ctrl+C to exit manually.\n")

started = False       # Has the code been 'started' yet?
mode_active = False   # Current mode state
pressed = False       # Are we currently in a pressed state?
press_start_time = None

try:
    while True:
        state = GPIO.input(BUTTON_PIN)  # HIGH = not pressed, LOW = pressed

        # Detect button press (edge: HIGH -> LOW)
        if not pressed and state == GPIO.LOW:
            pressed = True
            press_start_time = time.time()
            # (We don't print anything yet; we wait for release to measure duration)

        # Detect button release (edge: LOW -> HIGH)
        elif pressed and state == GPIO.HIGH:
            press_duration = time.time() - press_start_time
            pressed = False

            if not started:
                # First press: always start code regardless of duration
                started = True
                print(f"Button pressed for {press_duration:.2f}s -> Starting code")
            else:
                # After started, interpret duration
                if press_duration >= LONG_PRESS_THRESHOLD:
                    print(f"Button held for {press_duration:.2f}s -> Stopping code")
                    break  # Exit main loop
                else:
                    # Short press toggles mode
                    mode_active = not mode_active
                    if mode_active:
                        print(f"Button pressed for {press_duration:.2f}s -> Mode activated")
                    else:
                        print(f"Button pressed for {press_duration:.2f}s -> Mode deactivated")

        time.sleep(0.05)  # 50 ms polling / basic debounce

except KeyboardInterrupt:
    print("\nKeyboardInterrupt received -> Exiting...")

finally:
    GPIO.cleanup()
    print("GPIO cleaned up.")
