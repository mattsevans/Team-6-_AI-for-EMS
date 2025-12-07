# spi_bus_probe.py
# Purpose: make SPI activity easy to see on a scope or logic analyzer,
# independent of any DAC behavior.

import spidev
import time

BUS = 0
CE  = 0        # 1 for CE1 (spidev0.1). Use 0 if DAC is on CE0.
HZ  = 2_000_000
MODE = 0
BATCH = 4096   # payload bytes per transfer (big = longer SCLK burst)

def main():
    spi = spidev.SpiDev()
    spi.open(BUS, CE)
    spi.max_speed_hz = HZ
    spi.mode = MODE
    spi.bits_per_word = 8

    # Pattern to make MOSI "look alive" (alternating, ramps)
    alt = [0xAA, 0x55] * (BATCH // 2)
    ramp = [i & 0xFF for i in range(BATCH)]

    print(f"SPI bus probe on /dev/spidev{BUS}.{CE} @ {HZ} Hz. Ctrl+C to stop.")
    print("Scope SCLK on GPIO11(pin23), MOSI on GPIO10(pin19), CE on GPIO7(pin26 for CE1) or GPIO8(pin24 for CE0).")

    try:
        while True:
            # Long burst #1
            spi.xfer2(alt)
            # Short idle so you can see CE high between bursts
            time.sleep(0.02)

            # Long burst #2
            spi.xfer2(ramp)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
