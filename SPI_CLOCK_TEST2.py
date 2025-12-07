# spi_clock_burner.py
import spidev, time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1_000_000
spi.mode = 0
spi.bits_per_word = 8

payload = [0xFF] * 4096  # 4 KB per transfer
print("Burning SCLK continuously at ~1 MHz. Ctrl+C to stop.")
try:
    while True:
        spi.xfer2(payload)  # CS stays low for the whole buffer, SCLK runs continuously
        # no sleep → near continuous clocking
except KeyboardInterrupt:
    pass
