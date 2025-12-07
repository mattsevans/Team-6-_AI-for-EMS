# spi_clock_probe.py
import spidev, time

spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1_000_000
spi.mode = 0
spi.bits_per_word = 8

print("Arming… start your scope, then press Enter to fire one transfer.")
input()
t0 = time.time()
rx = spi.xfer2([0x00] * 1024)  # ~8 Kbits @ 1 MHz ~ 8 ms burst
t1 = time.time()
print(f"Transfer took {1000*(t1-t0):.2f} ms")
