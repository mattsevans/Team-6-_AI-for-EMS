import spidev
import time

# Create SPI object
spi = spidev.SpiDev()

# Open SPI bus 0, device 0 (CE0)
spi.open(0, 0)

# Configure SPI settings
spi.max_speed_hz = 500000      # 500 kHz
spi.mode = 0b00                # SPI mode 0
spi.bits_per_word = 8

def spi_write(data):
    """Send bytes to SPI slave"""
    spi.xfer2(data)
    print(f"Sent: {data}")

def spi_read(num_bytes):
    """Read bytes from SPI slave"""
    data = spi.readbytes(num_bytes)
    print(f"Received: {data}")
    return data

def spi_write_read(data):
    """Full-duplex transfer"""
    resp = spi.xfer2(data)
    print(f"Sent: {data}, Received: {resp}")
    return resp

try:
    while True:
        # Example transaction
        spi_write_read([0x01, 0x02, 0x03])
        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting program")

finally:
    spi.close()
