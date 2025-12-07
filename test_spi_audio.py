# test_spi_audio.py
import signal, sys
import numpy as np
from spi_audio import SpiAudioIO

def main():
    io = SpiAudioIO(sample_rate=16000, frame_ms=20, spi_speed_hz=1_000_000)
    io.start()

    def on_sigint(sig, frame):
        io.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    print("SPI audio loopback running. Ctrl+C to stop.")
    for frame in io.frames():
        # Optionally apply gain or AGC here
        io.write_frame(frame)

if __name__ == "__main__":
    main()

