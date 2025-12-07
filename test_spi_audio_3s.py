# test_spi_audio_3Ss.py
import sys, signal, time
import numpy as np
from spi_audio import SpiAudioIO

SEC_BUFFER = 1.0  # seconds of audio per block you want to hear (can change to 2.0, 3.0, etc.)
SPI_SPEED  = 1_000_000  # Hz; raise to 2_000_000 if you see choppiness
SR         = 16000      # sample rate

def main():
    io = SpiAudioIO(sample_rate=SR, frame_ms=20, spi_speed_hz=SPI_SPEED)
    io.start()

    def on_sigint(sig, frame):
        io.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)
    print(f"SPI audio 1s loopback running @ {SR} Hz. Ctrl+C to stop.")

    # How many of the 20 ms frames make up ~1 second?
    frames_per_sec = int(round(SEC_BUFFER * SR / io.frame_len))
    block_samples  = frames_per_sec * io.frame_len

    # rolling counters for RMS printouts (once per second-ish)
    last_print = time.time()
    blocks_done = 0

    while True:
        # ---- CAPTURE: collect ~1 second into one big buffer ----
        buf_list = []
        for _ in range(frames_per_sec):
            f = io.read_frame(timeout=1.0)  # int16 mono
            buf_list.append(f)
        big_block = np.concatenate(buf_list).astype(np.int16)
        if big_block.size != block_samples:
            # shouldn't happen, but guard anyway
            big_block = big_block[:block_samples]

        # ---- STATS: compute and print RMS once per second ----
        # RMS on the captured second
        rms = float(np.sqrt(np.mean(big_block.astype(np.float32) ** 2)))
        blocks_done += 1
        now = time.time()
        # print every second OR every block (since block is ~1s)
        if (now - last_print) >= 1.0 or blocks_done % 1 == 0:
            peak = int(np.max(np.abs(big_block)))
            print(f"[{blocks_done:04d}] RMS={rms:7.1f}  peak={peak:5d}")
            last_print = now

        # ---- PLAYBACK: write that same second back out ----
        # We feed it back in 20 ms chunks so the playback thread can keep pace.
        # This introduces ~1 second latency (by design) so you can clearly hear it.
        for i in range(0, big_block.size, io.frame_len):
            io.write_frame(big_block[i:i+io.frame_len])

if __name__ == "__main__":
    main()
