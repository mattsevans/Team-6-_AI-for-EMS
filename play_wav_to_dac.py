#!/usr/bin/env python3
import sys, time, wave, struct
import numpy as np
import spidev

# ---------- USER SETTINGS ----------
WAV_PATH   = sys.argv[1] if len(sys.argv) > 1 else "test_audio\recording8_16k.wav"
BUS, CE    = 0, 1            # CE=1 -> spidev0.1 (change to 0 if on CE0)
SPI_HZ     = 1_000_000
SPI_MODE   = 0               # MCP4822 supports mode 0 or 1; data on rising edge
GAIN_X2    = False           # False = 1x (~0..2.048V FS), True = 2x (~0..4.096V FS, watch headroom)
CHUNK_SAMP = 1024            # samples per transfer (<= 2048 -> 4096 bytes)
# -----------------------------------

# MCP4822 16-bit frame:
# [15] A/B (0=A, 1=B)
# [14] don't care
# [13] GA (1=1x, 0=2x)
# [12] SHDN (1=active, 0=tristate)
# [11:0] 12-bit code
def pack_word(u12, shdn=True, ch='A'):
    u12 = int(u12) & 0x0FFF
    ab  = 0 if str(ch).upper() == 'A' else 1
    ga  = 0 if GAIN_X2 else 1      # GA=0 -> 2x, GA=1 -> 1x
    sd  = 1 if shdn else 0         # SHDN=0 -> output off (tristate)
    word = (ab << 15) | (0 << 14) | (ga << 13) | (sd << 12) | u12
    return (word >> 8) & 0xFF, word & 0xFF


def open_spi():
    spi = spidev.SpiDev()
    spi.open(BUS, CE)
    spi.max_speed_hz = SPI_HZ
    spi.mode = SPI_MODE
    spi.bits_per_word = 8
    return spi

def read_wav_frames(path):
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels()
        sampw = wf.getsampwidth()
        rate  = wf.getframerate()
        nfrm  = wf.getnframes()
        if sampw != 2:
            raise RuntimeError(f"Only 16-bit PCM supported (got {8*sampw}-bit).")
        # Read in moderate blocks to limit RAM
        block = 4096
        while True:
            raw = wf.readframes(block)
            if not raw:
                break
            # little-endian int16
            data = np.frombuffer(raw, dtype="<i2")
            if nchan == 2:
                # downmix to mono: average L/R
                data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            yield data, rate

def i16_to_u12_stream(x_i16, headroom=0.95):
    """
    Map int16 [-32768..32767] -> 12-bit around midscale (AC-coupled downstream).
    headroom < 1.0 prevents hitting rails (glitches near 0/4095).
    """
    x = x_i16.astype(np.int32)
    # Normalize to [-1.0, +1.0)
    x = x / 32768.0
    # Scale to ~±(4095/2) with headroom
    amp = headroom * (4095.0 / 2.0)
    codes = np.round(2048.0 + amp * x).astype(np.int32)
    np.clip(codes, 0, 4095, out=codes)
    return codes.astype(np.uint16)

def main():
    spi = open_spi()
    # --- Put channel B into shutdown (tristate) so it can't inject noise ---
    hi, lo = pack_word(0, shdn=False, ch='B')   # SHDN=0 on channel B
    spi.xfer2([hi, lo])
    # -----------------------------------------------------------------------
    print(f"Playing {WAV_PATH} via MCP4822 on /dev/spidev{BUS}.{CE} @ {SPI_HZ} Hz (mode {SPI_MODE})")
    print(f"Gain={'2x' if GAIN_X2 else '1x'}; channel=A; chunk={CHUNK_SAMP} samples")

    t_prev = time.perf_counter()
    for block_i16, fs in read_wav_frames(WAV_PATH):
        # Convert to u12 around midscale
        codes = i16_to_u12_stream(block_i16, headroom=0.95)

        # Pace by audio time
        # We'll send in CHUNK_SAMP groups and sleep to match realtime
        i = 0
        nsamp = len(codes)
        while i < nsamp:
            j = min(i + CHUNK_SAMP, nsamp)
            seg = codes[i:j]

            # pack whole segment into bytes
            tx = []
            for u in seg:
                hi, lo = pack_word(u, shdn=True)
                tx.append(hi); tx.append(lo)

            spi.xfer2(tx)

            # crude realtime pacing: sleep for segment duration minus time spent
            # duration seconds for this segment:
            seg_dur = (j - i) / float(fs)
            t_now = time.perf_counter()
            elapsed = t_now - t_prev
            sleep = seg_dur - elapsed
            if sleep > 0:
                time.sleep(sleep)
                t_prev = time.perf_counter()
            else:
                # we're late; reset timebase to avoid runaway drift
                t_prev = t_now

            i = j

    print("Done.")
    spi.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 play_wav_to_mcp4822.py <path/to/file.wav>")
        sys.exit(1)
    main()
