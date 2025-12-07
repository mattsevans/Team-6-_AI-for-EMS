#!/usr/bin/env python3
"""
play_wav_to_dac3.py — SPI WAV/sine player for MCP4822 DAC on Raspberry Pi

Device: Microchip MCP4822 (dual-channel 12-bit DAC)
SPI:    Mode 0 (CPOL=0, CPHA=0), 16 bits per DAC update (two bytes)
Default: 16 kHz playback (16-bit PCM WAV)

Wiring (typical):
  Pi 3.3V  -> MCP4822 VDD (and VREF if applicable)
  Pi GND   -> MCP4822 GND (AVSS)
  Pi SCLK  -> MCP4822 SCK
  Pi MOSI  -> MCP4822 SDI
  Pi CE1   -> MCP4822 CS (chip select)
  (Tie MCP4822 LDAC low to latch outputs on each command, or control via a GPIO if required)

Timing modes (select with --per-sample):

1) Chunked mode (default, --per-sample NOT set):
   - Split samples into chunks (e.g. 512 samples).
   - For each chunk:
       * Send EACH sample as its own spi.xfer2([hi, lo]) → CS pulses per 16-bit frame.
       * Then sleep the remaining chunk duration so the *average* playback rate
         matches --rate (e.g. 16 kHz).
   - Pros: good average timing accuracy (pitch/speed correct), CS pulses per frame.
   - Cons: SPI is bursty; DAC holds the last sample between chunks.

2) Per-sample mode (--per-sample):
   - Send exactly one spi.xfer2([hi, lo]) per sample in a tight loop (no sleep).
   - SPI clock is chosen so spi_speed_hz ≈ 16 * rate by default.
   - Pros: SPI activity is nearly continuous; one burst per sample.
   - Cons: Python + spidev overhead slows actual effective sample rate;
     actual playback rate < --rate unless compensated.
"""

import argparse
import sys
import time
import numpy as np
import wave

try:
    import spidev
except ImportError:
    raise RuntimeError("Install spidev on the Pi: sudo apt-get install -y python3-spidev")


def open_dac_spi(spi_bus: int, spi_ce: int, spi_mode: int, spi_speed_hz: int):
    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_ce)
    spi.max_speed_hz = spi_speed_hz
    spi.mode = spi_mode
    spi.bits_per_word = 8
    return spi


def read_wav_frames(path: str, expected_rate: int = None):
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels()
        sampw = wf.getsampwidth()
        rate = wf.getframerate()
        if sampw != 2:
            raise RuntimeError(f"Only 16-bit PCM WAV files are supported (got {sampw * 8}-bit).")
        if nchan != 1:
            raise RuntimeError(f"Only mono WAV files are supported (got {nchan} channels).")
        if expected_rate is not None and rate != expected_rate:
            print(
                f"Warning: WAV file sample rate ({rate} Hz) != --rate ({expected_rate} Hz). "
                f"Using WAV file rate for timing assumptions.",
                file=sys.stderr,
            )
        block_size = 4096
        while True:
            raw = wf.readframes(block_size)
            if not raw:
                break
            data = np.frombuffer(raw, dtype="<i2")
            yield data, rate


def i16_to_u12_stream(x_i16: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """
    Map int16 audio samples to 12-bit unsigned DAC codes centered at midscale (2048).
    """
    x = x_i16.astype(np.int32)
    x = x / 32768.0
    amp = headroom * (4095.0 / 2.0)
    codes = np.round(2048.0 + amp * x).astype(np.int32)
    np.clip(codes, 0, 4095, out=codes)
    return codes.astype(np.uint16)


def generate_sine_wave(frequency: float, duration: float, sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Generate a sine wave as int16 samples (0.9 full-scale) at the given sample_rate.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.9 * 32767 * np.sin(2 * np.pi * frequency * t)
    return waveform.astype(np.int16), sample_rate


def play_wav_to_dac(
    wav_path: str,
    sample_rate: int = 16000,
    spi_bus: int = 0,
    spi_ce: int = 1,
    spi_mode: int = 0,
    spi_speed_hz: int = 256000,
    gain_x2: bool = False,
    chunk_samp: int = 512,
    per_sample: bool = False,
    sine_test: tuple[float, float] = None,
):
    """
    Stream a WAV file or generated sine wave to the MCP4822 DAC via SPI.

    If per_sample is False (default): use chunked+sleep mode, but still send one
    16-bit frame per spi.xfer2 so CS pulses per sample.

    If per_sample is True: use per-sample mode (one spi.xfer2 per sample, no sleep).
    """
    spi = open_dac_spi(spi_bus, spi_ce, spi_mode, spi_speed_hz)

    def pack_word(u12: int, shdn: bool = True, ch: str = "A"):
        """
        Pack a 12-bit DAC code and control bits into a 16-bit MCP4822 frame.
        """
        u12 = u12 & 0x0FFF
        ab = 0 if ch.upper() == "A" else 1
        ga = 0 if gain_x2 else 1  # GA=0 => 2x gain, GA=1 => 1x
        sd = 1 if shdn else 0     # SHDN=1 => active, 0 => shutdown
        word = (ab << 15) | (0 << 14) | (ga << 13) | (sd << 12) | u12
        return int((word >> 8) & 0xFF), int(word & 0xFF)

    # Shut down channel B once up front
    hi, lo = pack_word(0, shdn=False, ch="B")
    spi.xfer2([hi, lo])

    # Source of samples: sine test vs WAV
    if sine_test:
        label = f"{sine_test[0]}Hz sine wave for {sine_test[1]}s"
        block_i16, fs = generate_sine_wave(sine_test[0], sine_test[1], sample_rate)
        data_blocks = [(block_i16, fs)]
    else:
        label = wav_path
        data_blocks = read_wav_frames(wav_path, expected_rate=sample_rate)

    print(
        f"Playing {label} via MCP4822 on /dev/spidev{spi_bus}.{spi_ce} "
        f"@ {spi_speed_hz} Hz (mode {spi_mode})"
    )
    if per_sample:
        eff_rate_est = spi_speed_hz / 16.0
        print(
            f"Mode: per-sample (one 16-bit frame per sample, no sleep). "
            f"Estimated max sample rate ≈ {eff_rate_est:.1f} Hz "
            "(actual may be lower due to Python/spidev overhead)."
        )
    else:
        print(
            f"Mode: chunked+sleep (chunk={chunk_samp} samples). "
            f"Target average sample rate ≈ {sample_rate} Hz. "
            "Each sample still sent via its own spi.xfer2 → CS pulses per frame."
        )
    print(f"Gain={'2x' if gain_x2 else '1x'}; output channel=A")

    if not per_sample:
        sample_period = 1.0 / float(sample_rate)
        next_deadline = time.perf_counter()

    try:
        if per_sample:
            # Option A: per-sample updates, SPI clock plus Python overhead sets effective sample rate.
            for block_i16, fs in data_blocks:
                codes = i16_to_u12_stream(block_i16)
                for code in codes:
                    hi, lo = pack_word(code, shdn=True, ch="A")
                    spi.xfer2([hi, lo])
        else:
            # Chunked mode with per-sample scheduling:
            # - chunk_samp controls how many samples we convert at once,
            #   but timing is enforced per sample using next_deadline.
            for block_i16, fs in data_blocks:
                codes = i16_to_u12_stream(block_i16)
                n = len(codes)
                i = 0
                while i < n:
                    j = min(i + chunk_samp, n)
                    segment = codes[i:j]

                    for code in segment:
                        # Wait until the next scheduled sample time
                        now = time.perf_counter()
                        dt = next_deadline - now
                        if dt > 0:
                            # Sleep the whole remaining time (coarse but simple)
                            time.sleep(dt)
                        # Output this sample
                        hi, lo = pack_word(code, shdn=True, ch="A")
                        spi.xfer2([hi, lo])

                        # Schedule the next sample time
                        next_deadline += sample_period

                    i = j


    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        spi.close()
        print("Done.")


def main():
    parser = argparse.ArgumentParser(description="SPI WAV/sine player for MCP4822 DAC")
    parser.add_argument("-w", "--wav", default="adc_capture.wav", help="Input WAV file path")
    parser.add_argument("-r", "--rate", type=int, default=16000, help="Nominal sample rate (Hz)")
    parser.add_argument("--bus", type=int, default=0, help="SPI bus")
    parser.add_argument("--ce", type=int, default=1, help="SPI chip select")
    parser.add_argument("--mode", type=int, default=0, help="SPI mode")
    parser.add_argument(
        "--speed",
        type=int,
        default=0,
        help=(
            "SPI speed (Hz). "
            "Default in per-sample mode: 16 * rate; "
            "default in chunked mode: 1_000_000."
        ),
    )
    parser.add_argument("--gain-x2", action="store_true", help="Enable 2x gain mode")
    parser.add_argument(
        "--chunk-samp",
        type=int,
        default=512,
        help="Samples per SPI chunk in chunked mode [default: 512].",
    )
    parser.add_argument(
        "--per-sample",
        action="store_true",
        help="Use per-sample SPI timing (SPI clock sets effective sample rate).",
    )
    parser.add_argument(
        "--sine-test",
        nargs=2,
        type=float,
        metavar=("FREQ_HZ", "DURATION_SEC"),
        help="Play a test sine wave instead of a WAV file",
    )
    args = parser.parse_args()
    sine_params = tuple(args.sine_test) if args.sine_test else None

    # Choose SPI speed based on timing mode
    if args.per_sample:
        # Option A: one 16-bit frame per sample
        spi_speed = args.speed if args.speed > 0 else (16 * args.rate)
    else:
        # Chunked: high SPI speed, use sleep to hit target avg rate
        spi_speed = args.speed if args.speed > 0 else 1_000_000

    play_wav_to_dac(
        wav_path=args.wav,
        sample_rate=args.rate,
        spi_bus=args.bus,
        spi_ce=args.ce,
        spi_mode=args.mode,
        spi_speed_hz=spi_speed,
        gain_x2=args.gain_x2,
        chunk_samp=args.chunk_samp,
        per_sample=args.per_sample,
        sine_test=sine_params,
    )


if __name__ == "__main__":
    main()
