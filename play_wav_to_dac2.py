#!/usr/bin/env python3
"""
play_wav_to_dac.py — Minimal SPI WAV player for MCP4822 DAC on Raspberry Pi

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

Notes:
- Only 16-bit mono WAV files are supported. The WAV's sample rate should match the --rate parameter (default 16 kHz).
- If the WAV file's rate differs from --rate, playback will proceed at the file's rate (a warning is issued).
- The MCP4822 expects a 12-bit unsigned value; this script maps int16 audio samples (±32768) to 12-bit DAC codes (0–4095) centered at mid-scale (2048 ≈ 0 level).
- A headroom factor (0.95) is applied to avoid hitting the DAC's 0 or 4095 extremes, which can cause output glitches.
- Channel A is used for output. Channel B is shut down (tri-stated) by default to prevent any noise.
"""
import argparse
import time
import sys
import numpy as np
import wave

try:
    import spidev
except ImportError:
    raise RuntimeError("Install spidev on the Pi: sudo apt-get install -y python3-spidev")

def open_dac_spi(spi_bus: int, spi_ce: int, spi_mode: int, spi_speed_hz: int):
    """Open and configure the SPI bus for MCP4822 DAC."""
    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_ce)
    spi.max_speed_hz = spi_speed_hz
    spi.mode = spi_mode
    spi.bits_per_word = 8
    # Most Raspberry Pi setups are MSB-first by default, which is required by MCP4822.
    return spi

def read_wav_frames(path: str, expected_rate: int = None):
    """Generator that yields blocks of int16 samples from a WAV file."""
    with wave.open(path, "rb") as wf:
        nchan = wf.getnchannels()
        sampw = wf.getsampwidth()
        rate = wf.getframerate()
        if sampw != 2:
            raise RuntimeError(f"Only 16-bit PCM WAV files are supported (got {sampw * 8}-bit).")
        if nchan != 1:
            raise RuntimeError(f"Only mono WAV files are supported (got {nchan} channels).")
        if expected_rate is not None and rate != expected_rate:
            print(f"Warning: WAV file sample rate ({rate} Hz) != --rate ({expected_rate} Hz). Using WAV file rate.", file=sys.stderr)
        block_size = 4096
        while True:
            raw = wf.readframes(block_size)
            if not raw:
                break
            data = np.frombuffer(raw, dtype='<i2')  # little-endian 16-bit
            if nchan == 2:
                data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
            yield data, rate

def i16_to_u12_stream(x_i16: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """
    Convert 16-bit signed samples to 12-bit unsigned DAC codes (mid-scale centered).

    The int16 range [-32768, 32767] is mapped to [0, 4095] with mid-scale bias:
    0 in audio -> ~2048 code. A headroom < 1.0 prevents hitting 0 or 4095 to avoid DAC glitches.
    """
    x = x_i16.astype(np.int32)
    x = x / 32768.0  # Normalize to [-1.0, +1.0)
    amp = 0.95 * (4095.0 / 2.0)  # headroom factor times half-range (2047.5)
    codes = np.round(2048.0 + amp * x).astype(np.int32)
    np.clip(codes, 0, 4095, out=codes)
    return codes.astype(np.uint16)

def generate_sine_wave(frequency: float, duration: float, sample_rate: int) -> tuple[np.ndarray, int]:
    """
    Generate a sine wave as int16 samples.
    Returns (array of samples, sample_rate).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.9 * 32767 * np.sin(2 * np.pi * frequency * t)  # 90% of full scale
    return waveform.astype(np.int16), sample_rate


def play_wav_to_dac(
    wav_path: str,
    sample_rate: int = 16000,
    spi_bus: int = 0,
    spi_ce: int = 1,
    spi_mode: int = 0,
    spi_speed_hz: int = 1_000_000,
    gain_x2: bool = False,
    chunk_samp: int = 1024,
    sine_test: tuple[float, float] = None,  # (frequency, duration)
):
    """
    Stream a WAV file to the MCP4822 DAC via SPI.

    This opens the SPI device and plays the WAV audio in real time through channel A of the MCP4822.
    If gain_x2 is True, the DAC's 2x gain mode is used (doubling the output range).
    Channel B is put into shutdown to avoid any interference.
    """
    spi = open_dac_spi(spi_bus, spi_ce, spi_mode, spi_speed_hz)
    # Helper to pack a 12-bit DAC value into two bytes (with config bits)
    def pack_word(u12: int, shdn: bool = True, ch: str = 'A'):
        """
        Pack a 12-bit value and control flags into a 16-bit frame for MCP4822.
        Bits: [15]=channel (0=A,1=B), [14]=don't care (0), [13]=GA (gain: 1=1x, 0=2x), [12]=SHDN (1=on, 0=shutdown), [11:0]=value.
        """
        u12 = u12 & 0x0FFF
        ab = 0 if ch.upper() == 'A' else 1
        ga = 0 if gain_x2 else 1  # GA=0 for 2x, 1 for 1x
        sd = 1 if shdn else 0     # SHDN=1 output active, 0 shutdown
        word = (ab << 15) | (0 << 14) | (ga << 13) | (sd << 12) | u12
        return (word >> 8) & 0xFF, word & 0xFF

    # Shut down channel B to prevent noise (tri-state output)
    hi, lo = pack_word(0, shdn=False, ch='B')
    spi.xfer2([hi, lo])

    print(f"Playing '{wav_path}' via MCP4822 on /dev/spidev{spi_bus}.{spi_ce} @ {spi_speed_hz/1e6:.1f} MHz (mode {spi_mode})")
    print(f"Sample rate: {sample_rate} Hz; Gain: {'2x' if gain_x2 else '1x'}; Output channel: A; Chunk size: {chunk_samp} samples")

    t_prev = time.perf_counter()
    try:
        if sine_test is not None:
            block_i16, fs = generate_sine_wave(frequency=sine_test[0], duration=sine_test[1], sample_rate=sample_rate)
            data_blocks = [(block_i16, fs)]
        else:
            data_blocks = read_wav_frames(wav_path, expected_rate=sample_rate)

        for block_i16, fs in data_blocks:
            #try:
            # for block_i16, fs in read_wav_frames(wav_path, expected_rate=sample_rate):
            codes = i16_to_u12_stream(block_i16, headroom=0.95)
            n = len(codes)
            i = 0
            while i < n:
                j = min(i + chunk_samp, n)
                segment = codes[i:j]
                tx_buffer = []
                for code in segment:
                    hi, lo = pack_word(code, shdn=True, ch='A')
                    tx_buffer.append(int(hi))
                    tx_buffer.append(int(lo))
                #spi.xfer2(tx_buffer)
                for k in range(0, len(tx_buffer), 2):
                    spi.xfer2([tx_buffer[k], tx_buffer[k+1]])                
                seg_dur = (j - i) / float(fs)
                t_now = time.perf_counter()
                elapsed = t_now - t_prev
                sleep_time = seg_dur - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    t_prev = time.perf_counter()
                else:
                    t_prev = t_now
                i = j
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return
    finally:
        try:
            spi.close()
        except Exception:
            pass
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="SPI WAV player for MCP4822 DAC")
    parser.add_argument("-w", "--wav", default="adc_capture.wav", help="Input WAV file path [default: adc_capture.wav]")
    parser.add_argument("-r", "--rate", type=int, default=16000, help="Sample rate (Hz) [default: 16000]")
    parser.add_argument("--bus", type=int, default=0, help="SPI bus [default: 0]")
    parser.add_argument("--ce", type=int, default=1, help="SPI chip select [default: 1 (CE1)]")
    parser.add_argument("--mode", type=int, default=0, help="SPI mode (0-3) [default: 0]")
    parser.add_argument("--speed", type=int, default=1000000, help="SPI speed (Hz) [default: 1000000]")
    parser.add_argument("--gain-x2", action="store_true", help="Enable 2x gain mode (output ~0-4.096 V range)")
    parser.add_argument("--chunk-samp", type=int, default=1024, help="Samples per SPI transfer [default: 1024]")
    parser.add_argument("--sine-test", nargs=2, type=float, metavar=("FREQ_HZ", "DURATION_SEC"),
                    help="Play a generated sine wave instead of a WAV file.")
    args = parser.parse_args()
    sine_params = tuple(args.sine_test) if args.sine_test else None

    try:
        play_wav_to_dac(
            wav_path=args.wav,
            sample_rate=args.rate,
            spi_bus=args.bus,
            spi_ce=args.ce,
            spi_mode=args.mode,
            spi_speed_hz=args.speed,
            gain_x2=args.gain_x2,
            chunk_samp=args.chunk_samp,
            sine_test=sine_params,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
