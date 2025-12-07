#!/usr/bin/env python3
"""
adc_record_mcp3201_cs_toggle.py — SPI recorder for MCP3201-BI/P with per-sample CS toggle

Device: Microchip MCP3201 (single-channel, 12-bit)
SPI:    Mode 0 (CPOL=0, CPHA=0), 16 clocks (2 bytes) per conversion
Output: 16-bit PCM WAV

Power & level shifting:
  - MCP3201 VDD and VREF at 5.0 V (external supply).
  - Pi <-> MCP3201 signals through level shifter:
      Pi SCLK  -> level shifter -> MCP3201 CLK
      Pi CE0   -> level shifter -> MCP3201 CS/!SHDN
      Pi MISO  <- level shifter <- MCP3201 DOUT (5 V -> 3.3 V)
  - Common ground between Pi and MCP3201.

Notes:
  - MCP3201 outputs 1 null bit then 12 data bits MSB-first while CS is LOW.
  - We want exactly one conversion per CS assertion:
        CS low + 16 clocks  → read 12 bits → CS high (re-arm).
  - This script uses spidev.xfer3 (if present) to toggle CS between 2-byte segments.
    If xfer3 is unavailable, it falls back to one xfer2() call per sample.
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

from typing import Union


def parse_mcp3201_frame(rx_bytes: Union[bytes, bytearray, list], n_samples: int, mid_code: int = 2048) -> np.ndarray:
    """
    Convert 2*n_samples bytes from MCP3201 into int16 audio.
      For each sample: raw12 = ((b0 & 0x1F) << 7) | (b1 >> 1)
                       s12   = raw12 - mid_code
                       out16 = s12 << 4
    """
    out = np.empty(n_samples, dtype=np.int16)
    # Ensure indexable ints
    if isinstance(rx_bytes, (bytes, bytearray)):
        b = rx_bytes
    else:
        b = bytearray(rx_bytes)

    for i in range(n_samples):
        b0 = b[2 * i]
        b1 = b[2 * i + 1]
        raw12 = ((b0 & 0x1F) << 7) | (b1 >> 1)
        s12 = raw12 - mid_code
        out[i] = np.int16(s12 << 4)
    return out


def record_adc_wav(
    outfile: str,
    duration_sec: float = 5.0,
    sample_rate: int = 16000,
    frame_ms: int = 20,
    spi_bus: int = 0,
    spi_ce: int = 0,
    spi_mode: int = 0,
    spi_speed_hz: int = 1_000_000,
    dc_block: bool = False,
    mid_code: int = 2048,
    delay_us: int = 0,  # small pause after each transfer when using xfer2 fallback
):
    # Derived sizes
    frame_len = int(sample_rate * frame_ms // 1000)
    total_samples_target = int(round(duration_sec * sample_rate))

    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_ce)
    spi.max_speed_hz = spi_speed_hz
    spi.mode = spi_mode
    spi.bits_per_word = 8

    have_xfer3 = hasattr(spi, "xfer3")

    print(
        f"MCP3201 record: {duration_sec:.2f}s @ {sample_rate} Hz, frame={frame_len} samples "
        f"(~{frame_ms} ms), SPI {spi_bus}.{spi_ce} {spi_speed_hz/1e6:.2f} MHz, mode={spi_mode}, "
        f"xfer3={'yes' if have_xfer3 else 'no'}"
    )
    print("Press Ctrl+C to stop early…")

    captured_chunks = []
    captured = 0

    t0 = time.perf_counter()

    #next_deadline = t0
    #frame_period = frame_len / sample_rate
    # --- per-sample pacing for xfer2 fallback ---
    sample_period_us = 1_000_000.0 / sample_rate          # e.g., 62.5 µs @ 16 kHz
    transfer_time_us = (16.0 / spi_speed_hz) * 1_000_000  # 16 SPI clocks per sample
    overhead_us = 5.0                                      # small cushion for Python/kernel
    target_delay_us = int(max(0.0, round(sample_period_us - transfer_time_us - overhead_us)))
    per_call_delay = int(delay_us) if (delay_us and delay_us > 0) else target_delay_us

    try:
        while captured < total_samples_target:
            # Try xfer3 first (per-sample CS toggle), fall through to xfer2 on any error.
            rx = None
            if have_xfer3:
                try:
                    # xfer3 expects a list of bytes-like segments; no keyword args on your build.
                    segments = [[0x00, 0x00] for _ in range(frame_len)]
                    res = spi.xfer3(segments)  # CS toggles between segments

                    # Normalize to flat bytearray of length 2*frame_len
                    flat = bytearray()
                    if isinstance(res, (bytes, bytearray)):
                        flat.extend(res)
                    elif isinstance(res, (list, tuple)):
                        for item in res:
                            if isinstance(item, (bytes, bytearray)):
                                flat.extend(item)
                            elif isinstance(item, (list, tuple)):
                                for x in item:
                                    flat.append(int(x) & 0xFF)
                            else:
                                flat.append(int(item) & 0xFF)
                    else:
                        for x in res:
                            flat.append(int(x) & 0xFF)

                    if len(flat) != 2 * frame_len:
                        raise RuntimeError(f"xfer3 returned {len(flat)} bytes, expected {2*frame_len}")

                    rx = flat  # success

                except Exception as e:
                    print(f"[warn] xfer3 failed or incompatible ({e}); using per-sample xfer2 fallback.")
                    have_xfer3 = False  # permanently switch to fallback

            # Fallback (or primary if xfer3 unavailable): one xfer2 per sample → CS blips per call
            if rx is None:
                rx = bytearray(2 * frame_len)
                for i in range(frame_len):
                    pair = spi.xfer2([0x00, 0x00], 0, per_call_delay, 8)
                    rx[2*i]   = pair[0]
                    rx[2*i+1] = pair[1]

            # Parse exactly frame_len samples, then pace
            frame = parse_mcp3201_frame(rx, frame_len, mid_code=mid_code)
            captured_chunks.append(frame)
            captured += frame_len

            #next_deadline += frame_period
            #rem = next_deadline - time.perf_counter()
            #if rem > 0:
            #    time.sleep(rem)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            spi.close()
        except Exception:
            pass

    # Concatenate and trim
    audio = np.concatenate(captured_chunks) if captured_chunks else np.zeros(0, dtype=np.int16)
    if len(audio) > total_samples_target:
        audio = audio[:total_samples_target]

    # Optional DC blocker
    if dc_block and len(audio) > 0:
        a = 0.995  # ~12.7 Hz cutoff at 16 kHz
        y = np.empty_like(audio, dtype=np.int16)
        px = np.int32(0)
        py = np.int32(0)
        for i, x in enumerate(audio.astype(np.int32)):
            yn = x - px + (a * py)
            if yn > 32767: yn = 32767
            elif yn < -32768: yn = -32768
            y[i] = np.int16(yn)
            px = x
            py = yn
        audio = y

    # Write WAV
    with wave.open(outfile, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    elapsed = time.perf_counter() - t0
    eff_sr = (len(audio) / elapsed) if elapsed > 0 else 0.0
    print(f"Wrote {len(audio)} samples to {outfile}")
    print(f"Elapsed: {elapsed:.3f}s  (~{len(audio)/sample_rate:.3f}s audio captured)")
    print(f"Effective capture throughput: {eff_sr:.1f} samples/s")


def main():
    p = argparse.ArgumentParser(description="MCP3201 SPI recorder with per-sample CS toggle → WAV")
    p.add_argument("-o", "--out", default="adc_capture.wav", help="Output WAV file path")
    p.add_argument("-d", "--duration", type=float, default=5.0, help="Duration to capture (seconds)")
    p.add_argument("-r", "--rate", type=int, default=16000, help="Sample rate (Hz)")
    p.add_argument("-f", "--frame-ms", type=int, default=20, help="Frame size (ms)")
    p.add_argument("--bus", type=int, default=0, help="SPI bus (default 0)")
    p.add_argument("--ce", type=int, default=0, help="SPI chip select (default CE0)")
    p.add_argument("--mode", type=int, default=0, help="SPI mode (0..3), default 0 for MCP3201")
    p.add_argument("--speed", type=int, default=1_000_000, help="SPI speed (Hz), default 1 MHz")
    p.add_argument("--dc-block", action="store_true", help="Apply lightweight DC blocker")
    p.add_argument("--mid", type=int, default=2048, help="Mid-code to subtract (default 2048)")
    p.add_argument("--delay-us", type=int, default=0, help="Per-sample delay (µs) for xfer2 fallback")
    args = p.parse_args()

    try:
        record_adc_wav(
            outfile=args.out,
            duration_sec=args.duration,
            sample_rate=args.rate,
            frame_ms=args.frame_ms,
            spi_bus=args.bus,
            spi_ce=args.ce,
            spi_mode=args.mode,
            spi_speed_hz=args.speed,
            dc_block=args.dc_block,
            mid_code=args.mid,
            delay_us=args.delay_us,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
