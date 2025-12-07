#!/usr/bin/env python3
"""
adc_record_test.py — Minimal SPI recorder for MCP3201-BI/P on Raspberry Pi

Device: Microchip MCP3201 (single-channel, 12-bit)
SPI:    Mode 0 (CPOL=0, CPHA=0), 2 bytes (16 clocks) per conversion
Default: 16 kHz capture → 16-bit PCM WAV

Wiring (typical):
  Pi 3.3V  -> MCP3201 VDD, VREF (per your analog design)
  Pi GND   -> MCP3201 VSS (GND)
  Pi SCLK  -> MCP3201 CLK
  Pi CE0   -> MCP3201 CS/!SHDN
  Pi MISO  -> MCP3201 DOUT
  (MOSI is NOT used by MCP3201)

Notes:
- The MCP3201 outputs a 13-bit word each conversion: 1 null bit then 12 data bits (MSB first).
- We issue 16 clocks by transferring two dummy bytes (0x00, 0x00) and parse the 12 data bits:
      raw12 = ((b0 & 0x1F) << 7) | (b1 >> 1)
- If your front-end biases the input at mid-scale (Vref/2), subtract 2048 to recentre to 0.
- Keep SCLK continuous while CS is low for a valid conversion.
"""

import argparse
import time
import sys
import numpy as np
import wave
from scipy.signal import cheby2, cheb2ord, sosfilt, sosfiltfilt


try:
    import spidev
except ImportError:
    spidev = None  # Fail later, at use time, instead of at import


def parse_mcp3201_frame(rx: list[int], n_samples: int) -> np.ndarray:
    """
    Convert MCP3201 burst response (2 bytes per sample) → int16 audio samples.

    For each sample (rx index = 2*i):
      b0, b1 = rx[2*i], rx[2*i+1]
      raw12  = ((b0 & 0x1F) << 7) | (b1 >> 1)   # 12-bit unsigned (0..4095)
      s12    = raw12 - 2048                     # signed w/ mid-scale as 0
      out16  = s12 << 4                         # scale to 16-bit lane
    """
    out = np.empty(n_samples, dtype=np.int16)
    for i in range(n_samples):
        b0 = rx[2 * i]
        b1 = rx[2 * i + 1]
        raw12 = ((b0 & 0x1F) << 7) | (b1 >> 1)
        MID_CODE = 2229
        s12 = raw12 - MID_CODE
        scaled = s12 * 8   # or << 3

        # Optional clipping (defensive)
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768

        out[i] = np.int16(scaled)
    return out

def apply_speech_band_filters(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply your tuned speech-band filters to a full recording:

      - High-pass ~50 Hz (remove very low-frequency rumble)
      - Low-pass  ~4 kHz (remove high-frequency hiss)

    audio: np.int16 array
    returns: np.int16 array (filtered)
    """

    if audio is None or audio.size == 0:
        return audio

    # Work in float for filtering
    x = audio.astype(np.float64)
    nyq = sample_rate / 2.0

    # ---- High-pass: ~50 Hz ----
    # Based on your MATLAB tuning: Fp_hp ≈ 50–60 Hz
    hp_cut = 50.0 / nyq          # normalized cutoff
    hp_order = 4                 # moderate order
    hp_rs = 60.0                 # stopband attenuation (dB)

    sos_hp = cheby2(
        hp_order,
        hp_rs,
        hp_cut,
        btype="highpass",
        output="sos",
    )
    # Zero-phase filtering
    x = sosfiltfilt(sos_hp, x)

    # ---- Low-pass: match MATLAB normalized design (Fp=4k, Fs=5k at 16k) ----
    # Reference (MATLAB) design:
    #   Fs_ref = 16000, Fn_ref = 8000
    #   Fp_ref = 4000, Fs_ref = 5000
    # ⇒ Wp_norm = 0.5, Ws_norm = 0.625
    Wp_norm = 4000.0 / 8000.0   # 0.5
    Ws_norm = 5000.0 / 8000.0   # 0.625

    Rp = 1.0    # passband ripple (dB)  – same as MATLAB
    Rs = 60.0   # stopband atten (dB)   – same as MATLAB

    # Design the digital low-pass with the *normalized* edges
    n_lp, Wn_lp = cheb2ord(Wp_norm, Ws_norm, Rp, Rs)
    sos_lp = cheby2(
        n_lp,
        Rs,
        Wn_lp,
        btype="lowpass",
        output="sos",
    )

    x = sosfiltfilt(sos_lp, x)


    # Convert back to int16 safely
    x = np.clip(x, -32768, 32767)
    return x.astype(np.int16)

class AdcSource:
    """
    Streaming MCP3201 source that mimics your *Source classes* and
    adds a speech-band filter:

        - high-pass:  ~60 Hz (stopband ~50 Hz)
        - low-pass:   ~4 kHz (stopband ~5 kHz)

    Usage (what deploy_whispercpp_main.py will do for INPUT_MODE="adc"):

        with AdcSource(sample_rate=44100, frame_ms=20, ...) as src:
            for frame_bytes in src.frames():
                ...  # each frame_bytes is 20 ms of mono int16 PCM
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        spi_bus: int = 0,
        spi_ce: int = 0,
        spi_mode: int = 0,             # MCP3201: mode 0
        spi_speed_hz: int = 1_000_000, # default matches record_adc_wav()
        dc_block: bool = False,
        use_bp_filter: bool = True,    # enable band-pass by default
    ):
        if spidev is None:
            raise RuntimeError(
                "AdcSource requires spidev. Install it on the Pi: "
                "sudo apt-get install -y python3-spidev"
            )

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_len = int(sample_rate * frame_ms // 1000)  # samples per frame

        self.spi_bus = spi_bus
        self.spi_ce = spi_ce
        self.spi_mode = spi_mode
        self.spi_speed_hz = spi_speed_hz

        self.dc_block = dc_block
        self.use_bp_filter = use_bp_filter

        self._spi = None

        # DC-blocker state carried across frames
        self._dc_prev_x = 0
        self._dc_prev_y = 0

        # Band-pass Chebyshev II filter state (HP then LP)
        self._sos_hp = None
        self._sos_lp = None
        self._zi_hp = None
        self._zi_lp = None

    # ---------------- Context manager ----------------

    def __enter__(self):
        spi = spidev.SpiDev()
        spi.open(self.spi_bus, self.spi_ce)
        spi.max_speed_hz = self.spi_speed_hz
        spi.mode = self.spi_mode
        spi.bits_per_word = 8
        self._spi = spi

        if self.use_bp_filter:
            self._design_bandpass_filter()

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
            self._spi = None

    # ---------------- Filter design ----------------

    def _design_bandpass_filter(self):
        """
        Design a Chebyshev Type II band-pass:

            - High-pass corner:  passband 60 Hz, stopband 50 Hz
            - Low-pass corner:   passband 4 kHz, stopband 5 kHz

        using your tuned values:
            Fp_hp = 50,  Fs_hp = 60
            Fp_lp = 4000, Fs_lp = 5000
        """
        fs = float(self.sample_rate)
        Fn = fs / 2.0

        # High-pass settings
        Fp_hp = 60.0  # passband start
        Fs_hp = 50.0  # stopband end
        Wp_hp = Fp_hp / Fn
        Ws_hp = Fs_hp / Fn

        # Low-pass settings
        
        Wp_lp = 0.5
        Ws_lp = 0.625
        Rp = 1.0   # passband ripple (dB)
        Rs = 60.0  # stopband attenuation (dB)

        # High-pass Chebyshev II
        n_hp, Wc_hp = cheb2ord(Wp_hp, Ws_hp, Rp, Rs)
        sos_hp = cheby2(n_hp, Rs, Wc_hp, btype="highpass", output="sos")

        # Low-pass Chebyshev II
        n_lp, Wc_lp = cheb2ord(Wp_lp, Ws_lp, Rp, Rs)
        sos_lp = cheby2(n_lp, Rs, Wc_lp, btype="lowpass", output="sos")

        self._sos_hp = sos_hp
        self._sos_lp = sos_lp

        # Initialize filter state (zi) for streaming
        self._zi_hp = np.zeros((sos_hp.shape[0], 2), dtype=np.float32)
        self._zi_lp = np.zeros((sos_lp.shape[0], 2), dtype=np.float32)

    def _apply_bandpass(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply HP then LP Chebyshev filters with state carried across frames.
        """
        if self._sos_hp is None or self._sos_lp is None:
            return frame

        x = frame.astype(np.float32)

        # High-pass first
        x, self._zi_hp = sosfilt(self._sos_hp, x, zi=self._zi_hp)

        # Then low-pass
        x, self._zi_lp = sosfilt(self._sos_lp, x, zi=self._zi_lp)

        # Clip back to int16
        x = np.clip(x, -32768.0, 32767.0)
        return x.astype(np.int16)

    # ---------------- ADC frame reading ----------------

    def _read_frame_samples(self) -> np.ndarray:
        """
        Perform one MCP3201 frame capture:
          - 2 dummy bytes per sample → 16 SCLKs
          - Parse via parse_mcp3201_frame()
          - Return np.int16[frame_len]
        """
        if self._spi is None:
            raise RuntimeError("AdcSource SPI not opened. Use 'with AdcSource(...) as src:'.")

        frame_len = self.frame_len
        spi = self._spi

        # Same xfer2/xfer3 logic as record_adc_wav(), just for a single frame
        if hasattr(spi, "xfer3"):
            segments = [[0x00, 0x00] for _ in range(frame_len)]
            rx_flat = []
            for seg in segments:
                r = spi.xfer3(seg)  # CS low for 2 bytes, then high
                rx_flat.extend(r)
            frame = parse_mcp3201_frame(rx_flat, frame_len)
        else:
            rx_bytes = bytearray(2 * frame_len)
            for i in range(frame_len):
                pair = spi.xfer2([0x00, 0x00])  # CS low for 16 clocks, then high
                rx_bytes[2 * i] = pair[0]
                rx_bytes[2 * i + 1] = pair[1]
            frame = parse_mcp3201_frame(list(rx_bytes), frame_len)

        # Optional DC blocker
        if self.dc_block:
            frame = self._dc_block_frame(frame)

        # Optional speech band-pass filter
        if self.use_bp_filter:
            frame = self._apply_bandpass(frame)

        return frame

    def _dc_block_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Single-pole high-pass DC blocker applied per frame but with
        state carried across frames:

            y[n] = x[n] - x[n-1] + a * y[n-1],  a ≈ 0.995
        """
        if frame.size == 0:
            return frame

        a = 0.995
        y = np.empty_like(frame, dtype=np.int16)

        prev_x = int(self._dc_prev_x)
        prev_y = int(self._dc_prev_y)

        for i, x in enumerate(frame.astype(np.int32)):
            yn = x - prev_x + int(a * prev_y)
            if yn > 32767:
                yn = 32767
            elif yn < -32768:
                yn = -32768
            y[i] = np.int16(yn)
            prev_x = int(x)
            prev_y = int(yn)

        # Update persistent state
        self._dc_prev_x = prev_x
        self._dc_prev_y = prev_y

        return y

    # ---------------- Public streaming API ----------------

    def frames(self):
        """
        Infinite generator of 20 ms frames as bytes.

        Each yielded object is 16-bit mono PCM bytes:
            len(frame_bytes) == frame_len * 2
        which matches LiteDiarizer.process_frame() expectations.
        """
        frame_period = self.frame_len / float(self.sample_rate)
        next_deadline = time.perf_counter()

        while True:
            frame = self._read_frame_samples()   # np.int16[frame_len]
            yield frame.tobytes()                # → bytes

            next_deadline += frame_period
            sleep = next_deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)


def record_adc_wav(
    outfile: str,
    duration_sec: float = 60.0,
    sample_rate: int = 16000,
    frame_ms: int = 20,
    spi_bus: int = 0,
    spi_ce: int = 0,
    spi_mode: int = 0,           # MCP3201: mode 0
    spi_speed_hz: int = 1_000_000,  # Start at 1 MHz; raise carefully if stable
    dc_block: bool = False,
):
    # MCP3201: exactly 2 bytes per sample
    frame_len = int(sample_rate * frame_ms // 1000)
    total_samples_target = int(round(duration_sec * sample_rate))

    spi = spidev.SpiDev()
    spi.open(spi_bus, spi_ce)
    spi.max_speed_hz = spi_speed_hz
    spi.mode = spi_mode
    spi.bits_per_word = 8
    # Most RPi setups default MSB-first, which matches MCP3201.

    captured_chunks = []
    captured = 0

    print(
        f"MCP3201 record: {duration_sec:.2f}s @ {sample_rate} Hz, frame={frame_len} samples "
        f"(~{frame_ms} ms), SPI {spi_bus}.{spi_ce} speed={spi_speed_hz/1e6:.1f} MHz, mode={spi_mode}"
    )
    print("Press Ctrl+C to stop early…")

    t0 = time.perf_counter()
    next_deadline = t0
    frame_period = frame_len / sample_rate

    try:
        while captured < total_samples_target:
                        # Build one burst: 2 bytes per sample, all zeros (dummy) to clock data out.
            # OLD:
            # tx = [0x00, 0x00] * frame_len
            # rx = spi.xfer2(tx)
            # frame = parse_mcp3201_frame(rx, frame_len)

            # NEW:
            #if hasattr(spi, "xfer3"):
            #    segments = [[0x00, 0x00] for _ in range(frame_len)]
            #    rx_flat = []
            #    for seg in segments:
            #        rx_flat.extend(spi.xfer3(seg))  # CS toggles for each segment
            #    frame = parse_mcp3201_frame(rx_flat, frame_len)
            if hasattr(spi, "xfer3"):
                segments = [[0x00, 0x00] for _ in range(frame_len)]
                rx_flat = []
                for i, seg in enumerate(segments):
                    r = spi.xfer3(seg)  # CS low for these 2 bytes, then high
                    if captured == 0 and i < 200:  # dump first 8 samples of the first frame
                        print(f"raw bytes[{i}]: b0=0x{r[0]:02X} b1=0x{r[1]:02X}")
                    rx_flat.extend(r)
                frame = parse_mcp3201_frame(rx_flat, frame_len)
            else:
                rx_bytes = bytearray(2 * frame_len)
                for i in range(frame_len):
                    pair = spi.xfer2([0x00, 0x00])   # CS low for 16 clocks, then high
                    if captured == 0 and i < 8:   # dump the first 8 samples (16 bytes)
                        print(f"raw bytes: b0=0x{pair[0]:02X} b1=0x{pair[1]:02X}")
                    rx_bytes[2*i]   = pair[0]
                    rx_bytes[2*i+1] = pair[1]
                frame = parse_mcp3201_frame(list(rx_bytes), frame_len)

            captured_chunks.append(frame)
            captured += frame_len

            # Best-effort pacing to target sample_rate
            next_deadline += frame_period
            sleep = next_deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        try:
            spi.close()
        except Exception:
            pass

    # Concatenate, trim to exact target length
    audio = np.concatenate(captured_chunks) if captured_chunks else np.zeros(0, dtype=np.int16)
    if len(audio) > total_samples_target:
        audio = audio[:total_samples_target]

    # Optional DC blocker (single-pole HPF)
    if dc_block and len(audio) > 0:
        # y[n] = x[n] - x[n-1] + a*y[n-1]; a≈0.995 → ~12.7 Hz corner @ 16 kHz
        a = 0.995
        y = np.empty_like(audio, dtype=np.int16)
        prev_x = np.int32(0)
        prev_y = np.int32(0)
        for i, x in enumerate(audio.astype(np.int32)):
            yn = x - prev_x + (a * prev_y)
            if yn > 32767: yn = 32767
            elif yn < -32768: yn = -32768
            y[i] = np.int16(yn)
            prev_x = x
            prev_y = yn
        audio = y

    
    # NEW: apply your 50 Hz high-pass and 4 kHz low-pass
    if len(audio) > 0:
        audio = apply_speech_band_filters(audio, sample_rate)

    # Write WAV
    with wave.open(outfile, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())

    t1 = time.perf_counter()
    elapsed = t1 - t0
    eff_sr = (len(audio) / elapsed) if elapsed > 0 else 0.0
    print(f"Wrote {len(audio)} samples to {outfile}")
    print(f"Elapsed: {elapsed:.3f}s  (~{len(audio)/sample_rate:.3f}s audio captured)")
    print(f"Effective capture throughput: {eff_sr:.1f} samples/s")


def main():
    p = argparse.ArgumentParser(description="SPI recorder for MCP3201 → WAV")
    p.add_argument("-o", "--out", default="adc_capture.wav", help="Output WAV file path")
    p.add_argument("-d", "--duration", type=float, default=60.0, help="Duration to capture (seconds)")
    p.add_argument("-r", "--rate", type=int, default=16000, help="Sample rate (Hz)")
    p.add_argument("-f", "--frame-ms", type=int, default=20, help="Frame size (ms)")
    p.add_argument("--bus", type=int, default=0, help="SPI bus (default 0)")
    p.add_argument("--ce", type=int, default=0, help="SPI chip select (default CE0)")
    p.add_argument("--mode", type=int, default=0, help="SPI mode (0..3), default 0 for MCP3201")
    p.add_argument("--speed", type=int, default=1_000_000, help="SPI speed (Hz), default 1 MHz")
    p.add_argument("--dc-block", action="store_true", help="Apply a lightweight DC blocker")
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
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
