# spi_audio.py
import time
import threading
import queue
import numpy as np

try:
    import spidev
except ImportError:
    raise RuntimeError("Install spidev on the Pi: sudo apt-get install -y python3-spidev")

class SpiAudioIO:
    """
    SPI-based audio I/O for Raspberry Pi using:
      - SPI0 CE0: ADC (microphone)
      - SPI0 CE1: DAC (speaker)

    Designed for 16 kHz mono streaming with small frames (e.g., 20 ms).
    """
    def __init__(
        self,
        sample_rate=16000,
        frame_ms=20,
        adc_bits=12,
        dac_bits=12,
        spi_bus=0,
        adc_ce=0,
        dac_ce=1,
        spi_mode=0,
        spi_speed_hz=1_000_000,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_len = int(self.sample_rate * frame_ms // 1000)  # samples/frame
        self.adc_bits = int(adc_bits)
        self.dac_bits = int(dac_bits)
        self.spi_bus = spi_bus
        self.adc_ce = adc_ce
        self.dac_ce = dac_ce
        self.spi_mode = spi_mode
        self.spi_speed_hz = spi_speed_hz

        # Queues for inter-thread audio flow
        self.capture_q = queue.Queue(maxsize=8)  # mic frames (int16)
        self.playback_q = queue.Queue(maxsize=8) # speaker frames (int16)

        # SPI devices
        self.spi_adc = spidev.SpiDev()
        self.spi_dac = spidev.SpiDev()

        self._stop = threading.Event()
        self._t_cap = None
        self._t_dac = None

    # ---------- Low-level helpers (adjust per your specific ADC/DAC) ----------
   
    def _adc_read_frame(self):
        """
        Read a frame from an MCP3201 (12-bit, single-channel) over SPI.

        MCP3201 protocol summary:
        - Pull CS low, clock 1 "null" bit, then 12 data bits MSB→LSB.
        - Data shifts out on CLK falling edges.
        - CS must go HIGH between conversions, so we do one SPI
            transfer per sample (2 bytes) to guarantee a CS pulse.

        spidev details:
        - xfer2([...]) toggles CS low at start and high at end of the call.
        - We send two dummy bytes per sample to clock out 16 bits.
        - After two bytes:
            hi = [??, ??, NULL, B11, B10, B9, B8, B7]
            lo = [ B6,  B5,  B4,  B3,  B2,  B1,  B0,  B1(repeat) ]
            So the 12-bit value is:
            raw12 = ((hi & 0x1F) << 7) | ((lo & 0xFE) >> 1)
        """
        n = self.frame_len
        out = np.empty(n, dtype=np.int16)

        # two bytes per sample; CS is pulsed each call guaranteeing a new conversion
        for i in range(n):
            hi, lo = self.spi_adc.xfer2([0x00, 0x00])
            raw12 = ((hi & 0x1F) << 7) | ((lo & 0xFE) >> 1)  # 0..4095

            # Convert unsigned 12-bit to signed 16-bit centered at 0.
            # (Assumes your mic signal is biased around mid-scale.)
            s12 = raw12 - 2048                 # −2048..+2047
            out[i] = np.int16(s12 << 4)        # scale to int16
        return out


    def _dac_write_frame(self, frame_i16):
        """
        Write a frame to the DAC over SPI.
        Example for MCP4921 (12-bit). Command: 4 config bits + 12 data bits.
        """
        # Clip and convert int16 -> 12-bit unsigned
        x = np.asarray(frame_i16, dtype=np.int16)
        # Shift down to 12-bit signed then to unsigned: [-2048..2047] -> [0..4095]
        s12 = (x.astype(np.int32) >> 4)  # [-2048..2047]
        u12 = (s12 + 2048).clip(0, 4095).astype(np.uint16)

        # MCP4921: 16-bit word: [C3..C0 D11..D0]; config: 0x3000 = (0b0011 << 12) => Buf=1, Gain=1x, Shutdown=Active
        tx = []
        for v in u12:
            word = 0x3000 | (v & 0x0FFF)
            tx.append((word >> 8) & 0xFF)
            tx.append(word & 0xFF)
        if tx:
            self.spi_dac.xfer2(tx)

    # ---------- Threads ----------
    def _capture_loop(self):
        next_deadline = time.perf_counter()
        frame_period = self.frame_len / self.sample_rate
        while not self._stop.is_set():
            try:
                frame = self._adc_read_frame()
                # Non-blocking put with drop-old behavior if full
                try:
                    self.capture_q.put_nowait(frame)
                except queue.Full:
                    _ = self.capture_q.get_nowait()
                    self.capture_q.put_nowait(frame)
            except Exception as e:
                # If something goes wrong, don’t crash the process—emit silence and continue
                self._safe_silence()
            # pace loop (best-effort)
            next_deadline += frame_period
            sleep = next_deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    def _playback_loop(self):
        next_deadline = time.perf_counter()
        frame_period = self.frame_len / self.sample_rate
        silence = np.zeros(self.frame_len, dtype=np.int16)
        while not self._stop.is_set():
            try:
                frame = self.playback_q.get_nowait()
            except queue.Empty:
                frame = silence
            try:
                self._dac_write_frame(frame)
            except Exception:
                pass
            next_deadline += frame_period
            sleep = next_deadline - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    def _safe_silence(self):
        # push a silence frame on capture to keep upstream happy
        try:
            self.capture_q.put_nowait(np.zeros(self.frame_len, dtype=np.int16))
        except queue.Full:
            pass

    # ---------- Public API ----------
    def start(self):
        # Open SPI devices
        self.spi_adc.open(self.spi_bus, self.adc_ce)
        self.spi_adc.max_speed_hz = self.spi_speed_hz
        self.spi_adc.mode = self.spi_mode
        self.spi_adc.bits_per_word = 8  # MCP parts use 8-bit transfers

        self.spi_dac.open(self.spi_bus, self.dac_ce)
        self.spi_dac.max_speed_hz = self.spi_speed_hz
        self.spi_dac.mode = self.spi_mode
        self.spi_dac.bits_per_word = 8

        self._stop.clear()
        self._t_cap = threading.Thread(target=self._capture_loop, name="spi-capture", daemon=True)
        self._t_dac = threading.Thread(target=self._playback_loop, name="spi-playback", daemon=True)
        self._t_cap.start()
        self._t_dac.start()

    def stop(self):
        self._stop.set()
        if self._t_cap: self._t_cap.join(timeout=1.0)
        if self._t_dac: self._t_dac.join(timeout=1.0)
        try:
            self.spi_adc.close()
        except Exception:
            pass
        try:
            self.spi_dac.close()
        except Exception:
            pass

    def read_frame(self, timeout=0.1):
        """Blocking-ish read of one int16 frame (len=frame_len)."""
        return self.capture_q.get(timeout=timeout)

    def write_frame(self, frame_i16):
        """Queue one int16 frame for playback."""
        try:
            self.playback_q.put_nowait(np.asarray(frame_i16, dtype=np.int16))
        except queue.Full:
            # drop oldest to keep latency bounded
            _ = self.playback_q.get_nowait()
            self.playback_q.put_nowait(np.asarray(frame_i16, dtype=np.int16))

    def frames(self):
        """Generator yielding successive capture frames."""
        while not self._stop.is_set():
            try:
                yield self.read_frame()
            except queue.Empty:
                yield np.zeros(self.frame_len, dtype=np.int16)
