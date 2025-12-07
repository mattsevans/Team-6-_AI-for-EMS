# dac_analog_tests.py
# Purpose: prove the DAC + bus by making activity clearly visible on a scope,
# then drive simple analog patterns. Assumes MCP4921/4922-style 12-bit DAC.

import time
import math
import spidev

# ---------- CONFIG: match your working probe setup ----------
BUS  = 0
CE   = 0          # <-- SET THIS to 0 or 1 to match spi_bus_probe.py
HZ   = 1_000_000  # SPI bit clock
MODE = 0          # MCP49xx is typically mode 0
# ------------------------------------------------------------

# MCP492x config: BUF=1, GA=1x, SHDN=Active => 0b0011 << 12 = 0x3000
BASE_CFG = 0x3000

def pack_mcp492x(u12: int):
    """Return two bytes for a 12-bit code with MCP492x config."""
    u12 = max(0, min(4095, int(u12)))
    word = BASE_CFG | (u12 & 0x0FFF)
    return [(word >> 8) & 0xFF, word & 0xFF]

def open_spi(ce: int = CE):
    spi = spidev.SpiDev()
    spi.open(BUS, ce)
    spi.max_speed_hz = HZ
    spi.mode = MODE
    spi.bits_per_word = 8
    return spi

def write_code(spi: spidev.SpiDev, u12: int):
    """One immediate DAC update (very short transfer)."""
    spi.xfer2(pack_mcp492x(u12))

# replace your existing write_code_burst with this
def write_code_burst(spi, u12, repeats: int = 2048):
    """
    Send the same DAC code as one long burst (<= 4096 bytes).
    Each frame is 2 bytes, so repeats * 2 must be <= 4096.
    """
    frame = pack_mcp492x(u12)
    repeats = int(min(repeats, 2048))  # 2048*2 = 4096 bytes
    tx = frame * repeats
    spi.xfer2(tx)

# ---------------------- TESTS ----------------------

def dc_steps_visible(spi: spidev.SpiDev, repeats: int = 4096, dwell_s: float = 1.0):
    """
    Big DC steps with LONG bursts so you clearly see CE low + SCLK activity.
    Probe SCLK (GPIO11), MOSI (GPIO10), CE (GPIO7/8) while this runs.
    """
    print(f"DC steps (visible): burst repeats={repeats}, dwell={dwell_s}s")
    codes = [0, 1024, 2048, 3072, 3890, 2048]
    while True:
        for c in codes:
            print(f"Set code {c} (burst)…")
            write_code_burst(spi, c, repeats=repeats)
            time.sleep(dwell_s)

def dc_steps(spi: spidev.SpiDev, dwell_s: float = 1.0):
    """
    Original ‘short’ writes (harder to see on a long timebase).
    Use dc_steps_visible() if you want long CE-low windows.
    """
    print(f"DC steps: dwell={dwell_s}s")
    codes = [0, 1024, 2048, 3072, 3890, 2048]
    while True:
        for c in codes:
            print(f"Set code {c}")
            write_code(spi, c)
            time.sleep(dwell_s)

def lowfreq_square_visible(spi: spidev.SpiDev, freq_hz: float = 1.0, seconds: int = 10, repeats: int = 2048):
    """
    Slow square wave between two codes, each edge sent as a visible burst.
    After AC-coupling, you'll see pulses at the amp input; by ear you'll hear thumps.
    """
    print(f"{freq_hz} Hz square (visible) for {seconds}s; repeats={repeats}")
    c_lo, c_hi = 1024, 3072
    half = 0.5 / max(0.01, freq_hz)
    t_end = time.time() + seconds
    while time.time() < t_end:
        write_code_burst(spi, c_hi, repeats=repeats)
        time.sleep(half)
        write_code_burst(spi, c_lo, repeats=repeats)
        time.sleep(half)
    print("Done.")

def tone_visible(spi: spidev.SpiDev, freq_hz: float = 1000, seconds: int = 5,
                 table_points: int = 256, amplitude: int = 1600, offset: int = 2048,
                 chunk: int = 32):
    """
    Audible tone with bursts per small chunk so CE/SCLK remain easy to see.
    Not precise audio timing—just a bench test to prove signal path.
    """
    print(f"{freq_hz} Hz tone (visible) for {seconds}s")
    # Build one cycle
    tbl = []
    for n in range(table_points):
        x = math.sin(2 * math.pi * n / table_points)
        u12 = int(offset + amplitude * x)
        tbl.append(max(0, min(4095, u12)))

    period = 1.0 / max(1.0, freq_hz)
    dt = period / table_points
    t_end = time.time() + seconds
    idx = 0
    while time.time() < t_end:
        # send a small chunk as a single burst (visible CE low + SCLK)
        end = idx + chunk
        segment = tbl[idx:end] if end <= len(tbl) else (tbl[idx:] + tbl[:end - len(tbl)])
        # pack the whole segment into one burst
        burst = []
        for u in segment:
            burst.extend(pack_mcp492x(u))
        spi.xfer2(burst)
        idx = (idx + chunk) % len(tbl)
        time.sleep(dt * chunk)
    print("Done.")
