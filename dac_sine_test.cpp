#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <time.h>

// Simple helper for error exit
static void die(const char *msg) {
    perror(msg);
    std::exit(1);
}

// Build a 16-bit MCP4822 frame:
// [15]  A/B   (0 = channel A, 1 = channel B)
// [14]  don't care (0)
// [13]  GA    (1 = 1x, 0 = 2x)
// [12]  SHDN  (1 = active, 0 = shutdown)
// [11:0]  value (12-bit)
static uint16_t mcp4822_frame(uint16_t code12, bool gain_x2, bool shdn, bool channelB) {
    code12 &= 0x0FFF;
    uint16_t ab = channelB ? 1 : 0;
    uint16_t ga = gain_x2 ? 0 : 1;    // GA=0 => 2x, GA=1 => 1x
    uint16_t sd = shdn ? 1 : 0;       // SHDN=1 => active, 0 => shutdown
    uint16_t word = (ab << 15) | (0 << 14) | (ga << 13) | (sd << 12) | code12;
    return word;
}

// Add seconds (double) to a timespec (absolute time arithmetic)
static void timespec_add_seconds(struct timespec &ts, double seconds) {
    long sec = static_cast<long>(seconds);
    double frac = seconds - static_cast<double>(sec);
    ts.tv_sec += sec;
    long nsec_add = static_cast<long>(frac * 1e9);
    ts.tv_nsec += nsec_add;
    // Normalize
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += ts.tv_nsec / 1000000000L;
        ts.tv_nsec = ts.tv_nsec % 1000000000L;
    } else if (ts.tv_nsec < 0) {
        long borrow = (-ts.tv_nsec + 999999999L) / 1000000000L;
        ts.tv_sec -= borrow;
        ts.tv_nsec += borrow * 1000000000L;
    }
}

int main(int argc, char **argv) {
    // Defaults similar to your Python test
    int   bus         = 0;          // SPI bus
    int   ce          = 1;          // SPI chip-select (CE1)
    int   sampleRate  = 16000;      // Hz
    double freq       = 1000.0;     // Hz (sine frequency)
    double duration   = 2.0;        // seconds
    bool  gain_x2     = true;       // GA=0 => 2x mode
    uint32_t spiSpeed = 1000000;    // Hz (1 MHz)

    // Very minimal CLI parsing: you can extend if needed
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--freq") && i + 1 < argc) {
            freq = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--rate") && i + 1 < argc) {
            sampleRate = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--dur") && i + 1 < argc) {
            duration = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--bus") && i + 1 < argc) {
            bus = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--ce") && i + 1 < argc) {
            ce = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--speed") && i + 1 < argc) {
            spiSpeed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--gain-1x")) {
            gain_x2 = false;
        } else if (!std::strcmp(argv[i], "--gain-2x")) {
            gain_x2 = true;
        }
    }

    // Build device path, e.g. "/dev/spidev0.1"
    char devPath[32];
    std::snprintf(devPath, sizeof(devPath), "/dev/spidev%d.%d", bus, ce);

    int fd = ::open(devPath, O_RDWR);
    if (fd < 0) die("open(spidev)");

    uint8_t mode = SPI_MODE_0;
    uint8_t bits = 8;

    if (ioctl(fd, SPI_IOC_WR_MODE, &mode) < 0) die("SPI_IOC_WR_MODE");
    if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) die("SPI_IOC_WR_BITS_PER_WORD");
    if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &spiSpeed) < 0) die("SPI_IOC_WR_MAX_SPEED_HZ");

    std::cout << "MCP4822 sine test on " << devPath << "\n";
    std::cout << "  Sample rate: " << sampleRate << " Hz\n";
    std::cout << "  Frequency:   " << freq       << " Hz\n";
    std::cout << "  Duration:    " << duration   << " s\n";
    std::cout << "  SPI speed:   " << spiSpeed   << " Hz\n";
    std::cout << "  Gain:        " << (gain_x2 ? "2x" : "1x") << "\n";

    // First, shut down channel B (tri-state) – same as Python
    {
        uint16_t word = mcp4822_frame(0, gain_x2, /*shdn=*/false, /*channelB=*/true);
        uint8_t tx[2] = {
            static_cast<uint8_t>((word >> 8) & 0xFF),
            static_cast<uint8_t>(word & 0xFF)
        };
        struct spi_ioc_transfer tr;
        std::memset(&tr, 0, sizeof(tr));
        tr.tx_buf = (uintptr_t)tx;
        tr.len = 2;
        tr.speed_hz = spiSpeed;
        tr.bits_per_word = bits;
        tr.cs_change = 0;
        if (ioctl(fd, SPI_IOC_MESSAGE(1), &tr) < 0) die("SPI_IOC_MESSAGE (shutdown B)");
    }

    // Prepare timing
    const double samplePeriod = 1.0 / static_cast<double>(sampleRate);
    const int totalSamples = static_cast<int>(duration * sampleRate);

    struct timespec next_ts;
    if (clock_gettime(CLOCK_MONOTONIC, &next_ts) != 0) {
        die("clock_gettime");
    }

    // Precompute phase increment for the sine
    const double twoPi = 2.0 * M_PI;
    const double phaseInc = twoPi * freq / static_cast<double>(sampleRate);
    double phase = 0.0;

    // Sine amplitude mapping: same logic as Python:
    // x in [-1,1] → DAC code ≈ 2048 + 0.95 * 2047 * x
    const double headroom = 0.95;
    const double halfRange = 4095.0 / 2.0;
    const double center = 2048.0;
    const double amp = headroom * halfRange;

    for (int n = 0; n < totalSamples; ++n) {
        // Wait until the next sample time
        // (absolute scheduling to reduce drift)
        timespec_add_seconds(next_ts, 0.0); // ensure normalized
        if (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_ts, nullptr) != 0) {
            // If interrupted, we just continue
        }

        // Compute sine sample
        double x = std::sin(phase);   // [-1, +1]
        phase += phaseInc;
        if (phase >= twoPi) phase -= twoPi;

        double dacVal = center + amp * x;
        if (dacVal < 0.0) dacVal = 0.0;
        if (dacVal > 4095.0) dacVal = 4095.0;
        uint16_t code12 = static_cast<uint16_t>(std::lround(dacVal));

        uint16_t word = mcp4822_frame(code12, gain_x2, /*shdn=*/true, /*channelB=*/false);

        uint8_t tx[2] = {
            static_cast<uint8_t>((word >> 8) & 0xFF),
            static_cast<uint8_t>(word & 0xFF)
        };

        struct spi_ioc_transfer tr;
        std::memset(&tr, 0, sizeof(tr));
        tr.tx_buf = (uintptr_t)tx;
        tr.len = 2;
        tr.speed_hz = spiSpeed;
        tr.bits_per_word = bits;
        tr.cs_change = 0;

        if (ioctl(fd, SPI_IOC_MESSAGE(1), &tr) < 0) {
            die("SPI_IOC_MESSAGE (sine frame)");
        }

        // Schedule next sample time
        timespec_add_seconds(next_ts, samplePeriod);
    }

    ::close(fd);
    return 0;
}
