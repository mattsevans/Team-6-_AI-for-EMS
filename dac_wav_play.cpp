#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>
#include <time.h>

#include <signal.h>
// Build a 16-bit MCP4822 frame:
// [15]  A/B   (0 = channel A, 1 = channel B)
// [14]  don't care (0)
// [13]  GA    (1 = 1x, 0 = 2x)
// [12]  SHDN  (1 = active, 0 = shutdown)
// [11:0]  value (12-bit)
static volatile sig_atomic_t g_stop = 0;
static void handle_sigint(int) {
    g_stop = 1;
}


static void die(const char *msg) {
    perror(msg);
    std::exit(1);
}

static uint16_t mcp4822_frame(uint16_t code12, bool gain_x2, bool shdn, bool channelB) {
    code12 &= 0x0FFF;
    uint16_t ab = channelB ? 1 : 0;
    uint16_t ga = gain_x2 ? 0 : 1;    // GA=0 => 2x, GA=1 => 1x
    uint16_t sd = shdn ? 1 : 0;       // SHDN=1 => active, 0 => shutdown
    uint16_t word = (ab << 15) | (0 << 14) | (ga << 13) | (sd << 12) | code12;
    return word;
}

static void timespec_add_seconds(struct timespec &ts, double seconds) {
    long sec = static_cast<long>(seconds);
    double frac = seconds - static_cast<double>(sec);
    ts.tv_sec += sec;
    long nsec_add = static_cast<long>(frac * 1e9);
    ts.tv_nsec += nsec_add;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += ts.tv_nsec / 1000000000L;
        ts.tv_nsec = ts.tv_nsec % 1000000000L;
    } else if (ts.tv_nsec < 0) {
        long borrow = (-ts.tv_nsec + 999999999L) / 1000000000L;
        ts.tv_sec -= borrow;
        ts.tv_nsec += borrow * 1000000000L;
    }
}

// Minimal WAV reader for PCM 16-bit mono/stereo
struct WavData {
    uint32_t sampleRate = 0;
    uint16_t channels = 0;
    std::vector<int16_t> samples;  // mono samples
};

static bool read_wav_file(const std::string &path, WavData &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "ERROR: Cannot open WAV file: " << path << "\n";
        return false;
    }

    char riff[4];
    f.read(riff, 4);
    if (f.gcount() != 4 || std::memcmp(riff, "RIFF", 4) != 0) {
        std::cerr << "ERROR: Not a RIFF file.\n";
        return false;
    }

    uint32_t riffSize = 0;
    f.read(reinterpret_cast<char*>(&riffSize), 4); // unused here

    char wave[4];
    f.read(wave, 4);
    if (f.gcount() != 4 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::cerr << "ERROR: Not a WAVE file.\n";
        return false;
    }

    bool haveFmt = false;
    bool haveData = false;
    uint16_t audioFormat = 0;
    uint16_t numChannels = 0;
    uint32_t sampleRate = 0;
    uint16_t bitsPerSample = 0;
    uint32_t dataSize = 0;
    std::streampos dataPos;

    while (f && !(haveFmt && haveData)) {
        char chunkId[4];
        uint32_t chunkSize = 0;
        f.read(chunkId, 4);
        if (f.gcount() != 4) break;
        f.read(reinterpret_cast<char*>(&chunkSize), 4);
        if (!f) break;

        if (std::memcmp(chunkId, "fmt ", 4) == 0) {
            // PCM fmt chunk
            if (chunkSize < 16) {
                std::cerr << "ERROR: fmt chunk too small.\n";
                return false;
            }
            uint16_t blockAlign = 0;
            uint32_t byteRate = 0;

            f.read(reinterpret_cast<char*>(&audioFormat), 2);
            f.read(reinterpret_cast<char*>(&numChannels), 2);
            f.read(reinterpret_cast<char*>(&sampleRate), 4);
            f.read(reinterpret_cast<char*>(&byteRate), 4);
            f.read(reinterpret_cast<char*>(&blockAlign), 2);
            f.read(reinterpret_cast<char*>(&bitsPerSample), 2);

            if (!f) {
                std::cerr << "ERROR: Failed reading fmt chunk.\n";
                return false;
            }

            if (chunkSize > 16) {
                // Skip any extra fmt bytes
                f.seekg(chunkSize - 16, std::ios::cur);
            }

            haveFmt = true;
        } else if (std::memcmp(chunkId, "data", 4) == 0) {
            dataPos = f.tellg();
            dataSize = chunkSize;
            haveData = true;
            // We can break after we have both fmt and data
            if (!haveFmt) {
                // fmt should come before data, but if not, be safe:
                // we continue scanning after data if needed.
                f.seekg(chunkSize, std::ios::cur);
            } else {
                break;
            }
        } else {
            // Skip unknown chunk
            f.seekg(chunkSize, std::ios::cur);
        }
    }

    if (!haveFmt) {
        std::cerr << "ERROR: No fmt chunk in WAV.\n";
        return false;
    }
    if (!haveData) {
        std::cerr << "ERROR: No data chunk in WAV.\n";
        return false;
    }

    if (audioFormat != 1) {
        std::cerr << "ERROR: Only PCM (format 1) WAV supported (got format " << audioFormat << ").\n";
        return false;
    }
    if (bitsPerSample != 16) {
        std::cerr << "ERROR: Only 16-bit WAV supported (got " << bitsPerSample << " bits).\n";
        return false;
    }
    if (numChannels != 1 && numChannels != 2) {
        std::cerr << "ERROR: Only mono or stereo WAV supported (got " << numChannels << " channels).\n";
        return false;
    }

    // Go to data
    f.seekg(dataPos);
    if (!f) {
        std::cerr << "ERROR: Failed seeking to data chunk.\n";
        return false;
    }

    // Read raw samples
    std::vector<int16_t> raw(dataSize / 2);  // 2 bytes per int16
    f.read(reinterpret_cast<char*>(raw.data()), dataSize);
    if (!f) {
        std::cerr << "ERROR: Failed reading WAV data.\n";
        return false;
    }

    // Convert to mono if needed
    std::vector<int16_t> mono;
    if (numChannels == 1) {
        mono = std::move(raw);
    } else {
        // Stereo: downmix by averaging L and R
        size_t frames = raw.size() / 2;
        mono.resize(frames);
        for (size_t i = 0; i < frames; ++i) {
            int32_t L = raw[2*i];
            int32_t R = raw[2*i + 1];
            int32_t avg = (L + R) / 2;
            if (avg < -32768) avg = -32768;
            if (avg >  32767) avg =  32767;
            mono[i] = static_cast<int16_t>(avg);
        }
    }

    out.sampleRate = sampleRate;
    out.channels = 1;
    out.samples = std::move(mono);
    return true;
}

int main(int argc, char **argv) {
    std::signal(SIGINT, handle_sigint);
    std::string wavPath;
    int   bus         = 0;
    int   ce          = 1;
    bool  gain_x2     = false;
    uint32_t spiSpeed = 1000000;

    // Very simple CLI:
    //   --wav <file>  (required)
    //   --bus N
    //   --ce N
    //   --speed Hz
    //   --gain-1x / --gain-2x
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--wav") && i + 1 < argc) {
            wavPath = argv[++i];
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
        } else {
            std::cerr << "Unknown arg: " << argv[i] << "\n";
        }
    }

    if (wavPath.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --wav <file.wav> [--bus N] [--ce N] [--speed Hz] [--gain-1x|--gain-2x]\n";
        return 1;
    }

    WavData wav;
    if (!read_wav_file(wavPath, wav)) {
        return 1;
    }

    std::cout << "Loaded WAV: " << wavPath << "\n";
    std::cout << "  Sample rate: " << wav.sampleRate << " Hz\n";
    std::cout << "  Samples:     " << wav.samples.size() << "\n";

    // Open SPI
    char devPath[32];
    std::snprintf(devPath, sizeof(devPath), "/dev/spidev%d.%d", bus, ce);
    int fd = ::open(devPath, O_RDWR);
    if (fd < 0) die("open(spidev)");

    uint8_t mode = SPI_MODE_0;
    uint8_t bits = 8;
    if (ioctl(fd, SPI_IOC_WR_MODE, &mode) < 0) die("SPI_IOC_WR_MODE");
    if (ioctl(fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0) die("SPI_IOC_WR_BITS_PER_WORD");
    if (ioctl(fd, SPI_IOC_WR_MAX_SPEED_HZ, &spiSpeed) < 0) die("SPI_IOC_WR_MAX_SPEED_HZ");

    std::cout << "MCP4822 WAV playback on " << devPath << "\n";
    std::cout << "  SPI speed: " << spiSpeed << " Hz\n";
    std::cout << "  Gain:      " << (gain_x2 ? "2x" : "1x") << "\n";

    // Shut down channel B
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

    // Timing
    const double samplePeriod = 1.0 / static_cast<double>(wav.sampleRate);
    struct timespec next_ts;
    if (clock_gettime(CLOCK_MONOTONIC, &next_ts) != 0) {
        die("clock_gettime");
    }

    // Mapping: int16 [-32768,32767] -> DAC 0..4095 with midscale bias
    const double headroom  = 0.95;
    const double halfRange = 4095.0 / 2.0;
    const double center    = 2048.0;
    const double amp       = headroom * halfRange;

    const size_t totalSamples = wav.samples.size();
    std::cout << "Playing... (approx " 
              << (static_cast<double>(totalSamples) / wav.sampleRate) 
              << " seconds)\n";
    
    for (size_t n = 0; n < totalSamples && !g_stop; ++n) {
        if (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_ts, nullptr) != 0) {
            // ignore interruptions; we still check g_stop
        }

        int16_t s = wav.samples[n];
        // Fetch sample
        // Lower the audio output level BEFORE sending to DAC
        const double atten = 0.12;  // (12% of original amplitude)
        double x = static_cast<double>(s) / 32768.0;  // [-1,1)
        x *= atten;

        double dacVal = center + amp * x;
        if (dacVal < 0.0)   dacVal = 0.0;
        if (dacVal > 4095.0) dacVal = 4095.0;
        uint16_t code12 = static_cast<uint16_t>(std::lround(dacVal));
        uint16_t word   = mcp4822_frame(code12, gain_x2, /*shdn=*/true, /*channelB=*/false);

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
            die("SPI_IOC_MESSAGE (WAV data)");
        }

        timespec_add_seconds(next_ts, samplePeriod);
    }

    {
        // Channel A → actively drive 0 V (code=0, SHDN=1)
        uint16_t wordA = mcp4822_frame(
            0,           // code12 = 0 -> near 0 V output
            gain_x2,     // gain bit (doesn't really matter at code 0)
            true,        // shdn = true => channel ON, output buffer active
            false        // channelB = false => channel A
        );
        uint8_t txA[2] = {
            static_cast<uint8_t>((wordA >> 8) & 0xFF),
            static_cast<uint8_t>(wordA & 0xFF)
        };
        struct spi_ioc_transfer trA;
        std::memset(&trA, 0, sizeof(trA));
        trA.tx_buf = (uintptr_t)txA;
        trA.len = 2;
        trA.speed_hz = spiSpeed;
        trA.bits_per_word = bits;
        trA.cs_change = 0;
        ioctl(fd, SPI_IOC_MESSAGE(1), &trA); // best-effort; ignore errors
    }

    std::cout << "Done.\n";
    ::close(fd);
    return 0;
}
