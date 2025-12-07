"""
Sean Bolger NLP for AI wearable device
MP3 to Wav
different methods to produce a wav file from an mp3 file. the bottom one which is not currently commented is recommended
"""
"""import pandas as pd
from pathlib import Path

EMS = pd.read_parquet("SpanishMedTermstrain.parquet")

for idx, row in EMS.iterrows():
    mp3_bytes = row["audio"]["bytes"]
    mp3_path = Path(f"ES_Med_{idx}.mp3")
    mp3_path.write_bytes(mp3_bytes)
    print(f"Wrote MP3: {mp3_path}")

    

import ffmpeg
import imageio_ffmpeg as iio
from pathlib import Path

FFMPEG_EXE = iio.get_ffmpeg_exe()


# point ffmpeg-python at the bundled binary
# (no need for ffprobe here, ffmpeg alone can decode+re‑encode)
for mp3_path in Path(".").glob("ES_Med_*.mp3"):
    wav_path = mp3_path.with_suffix(".wav")
    (
        ffmpeg
        .input(str(mp3_path))                       # let ffmpeg autodetect MP3
        .output(
            str(wav_path),
            format="wav",
            acodec="pcm_s16le",  # 16‑bit PCM
            ac=1,                # mono
            ar=16000             # 16 kHz
        )
        .run(cmd=FFMPEG_EXE, capture_stdout=True, capture_stderr=True)
    )
    print(f"✅ Converted to WAV: {wav_path}")


"""

import subprocess
from pathlib import Path
import imageio_ffmpeg as iio

# Locate the bundled ffmpeg binary
FFMPEG_EXE = iio.get_ffmpeg_exe()

#produces an wav for each mp3 file
for mp3_path in Path(".").glob("ES_Med_*.mp3"):
    wav_path = mp3_path.with_suffix(".wav")

    cmd = [
        FFMPEG_EXE,
        "-y",                        # overwrite without asking
        "-i", str(mp3_path),         # input file
        "-ac", "1",                  # mono
        "-ar", "16000",              # 16 kHz
        "-acodec", "pcm_s16le",      # 16‑bit PCM
        str(wav_path)                # output file
    ]

    print(f"Running:\n  {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60                 
    )
    if proc.returncode != 0:
        print(f"ffmpeg failed for {mp3_path.name}:")
        print(proc.stderr)
        break
    else:
        print(f"Converted {mp3_path.name} → {wav_path.name}")
