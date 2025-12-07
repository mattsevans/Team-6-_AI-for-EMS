# enroll_target.py
import glob, numpy as np, soundfile as sf
from python_speech_features import mfcc

RATE = 16000
EMB_N_MFCC = 24  # match whisper.cppMain.py

def wav_to_embed(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if sr != RATE:
        raise ValueError(f"Expected {RATE} Hz, got {sr}")
    if x.ndim == 2:
        x = x.mean(axis=1)

    # --- MATCHES _compute_embedding in your runtime ---
    m = mfcc(
        x,
        samplerate=RATE,
        winlen=0.025,
        winstep=0.010,
        numcep=EMB_N_MFCC,
        nfilt=26,
        nfft=512,
        preemph=0.97,
        appendEnergy=True
    )
    emb = m.mean(axis=0).astype(np.float32)   # (24,)
    # --------------------------------------------------
    # we'll L2-normalize when *using* it; saving raw is fine
    return emb

def pick_prototypes(embs: np.ndarray, k: int = 6) -> np.ndarray:
    # simple farthest-point picking in cosine space
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    centroid = embs_n.mean(axis=0); centroid /= (np.linalg.norm(centroid) + 1e-9)
    protos = [centroid]
    if len(embs_n) == 1:
        return np.stack(protos, axis=0)
    rest = embs_n.copy()
    for _ in range(max(1, k-1)):
        sims = np.max([rest @ p for p in protos], axis=0)
        idx = int(np.argmin(sims))   # farthest
        protos.append(rest[idx] / (np.linalg.norm(rest[idx]) + 1e-9))
    return np.stack(protos, axis=0)

def main(glob_pat="enroll_clips/enroll_16k/*.wav", out_path="enrolled_target.npz", max_protos=6):
    files = sorted(glob.glob(glob_pat))
    assert files, f"No WAV files found at {glob_pat}"
    embs = np.stack([wav_to_embed(f) for f in files], axis=0)   # (N, D)

    # similarity stats vs. centroid (cosine) for adaptive threshold at runtime
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    centroid_n = embs_n.mean(axis=0); centroid_n /= (np.linalg.norm(centroid_n) + 1e-9)
    sims = embs_n @ centroid_n
    mu, sig = float(sims.mean()), float(sims.std() + 1e-6)
    prototypes = pick_prototypes(embs, k=max_protos)

    # store raw centroid (pre-norm), raw prototypes (pre-norm), and stats
    np.savez(out_path, centroid=embs.mean(axis=0), prototypes=prototypes, mu=mu, sig=sig)
    print(f"Saved {out_path} | mu={mu:.3f}, sig={sig:.3f}, files={len(files)}, protos={prototypes.shape[0]}")

if __name__ == "__main__":
    main()
