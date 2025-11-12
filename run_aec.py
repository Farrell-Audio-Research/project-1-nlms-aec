#!/usr/bin/env python3
# Simple AEC runner using our patched MetaAF loader
import argparse
import numpy as np
import soundfile as sf
import librosa

from metaaf import learners as L

SR = 16000  # expected by the checkpoint / dummy dataset

def load_mono_16k(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32), SR

def main():
    ap = argparse.ArgumentParser(description="Run MetaAF AEC inference (CPU-only).")
    ap.add_argument("--mic", required=True, help="Path to mic (near-end+echo) WAV")
    ap.add_argument("--ref", required=True, help="Path to ref (far-end) WAV")
    ap.add_argument("--model_dir", required=True, help="Path to v1.0.1 aec model dir (e.g., .../v1.0.1_models/aec)")
    ap.add_argument("--out", required=True, help="Output WAV path")
    args = ap.parse_args()

    # Load audio
    d, _ = load_mono_16k(args.mic)
    u, _ = load_mono_16k(args.ref)

    # Build callable from learners (our patched loader)
    # model_name expects something like 'aec' subdir under v1.0.1_models
    # The function itself discovers the run subdir (e.g., 2022_10_19_*).
    aec = L.load_pretrained_model("aec", use_test_init=True)

    # Run
    out = aec(u, d)

    # Save
    sf.write(args.out, out, SR)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
