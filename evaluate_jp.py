#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa, librosa.display
import pyworld as pw
import parselmouth
from pesq import pesq
import whisper
from jiwer import wer 
from tqdm import tqdm

# --- helper functions (as before) ---
model = whisper.load_model("base")
def asr_transcribe(path):
    audio, _ = librosa.load(path, sr=16000)
    result = model.transcribe(audio, language="ja")
    return result["text"]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def plot_waveform(y, sr, out_path, title):
    plt.figure(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"{title} waveform")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def plot_spectrogram(y, sr, out_path, title):
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} spectrogram")
    plt.tight_layout()
    plt.savefig(out_path); plt.close()

def extract_pitch_sp(y, sr):
    f0, _ = pw.harvest(y.astype(np.float64), sr)
    f0v   = f0[f0>0]
    return {'f0_contour': f0,
            'f0_mean': float(np.nanmean(f0v)),
            'f0_std':  float(np.nanstd(f0v))}

def extract_formant_jitter_shimmer_hnr(path):
    snd  = parselmouth.Sound(path)
    form = snd.to_formant_burg()
    times = np.linspace(snd.xmin, snd.xmax, 100)
    f1 = [form.get_value_at_time(1, t) for t in times]
    f2 = [form.get_value_at_time(2, t) for t in times]
    f3 = [form.get_value_at_time(3, t) for t in times]
    pp      = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
    jitter  = parselmouth.praat.call(pp, "Get jitter (local)", 0,0,0.0001,0.02,1.3)
    shimmer = parselmouth.praat.call([snd,pp], "Get shimmer (local)", 0,0,0.0001,0.02,1.3,1.6)
    hnr_obj = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01,75.0,0.1,4.5)
    hnr     = parselmouth.praat.call(hnr_obj, "Get mean", 0,0)
    return {
      'F1_mean':  np.nanmean(f1),
      'F2_mean':  np.nanmean(f2),
      'F3_mean':  np.nanmean(f3),
      'jitter':   float(jitter),
      'shimmer':  float(shimmer),
      'hnr':      float(hnr)
    }


def compute_pesq(ref_path, syn_path, fs=16000, mode='wb'):
    ref, _ = librosa.load(ref_path, sr=fs)
    syn, _ = librosa.load(syn_path, sr=fs)
    return pesq(fs, ref, syn, mode)

# --- end helper functions ---

def main():
    DATA_DIR    = Path("./test_dataset")
    TRANS_FILES = {"jp": "japanese_transcripts.csv"}
    VARIANTS    = ["original","pure","LoRA_whole","LoRA_AR","LoRA_NAR"]

    records = []
    model = whisper.load_model("base")

    for lang, csv_name in TRANS_FILES.items():
        df_text = pd.read_csv(DATA_DIR/lang/csv_name)
        texts   = dict(zip(df_text.id, df_text.text))
        ref_dir = DATA_DIR/lang/"audio"

        # outer bar over variants
        for variant in tqdm(VARIANTS, desc=f"{lang} variants"):
            syn_dir = DATA_DIR/lang/variant

            # inner bar over utterances
            for uid, ref_text in tqdm(
                    list(texts.items()),
                    total=len(texts),
                    desc=f"{lang}/{variant}",
                    leave=False
                ):
                ref_wav = ref_dir/f"{uid}.wav"
                syn_wav = syn_dir/f"{uid}_{variant}.wav"
                if not ref_wav.exists() or not syn_wav.exists():
                    continue

                # … the rest of your processing …
                y_ref, sr = librosa.load(ref_wav, sr=16000)
                y_syn, _  = librosa.load(syn_wav, sr=sr)
                p_ref = extract_pitch_sp(y_ref, sr)
                p_syn = extract_pitch_sp(y_syn, sr)
                f_ref = extract_formant_jitter_shimmer_hnr(str(ref_wav))
                f_syn = extract_formant_jitter_shimmer_hnr(str(syn_wav))

                hyp = asr_transcribe(str(syn_wav))
                try:
                    error_rate = wer(ref_text, hyp)
                except Exception:
                    error_rate = np.nan

                try:
                    pesq_score = compute_pesq(str(ref_wav), str(syn_wav), fs=sr)
                except:
                    pesq_score = np.nan

                records.append({
                    "lang": lang,
                    "variant": variant,
                    "id": uid,
                    "f0_ref_mean": p_ref["f0_mean"],
                    "f0_syn_mean": p_syn["f0_mean"],
                    "F1_ref": f_ref["F1_mean"],
                    "F1_syn": f_syn["F1_mean"],
                    "jitter_ref": f_ref["jitter"],
                    "jitter_syn": f_syn["jitter"],
                    "wer": error_rate,
                    "pesq": pesq_score
                })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv("results/summary_jp.csv", index=False)
    print("✅ Written summary_jp.csv")

if __name__ == "__main__":
    main()
