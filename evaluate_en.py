#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa, librosa.display
import pyworld as pw
import parselmouth
from language_tool_python import LanguageTool
from pesq import pesq
import whisper
from tqdm import tqdm

# --- helper functions (as before) ---
model = whisper.load_model("base")
def asr_transcribe(path):
    audio, _ = librosa.load(path, sr=16000)
    result = model.transcribe(audio, language="en")
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

def extract_errors(text):
    tool = LanguageTool('en-US', remote_server='https://api.languagetool.org')
    matches = tool.check(text)
    return {(m.offset, m.offset+m.errorLength, m.ruleId) for m in matches}

def grammar_fidelity(orig_text, synth_text):
    o = extract_errors(orig_text)
    s = extract_errors(synth_text)
    tp = len(o & s)
    fn = len(o - s)
    fp = len(s - o)
    rec   = tp/(tp+fn) if tp+fn else 1.0
    prec  = tp/(tp+fp) if tp+fp else 1.0
    f1    = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return prec, rec, f1

def compute_pesq(ref_path, syn_path, fs=16000, mode='wb'):
    ref, _ = librosa.load(ref_path, sr=fs)
    syn, _ = librosa.load(syn_path, sr=fs)
    return pesq(fs, ref, syn, mode)

# --- end helper functions ---

def main():
    DATA_DIR    = Path("./test_dataset")
    TRANS_FILES = {"en": "english_transcripts.csv"}
    VARIANTS    = ["original","pure","LoRA_whole","LoRA_AR","LoRA_NAR"]

    # Will collect rows of summary
    records = []

    for lang, csv_name in TRANS_FILES.items():
        df_text = pd.read_csv(DATA_DIR/lang/csv_name)
        texts   = dict(zip(df_text.id, df_text.text))
        ref_dir = DATA_DIR/lang/"audio"

        for variant in tqdm(VARIANTS, desc=f"{lang} variants"):
            syn_dir = DATA_DIR/lang/variant
            out_root= Path("results")/lang/variant

            for uid, ref_text in tqdm(texts.items(),
                                     total=len(texts),
                                     desc=f"{lang}-{variant} files"):
                ref_wav = ref_dir/f"{uid}.wav"
                # note the file naming pattern
                syn_wav = syn_dir/f"{uid}_{variant}.wav"

                if not ref_wav.exists() or not syn_wav.exists():
                    print(f"⚠️  Missing {uid} in {lang}/{variant}")
                    continue

                # optional: make per-utterance dir for plots
                out_dir = out_root/uid
                ensure_dir(out_dir)

                # load
                y_ref, sr = librosa.load(ref_wav, sr=16000)
                y_syn, _  = librosa.load(syn_wav, sr=sr)

                # # (1) wave/spec
                # plot_waveform   (y_ref, sr, out_dir/"ref_wave.png",   "Reference")
                # plot_spectrogram(y_ref, sr, out_dir/"ref_spec.png",   "Reference")
                # plot_waveform   (y_syn, sr, out_dir/"syn_wave.png",   "Synthesized")
                # plot_spectrogram(y_syn, sr, out_dir/"syn_spec.png",   "Synthesized")

                # (2) pitch & formants
                p_ref = extract_pitch_sp(y_ref, sr)
                p_syn = extract_pitch_sp(y_syn, sr)
                f_ref = extract_formant_jitter_shimmer_hnr(str(ref_wav))
                f_syn = extract_formant_jitter_shimmer_hnr(str(syn_wav))

                # (3) ASR-based grammar fidelity
                hyp = asr_transcribe(str(syn_wav))
                prec, rec, f1 = grammar_fidelity(ref_text, hyp)

                # (4) PESQ
                try:
                    pesq_score = compute_pesq(str(ref_wav), str(syn_wav), fs=sr)
                except Exception:
                    pesq_score = np.nan

                # collect record
                rec = {
                    "lang": lang,
                    "variant": variant,
                    "id": uid,
                    "f0_ref_mean": p_ref["f0_mean"],
                    "f0_syn_mean": p_syn["f0_mean"],
                    "F1_ref": f_ref["F1_mean"],
                    "F1_syn": f_syn["F1_mean"],
                    "jitter_ref": f_ref["jitter"],
                    "jitter_syn": f_syn["jitter"],
                    "prec_ASR": prec,
                    "rec_ASR": rec,
                    "f1_ASR": f1,
                    "pesq": pesq_score
                }
                records.append(rec)
                print(f"Processed {lang}/{variant}/{uid}")

    # write out summary CSV
    summary_df = pd.DataFrame.from_records(records)
    summary_csv = Path("results")/"summary_en.csv"
    ensure_dir(summary_csv.parent)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Summary written to {summary_csv}")

if __name__ == "__main__":
    main()
