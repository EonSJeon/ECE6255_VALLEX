#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import pyworld as pw
import parselmouth
import soundfile as sf
from language_tool_python import LanguageTool

def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def plot_waveform(y, sr, out_path, title):
    plt.figure(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr, color='red')
    plt.title(f"{title} waveform")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_spectrogram(y, sr, out_path, title):
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(10,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} spectrogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def extract_pitch_sp(y, sr):
    f0, t = pw.harvest(y, sr)
    # drop unvoiced (0) for stats
    y64 = y.astype(np.float64)
    f0, t = pw.harvest(y64, sr)
    f0v = f0[f0>0]
    return {
        'f0_contour': f0,
        'f0_mean': float(np.nanmean(f0v)),
        'f0_std':  float(np.nanstd(f0v))
    }

def extract_formant_jitter_shimmer_hnr(path):
    snd = parselmouth.Sound(path)
    form = snd.to_formant_burg()
    t = np.linspace(snd.xmin, snd.xmax, 100)
    f1 = [form.get_value_at_time(1, ti) for ti in t]
    f2 = [form.get_value_at_time(2, ti) for ti in t]
    f3 = [form.get_value_at_time(3, ti) for ti in t]
    # jitter/shimmer/hnr
    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
    jitter  = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([snd,pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    hnr_obj = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 4.5)
    hnr     = parselmouth.praat.call(hnr_obj, "Get mean", 0, 0)
    return {
        'F1_mean': np.nanmean(f1),
        'F2_mean': np.nanmean(f2),
        'F3_mean': np.nanmean(f3),
        'jitter': float(jitter),
        'shimmer': float(shimmer),
        'hnr': float(hnr)
    }

def extract_errors(text):
    tool = LanguageTool('en-US', remote_server='https://api.languagetool.org')
    matches = tool.check(text)
    return {(m.offset, m.offset + m.errorLength, m.ruleId) for m in matches}

def grammar_fidelity(orig_text, synth_text):
    orig_errs  = extract_errors(orig_text)
    synth_errs = extract_errors(synth_text)
    tp = len(orig_errs & synth_errs)
    fn = len(orig_errs - synth_errs)
    fp = len(synth_errs - orig_errs)
    recall    = tp / (tp + fn) if tp+fn else 1.0
    precision = tp / (tp + fp) if tp+fp else 1.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return precision, recall, f1

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ref-audio',  required=True)
    p.add_argument('--syn-audio',  required=True)
    p.add_argument('--ref-text',   required=True)
    p.add_argument('--syn-text',   required=True)
    p.add_argument('--out-dir',    default='results/')
    args = p.parse_args()

    ensure_dir(args.out_dir)
    y_ref, sr = librosa.load(args.ref_audio, sr=16000)
    y_ref = y_ref.astype(np.float64)
    y_syn, _  = librosa.load(args.syn_audio, sr=sr)
    y_syn = y_syn.astype(np.float64)

    # plots
    plot_waveform   (y_ref, sr, os.path.join(args.out_dir,'ref_waveform.png'),   'Reference')
    plot_spectrogram(y_ref, sr, os.path.join(args.out_dir,'ref_spectrogram.png'), 'Reference')
    plot_waveform   (y_syn, sr, os.path.join(args.out_dir,'syn_waveform.png'),   'Synthesized')
    plot_spectrogram(y_syn, sr, os.path.join(args.out_dir,'syn_spectrogram.png'), 'Synthesized')

    # features
    pitch_r = extract_pitch_sp(y_ref, sr)
    pitch_s = extract_pitch_sp(y_syn, sr)
    fmt_r   = extract_formant_jitter_shimmer_hnr(args.ref_audio)
    fmt_s   = extract_formant_jitter_shimmer_hnr(args.syn_audio)

    print("\n=== Pitch Statistics ===")
    for k in ['f0_mean','f0_std']:
        print(f"Ref {k:8s}: {pitch_r[k]:6.1f}  |  Syn {k:8s}: {pitch_s[k]:6.1f}")
    print("\n=== Formants & Perturbation ===")
    for k in ['F1_mean','F2_mean','F3_mean','jitter','shimmer','hnr']:
        print(f"Ref {k:8s}: {fmt_r[k]:6.3f}  |  Syn {k:8s}: {fmt_s[k]:6.3f}")

    # grammar fidelity
    orig_text  = open(args.ref_text,'r',encoding='utf8').read()
    synth_text = open(args.syn_text,'r',encoding='utf8').read()
    prec, rec, f1 = grammar_fidelity(orig_text, synth_text)
    print("\n=== Grammar‐Error Fidelity ===")
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1-score: {f1:.3f}")

if __name__ == "__main__":
    main()
