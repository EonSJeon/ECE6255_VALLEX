# import os
# import pandas as pd
# import shutil
# from pathlib import Path

# # --- CONFIG ---
# DATA_DIR = Path("./customs/train_data")
# CSV_IN   = DATA_DIR / "txt.done.csv"
# WAV_DIR  = DATA_DIR / "wav"
# OUT_DIR  = Path("split_data")
# TRAIN_RATIO = 0.8
# SEED = 42

# # Create output directories
# train_csv = OUT_DIR / "train.csv"
# val_csv   = OUT_DIR / "val.csv"
# train_wav = OUT_DIR / "train_wav"
# val_wav   = OUT_DIR / "val_wav"
# for d in (OUT_DIR, train_wav, val_wav):
#     d.mkdir(parents=True, exist_ok=True)

# # 1) Load and shuffle
# df = pd.read_csv(CSV_IN)
# df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# # 2) Split
# split_idx = int(len(df) * TRAIN_RATIO)
# train_df = df.iloc[:split_idx]
# val_df   = df.iloc[split_idx:]

# # 3) Write csvs
# train_df.to_csv(train_csv, index=False)
# val_df.to_csv(val_csv, index=False)
# print(f"Wrote {len(train_df)} rows to {train_csv}")
# print(f"Wrote {len(val_df)} rows to {val_csv}")

# # 4) Copy wav files
# def copy_wavs(subset_df, dest_dir):
#     for uid in subset_df["id"]:
#         src = WAV_DIR / f"{uid}.wav"
#         dst = dest_dir / f"{uid}.wav"
#         if src.exists():
#             shutil.copy(src, dst)
#         else:
#             print(f"WARNING: missing {src}")

# copy_wavs(train_df, train_wav)
# copy_wavs(val_df,   val_wav)
# print(f"Copied train wavs → {train_wav}")
# print(f"Copied val   wavs → {val_wav}")


from customs.make_custom_dataset import create_dataset

data_dir = "./customs/valid_data"
create_dataset(data_dir, dataloader_process_only=True)

data_dir = "./customs/train_data"
create_dataset(data_dir, dataloader_process_only=True)