import re
import pandas as pd

LOG_PATH = "./train_record/LoRA-AR/log/log-train-2025-04-29-20-30-10"         # <-- update to your log file name
OUT_CSV  = "epoch_summary-LoRA-AR.csv"

# Matches the last batch line of each epoch (train + tot metrics)
train_re = re.compile(
    r"Epoch\s+(\d+),\s+batch\s+\d+,\s+"
    r"train_loss\[loss=([\d\.]+),\s*ArTop10Accuracy=([\d\.]+),[^\]]*\],\s*"
    r"tot_loss\[loss=([\d\.]+),\s*ArTop10Accuracy=([\d\.]+),"
)

# Matches the validation line at the end of each epoch
valid_re = re.compile(
    r"Epoch\s+(\d+),\s+validation:\s*"
    r"loss=([\d\.]+),\s*ArTop10Accuracy=([\d\.]+),"
)

data = {}

with open(LOG_PATH, "r") as f:
    for line in f:
        m = train_re.search(line)
        if m:
            ep = int(m.group(1))
            # overwrite so we keep the last batch entry for the epoch
            data.setdefault(ep, {}).update({
                "train_loss": float(m.group(2)),
                "train_ArTop10Acc": float(m.group(3)),
                "tot_loss": float(m.group(4)),
                "tot_ArTop10Acc": float(m.group(5)),
            })
            continue

        m = valid_re.search(line)
        if m:
            ep = int(m.group(1))
            data.setdefault(ep, {}).update({
                "valid_loss": float(m.group(2)),
                "valid_ArTop10Acc": float(m.group(3)),
            })

# Build sorted DataFrame
rows = []
for ep in sorted(data):
    row = {"epoch": ep}
    row.update(data[ep])
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} epochs to {OUT_CSV}")