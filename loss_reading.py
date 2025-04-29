import glob
from tensorboard.backend.event_processing import event_accumulator
import csv

# 1) Find your event files
event_files = glob.glob("./tensorboard/events.out.tfevents.*")

# 2) Choose the tags you care about
tags = ["train/current_loss", "train/tot_loss", "train/valid_loss"]

# 3) Open a CSV and write a header
with open("loss_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "tag", "step", "value"])

    # 4) For each file, load all scalars and filter by tag
    for ef in event_files:
        ea = event_accumulator.EventAccumulator(ef, size_guidance={"scalars": 0})
        print(ea.Tags()["scalars"])

        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        for tag in tags:
            if tag in available:
                for event in ea.Scalars(tag):
                    writer.writerow([ef, tag, event.step, event.value])

print("Wrote CSV with losses to loss_history.csv")

import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("loss_history.csv")
for tag in df.tag.unique():
    sub = df[df.tag == tag]
    plt.plot(sub.step, sub.value, label=tag)
plt.legend(); plt.xlabel("Step"); plt.ylabel("Loss"); plt.show()
