import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Loop over all summary CSV files in the current directory
for csv_path in glob.glob("epoch_summary-*.csv"):
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Convert valid_loss to numeric (blanks become NaN)
    df['valid_loss'] = pd.to_numeric(df['valid_loss'], errors='coerce')
    
    # Compute cumulative average of train loss
    df['cumavg_train_loss'] = df['train_loss'].expanding().mean()
    
    # Prepare the valid‐loss subset so matplotlib will connect the dots
    valid_df = df.dropna(subset=['valid_loss'])
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'],        marker='o', label='Train Loss')
    plt.plot(valid_df['epoch'], valid_df['valid_loss'], marker='o', label='Valid Loss')
    plt.plot(df['epoch'], df['cumavg_train_loss'], marker='o', linestyle='--', label='Train Loss (Cumulative Avg)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    model_name = os.path.splitext(os.path.basename(csv_path))[0].replace("epoch_summary-", "")
    plt.title(f'{model_name}: Loss Curves per Epoch')
    plt.xticks(df['epoch'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_name = f"{model_name}_loss_curve.png"
    plt.savefig(out_name, dpi=150)
    plt.close()
