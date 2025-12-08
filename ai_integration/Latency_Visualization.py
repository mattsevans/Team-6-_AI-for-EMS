import pandas as pd
import matplotlib.pyplot as plt

# --- Load the Zora latency data ---
df = pd.read_csv("zora_latency_log.csv")

# --- Basic data cleaning ---
# Remove any rows where latency_seconds might be missing
df = df.dropna(subset=["latency_seconds"])

# --- Histogram of latency times ---
plt.figure(figsize=(10,6))
plt.hist(df["latency_seconds"], bins=10, edgecolor='black', color='skyblue')
plt.title("Distribution of Zora Response Times", fontsize=16)
plt.xlabel("Latency (seconds)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig("zora_latency_histogram.png")  # Save histogram
plt.show()

# --- Box Plot of latency times ---
plt.figure(figsize=(8,4))
plt.boxplot(df["latency_seconds"], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
plt.title("Zora Response Time Spread (Boxplot)", fontsize=16)
plt.xlabel("Latency (seconds)", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.savefig("zora_latency_boxplot.png")  # Save box plot
plt.show()
