# visualization code for my linkedin post - mostly written by GPT-OSS:20b
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "Name": [
        "MolGPS LargeMix",
        "MolGPS Phenomics",
        "CheMeleonMOE",
        "CheMeleon",
        "LLM/GCN Hybrid",
        "RandomForestECFP",
    ],
    "MAE": [0.312, 0.308, 0.317, 0.355, 0.397, 0.460],
    "Pearson": [0.770, 0.764, 0.729, 0.682, 0.654, 0.439],
}
df = pd.DataFrame(data)

# --- Aesthetic Enhancements ---
# Set a professional style
plt.style.use('default') # Reset to default if any other style was active
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif', 'axes.edgecolor':'lightgray', 'axes.linewidth':0.5})

# Define Colors
HIGHLIGHT_COLOR = "#E65100"  # Vibrant Orange/Red
DEFAULT_COLOR = "#0D47A1"    # Deep Blue

# Get colors for sorted data
colors_mae = [HIGHLIGHT_COLOR if name == "CheMeleonMOE" else DEFAULT_COLOR for name in df["Name"]]
colors_pearson = [HIGHLIGHT_COLOR if name == "CheMeleonMOE" else DEFAULT_COLOR for name in df["Name"]]

# Create the figure with a better aspect ratio for horizontal bars
fig, axes = plt.subplots(1, 2, figsize=(7, 4)) # Wider and taller

# --- MAE horizontal bar chart ---
ax0 = axes[0]
bars1 = ax0.barh(df["Name"], df["MAE"], color=colors_mae, height=0.7) # Use barh

# Add value labels inside the bars
for bar in bars1:
    width = bar.get_width()
    ax0.text(width - 0.005, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", ha="right", va="center", fontsize=10, color="white", weight="bold")

ax0.set_title("MAE (↓)", fontsize=14, weight="bold")
# ax0.set_xlabel("MAE Score", fontsize=10)
# Clean up axes
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.tick_params(axis='y', length=0) # Remove y ticks
ax0.set_xlim(0, df["MAE"].max() + 0.03) # Adjust limit for labels
ax0.invert_yaxis() # Display best performers at the top
ax0.xaxis.grid(True, linestyle='--', alpha=0.6) # Add subtle vertical grid

# --- Pearson r horizontal bar chart ---
ax1 = axes[1]
bars2 = ax1.barh(df["Name"], df["Pearson"], color=colors_pearson, height=0.7) # Use barh

# Add value labels inside the bars
for bar in bars2:
    width = bar.get_width()
    ax1.text(width - 0.01, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", ha="right", va="center", fontsize=10, color="white", weight="bold")

ax1.set_title("Pearson r (↑)", fontsize=14, weight="bold")
# ax1.set_xlabel("Pearson r Score", fontsize=10)
# Clean up axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='y', length=0) # Remove y ticks
ax1.set_xlim(0.35, 0.8) # Adjust limit for labels
ax1.invert_yaxis() # Display best performers at the top
ax1.xaxis.grid(True, linestyle='--', alpha=0.6) # Add subtle vertical grid
ax1.tick_params(axis='y', labelleft=False, length=0)


plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("model_performance_linkedin.png", dpi=300, bbox_inches="tight")
