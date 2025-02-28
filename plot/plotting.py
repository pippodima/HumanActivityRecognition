import re
import matplotlib
import matplotlib.pyplot as plt
import argparse

matplotlib.use("TkAgg")


# Set up argparse to handle command-line arguments
parser = argparse.ArgumentParser(description="Plot metrics from a log file.")
parser.add_argument("-f","--filename", help="Path to the log file", default="rawLogs/100runLR0.001.txt")
parser.add_argument("-s", "--save", help="Save the plots (default: False)", action="store_true")
args = parser.parse_args()

# Read the log file
with open(args.filename, "r") as file:
    data = file.read()

# Extract loss values
loss_pattern = re.findall(r"round (\d+): ([\d.]+)", data)
rounds_loss = [int(round_num) for round_num, _ in loss_pattern]
loss_values = [float(loss) for _, loss in loss_pattern]

# Extract avg_train_loss values
train_loss_pattern = re.findall(r"\((\d+), ([\d.]+)\)", data)
train_rounds = [int(round_num) for round_num, _ in train_loss_pattern]
train_loss_values = [float(loss) for _, loss in train_loss_pattern]

# Number of rounds for each metric
num_rounds = len(train_rounds) // 5

# Split train_loss_values into five metrics
train_loss = train_loss_values[:num_rounds]
accuracy = train_loss_values[num_rounds:2*num_rounds]
f1_score = train_loss_values[2*num_rounds:3*num_rounds]
precision = train_loss_values[3*num_rounds:4*num_rounds]
recall = train_loss_values[4*num_rounds:5*num_rounds]

# === Combined Figure with 5 Subplots ===
fig, axes = plt.subplots(5, 1, figsize=(10, 22))  # Slightly increased height for better spacing

metrics = [
    ("Loss", rounds_loss, loss_values, train_rounds[:num_rounds], train_loss, "red", "blue"),
    ("Accuracy", train_rounds[num_rounds:2*num_rounds], accuracy, None, None, "green", None),
    ("F1 Score", train_rounds[2*num_rounds:3*num_rounds], f1_score, None, None, "purple", None),
    ("Precision", train_rounds[3*num_rounds:4*num_rounds], precision, None, None, "orange", None),
    ("Recall", train_rounds[4*num_rounds:5*num_rounds], recall, None, None, "brown", None)
]

for i, (title, x1, y1, x2, y2, color1, color2) in enumerate(metrics):
    axes[i].plot(x1, y1, marker="o", linestyle="-", label=title, color=color1)
    if x2 and y2:
        axes[i].plot(x2, y2, marker="s", linestyle="--", label=f"Avg {title}", color=color2)
    axes[i].set_xlabel("Rounds")
    axes[i].set_ylabel(title)
    axes[i].set_title(f"{title} over Rounds")
    axes[i].legend()
    axes[i].grid()

# Adjust layout with extra padding
plt.tight_layout(pad=7.0)  # Adds more space between plots

# If the --save flag is set, save the combined plot
if args.save:
    combined_plot_filename = "combined_metrics_plot.png"
    plt.savefig(combined_plot_filename)

plt.show()

# === Individual Plots ===
for title, x1, y1, x2, y2, color1, color2 in metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(x1, y1, marker="o", linestyle="-", label=title, color=color1)
    if x2 and y2:
        plt.plot(x2, y2, marker="s", linestyle="--", label=f"Avg {title}", color=color2)
    plt.xlabel("Rounds")
    plt.ylabel(title)
    plt.title(f"{title} over Rounds")
    plt.legend()
    plt.grid()

    # If the --save flag is set, save individual plots
    if args.save:
        individual_plot_filename = f"{title.lower().replace(' ', '_')}_plot.png"
        plt.savefig(individual_plot_filename)

    plt.show()
