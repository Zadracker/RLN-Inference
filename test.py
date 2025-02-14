import numpy as np
from pathlib import Path
import time
import torch
import torch.nn as nn

from data import data_loaders
from model import RandLANet
from utils.ply import read_ply, write_ply

# Start time measurement
t0 = time.time()

# Define paths
CHECKPOINT_PATH = Path("models/checkpoint.tar")  # Model checkpoint
TEST_PLY_PATH = Path("data/L18-1-M01-001.ply")  # Input .ply file
OUTPUT_PLY_PATH = Path("output/output.ply")  # Output .ply file
OUTPUT_TXT_PATH = Path("output/output.txt")  # Output labels

# Set device to CPU (since you are not using GPU)
device = torch.device("cpu")

print("Loading data...")
loader, _ = data_loaders(TEST_PLY_PATH)

print("Loading model...")
d_in = 6
num_classes = 14

# Initialize model
model = RandLANet(d_in, num_classes, 16, 4, device)

# Load checkpoint (ensure it's loaded to CPU)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load data
points, labels = next(iter(loader))

print("Predicting labels...")
with torch.no_grad():
    points = points.to(device)
    labels = labels.to(device)
    scores = model(points)
    predictions = torch.max(scores, dim=-2).indices
    accuracy = (predictions == labels).float().mean()  # TODO: Compute mIoU
    print("Accuracy:", accuracy.item())
    predictions = predictions.cpu().numpy()

print("Writing results...")

# Save predictions to output.txt
np.savetxt(OUTPUT_TXT_PATH, predictions, fmt="%d", delimiter="\n")

# Assign labels to point cloud and save as .ply
print("Assigning labels to the point cloud...")
cloud = points.squeeze(0)[:, :3]  # Extract x, y, z coordinates
write_ply(OUTPUT_PLY_PATH, [cloud, predictions], ["x", "y", "z", "class"])

t1 = time.time()
print(f"Done. Time elapsed: {t1 - t0:.1f}s")
