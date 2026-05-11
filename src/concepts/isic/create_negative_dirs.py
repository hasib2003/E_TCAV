
import random
from pathlib import Path
import shutil

ROOT = Path("/netscratch/aslam/TCAV/derm7pt/concepts")  # your current root directory
NUM_FILES = 70       # number of files per new folder
NUM_RANDOM_FOLDERS = 40  # for example, how many random_50_* folders you want

# Step 1: collect all files from all negative_* folders
all_neg_files = []
for neg_folder in ROOT.glob("negative_*"):
    if neg_folder.is_dir():
        all_neg_files.extend([f for f in neg_folder.iterdir() if f.is_file()])

print(f"Total negative files collected: {len(all_neg_files)}")

# Step 2: create random_50_* folders and move sampled files
for i in range(1, NUM_RANDOM_FOLDERS + 1):
    random_folder = ROOT / f"random_50_{i}"
    random_folder.mkdir(exist_ok=True)

    # randomly sample without replacement
    sampled_files = random.sample(all_neg_files, min(NUM_FILES, len(all_neg_files)))

    # move files
    for f in sampled_files:
        shutil.move(str(f), random_folder / f.name)
        # also remove moved files from the pool to avoid reusing them
        all_neg_files.remove(f)

    print(f"Created {random_folder} with {len(sampled_files)} files")
