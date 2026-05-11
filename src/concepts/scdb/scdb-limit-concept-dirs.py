import os
import shutil
import random

root = "/netscratch/aslam/TCAV/SCDB/concept-dirs"
NUM_SAMPLES = 200

for sub_dir in os.listdir(root):
    sub_dir_path = os.path.join(root, sub_dir)

    # sanity checks
    if not os.path.isdir(sub_dir_path):
        continue

    if "random" in sub_dir:
        continue

    new_dir_path = os.path.join(root, f"{sub_dir}_200")
    os.makedirs(new_dir_path, exist_ok=True)

    # list only files (not directories)
    files = [
        f for f in os.listdir(sub_dir_path)
        if os.path.isfile(os.path.join(sub_dir_path, f))
    ]

    if len(files) < NUM_SAMPLES:
        raise ValueError(
            f"{sub_dir} has only {len(files)} files, cannot sample {NUM_SAMPLES}"
        )

    sampled_files = random.sample(files, NUM_SAMPLES)

    for fname in sampled_files:
        src = os.path.join(sub_dir_path, fname)
        dst = os.path.join(new_dir_path, fname)
        shutil.copy2(src, dst)
