import os
import random
import shutil
import argparse

def create_concept_dataset(
    celeba_root,
    output_path,
    split="test",
    attribute_name="Blond_Hair",
    n_sample=500,
    seed=42
):
    """
    Creates directory with images for 'blonde and non blonde haris' for TCAV evaluation.
    Produces two folders:
        - blonde_hair
        - non_blonde_hair
    Each folder contains n_sample images from the specified split.
    """

    random.seed(seed)

    # --------------------------
    # Load attributes
    # --------------------------
    attr_path = os.path.join(celeba_root, "list_attr_celeba.txt")
    with open(attr_path, "r") as f:
        lines = f.readlines()
    attr_names = lines[1].strip().split()
    attr_data = {}
    for line in lines[2:]:
        parts = line.strip().split()
        fname = parts[0]
        values = list(map(int, parts[1:]))
        attr_data[fname] = dict(zip(attr_names, values))

    # --------------------------
    # Load split info
    # --------------------------
    split_path = os.path.join(celeba_root, "list_eval_partition.txt")
    split_map = {}
    with open(split_path, "r") as f:
        for line in f:
            fname, sid = line.strip().split()
            split_map[fname] = int(sid)
    split_id = {"train": 0, "val": 1, "test": 2}[split]

    # --------------------------
    # Filter filenames by split
    # --------------------------
    split_fnames = [fname for fname, sid in split_map.items() if sid == split_id]

    # --------------------------
    # Separate blonde / non blonde hair
    # --------------------------
    blonde_hair_fnames = [fname for fname in split_fnames if attr_data[fname][attribute_name] == 1]
    non_blonde_hair_fnames = [fname for fname in split_fnames if attr_data[fname][attribute_name] == -1]

    # --------------------------
    # Prepare output dirs
    # --------------------------
    blonde_hair_dir = os.path.join(output_path, f"with_{attribute_name}")
    non_blonde_hair_dir = os.path.join(output_path, f"without_{attribute_name}")
    
    os.makedirs(blonde_hair_dir, exist_ok=True)
    os.makedirs(non_blonde_hair_dir, exist_ok=True)

    img_dir = os.path.join(celeba_root, "img_align_celeba")


    # --------------------------
    # Sample images
    # --------------------------
    blonde_hair_samples = random.sample(blonde_hair_fnames, min(n_sample, len(blonde_hair_fnames)))
    non_blonde_hair_samples = random.sample(non_blonde_hair_fnames, min(n_sample, len(non_blonde_hair_fnames)))


    for fname in blonde_hair_samples:
        shutil.copy(os.path.join(img_dir, fname), os.path.join(blonde_hair_dir,fname))

    for fname in non_blonde_hair_samples:
        shutil.copy(os.path.join(img_dir, fname), os.path.join(non_blonde_hair_dir, fname))

    print(f"with_{attribute_name}: {len(blonde_hair_samples)} images")
    print(f"without_{attribute_name}: {len(non_blonde_hair_samples)} images")
    
    print(f"Successfully created concept dataset at {output_path} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeba_root", type=str, default="/ds/images/celeba", help="Path to CelebA dataset")
    parser.add_argument("--attribute_name", type=str, default="Blond_Hair", help="name of the attribute")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save concept images")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_sample", type=int, default=200, help="Number of images per concept")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_concept_dataset(
        celeba_root=args.celeba_root,
        attribute_name=args.attribute_name,
        output_path=args.output_path,
        split=args.split,
        n_sample=args.n_sample,
        seed=args.seed
    )
