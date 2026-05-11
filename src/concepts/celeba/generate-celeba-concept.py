import os
import random
import shutil
import argparse

def create_concept_dataset(
    celeba_root,
    output_path,
    split="val",
    attribute_name="Wearing_Necktie",
    n_sample=500,
    n_dirs=30,
    seed=42
):
    """
    Creates concept datasets for '{attribute_name}' for TCAV.
    Produces two folders:
        - with_{attribute_name}
        - without_{attribute_name}
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
    # Separate necktie / no-neck tie
    # --------------------------
    necktie_fnames = [fname for fname in split_fnames if attr_data[fname][attribute_name] == 1]
    no_necktie_fnames = [fname for fname in split_fnames if attr_data[fname][attribute_name] == -1]


    img_dir = os.path.join(celeba_root, "img_align_celeba")

    for idx in range(n_dirs):

        # --------------------------
        # Sample images
        # --------------------------
        necktie_sample = random.sample(necktie_fnames, min(n_sample, len(necktie_fnames)))
        no_necktie_sample = random.sample(no_necktie_fnames, min(n_sample, len(no_necktie_fnames)))

        # --------------------------
        # Copy images
        # --------------------------

        run_neck_dir = os.path.join(output_path,f"with_{attribute_name}_{idx}")
        run_no_neck_dir = os.path.join(output_path,f"without_{attribute_name}_{idx}")

        os.makedirs(run_neck_dir, exist_ok=True)
        os.makedirs(run_no_neck_dir, exist_ok=True)

        print(f"Created neck dir at {run_neck_dir=}")
        print(f"Created no neck dir at {run_no_neck_dir=}")

        

        for fname in necktie_sample:
            shutil.copy(os.path.join(img_dir, fname), os.path.join(run_neck_dir,fname))

        for fname in no_necktie_sample:
            shutil.copy(os.path.join(img_dir, fname), os.path.join(run_no_neck_dir, fname))

        print(f"  with_{attribute_name}: {len(necktie_sample)} images")
        print(f"  without_{attribute_name}: {len(no_necktie_sample)} images")
    
    print(f"Successfully created concept dataset at {output_path} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeba_root", type=str, default="/ds/images/celeba", help="Path to CelebA dataset")
    parser.add_argument("--output_path", type=str, default="/netscratch/aslam/TCAV/celeba/concepts", help="Where to save concept images")
    parser.add_argument("--attribute_name", type=str, default="Wearing_Necktie", help="The attribute to use")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--n_sample", type=int, default=200, help="Number of images per concept")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_concept_dataset(
        celeba_root=args.celeba_root,
        output_path=args.output_path,
        split=args.split,
        attribute_name=args.attribute_name,
        n_sample=args.n_sample,
        seed=args.seed
    )
