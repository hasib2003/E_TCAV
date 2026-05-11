import os
import random
from torch.utils.data import Dataset
from PIL import Image


class CelebABlondeNecktieBiased(Dataset):
    def __init__(
        self,
        celeba_root="/ds/images/celeba",
        split="train",
        alpha=0.0,
        transform=None,
        seed=42,
    ):
        """
        Args:
            alpha (float): fraction of *additional* unbiased samples
                           relative to the anti-correlated core size.
                           Used ONLY for train split.
        """

        assert split in {"train", "val", "test"}
        assert 0.0 <= alpha <= 1.0

        self.celeba_root = celeba_root
        self.img_dir = os.path.join(celeba_root, "img_align_celeba")
        self.transform = transform

        split_id = {"train": 0, "val": 1, "test": 2}[split]

        # --------------------------------------------------
        # Load attributes
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Load split information
        # --------------------------------------------------
        split_path = os.path.join(celeba_root, "list_eval_partition.txt")
        split_map = {}
        with open(split_path, "r") as f:
            for line in f:
                fname, sid = line.strip().split()
                split_map[fname] = int(sid)

        split_fnames = [
            fname for fname in attr_data
            if split_map[fname] == split_id
        ]

        # ==================================================
        # TRAIN SPLIT: controlled anti-correlation
        # ==================================================
        if split == "train":

            anti_pos = []     # Blond=1, Necktie=-1
            anti_neg = []     # Blond=-1, Necktie=1
            unbiased = []     # everything else

            for fname in split_fnames:
                attrs = attr_data[fname]
                b = attrs["Blond_Hair"]
                n = attrs["Wearing_Necktie"]

                if b == 1 and n == -1:
                    anti_pos.append((fname, attrs))
                elif b == -1 and n == 1:
                    anti_neg.append((fname, attrs))
                else:
                    unbiased.append((fname, attrs))

            # Core perfectly anti-correlated dataset
            core_size = min(len(anti_pos), len(anti_neg))
            assert core_size > 0, "No anti-correlated samples found"

            n_unbiased = int(alpha * core_size)
            core_size  = core_size - n_unbiased

            random.seed(seed)

            pos_sample = random.sample(anti_pos, core_size)
            neg_sample = random.sample(anti_neg, core_size)

            if n_unbiased > 0:
                assert n_unbiased <= len(unbiased)
                unbiased_sample = random.sample(unbiased, n_unbiased)
            else:
                unbiased_sample = []

            self.records = pos_sample + neg_sample + unbiased_sample
            random.shuffle(self.records)

        # ==================================================
        # VAL / TEST SPLIT: untouched CelebA
        # ==================================================
        else:
            self.records = [
                (fname, attr_data[fname])
                for fname in split_fnames
            ]

        # --------------------------------------------------
        # Logging (sanity check)
        # --------------------------------------------------
        n_total = len(self.records)
        n_necktie = sum(attrs["Wearing_Necktie"] == 1 for _, attrs in self.records)
        n_blonde = sum(attrs["Blond_Hair"] == 1 for _, attrs in self.records)

        print(
            f"[CelebA | {split}] "
            f"Total={n_total} | "
            f"Necktie={n_necktie} ({n_necktie/n_total:.3f}) | "
            f"Blonde={n_blonde} ({n_blonde/n_total:.3f})"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, attrs = self.records[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Target: Blond_Hair ∈ {0,1}
        label = 1 if attrs["Blond_Hair"] == 1 else 0
        return img, label


class CelebABlondeNecktieLessBiased(Dataset):
    """
    Biased CelebA dataset:
      - 10% uniform subsample
      - + all samples with Wearing_Necktie == 1
    Target label:
      - Blond_Hair (binary)
    """

    def __init__(
        self,
        celeba_root="/ds/images/celeba",
        split="train",
        sample_frac=0.1,
        transform=None,
        seed=42,
    ):
        self.celeba_root = celeba_root
        self.img_dir = os.path.join(celeba_root, "img_align_celeba")
        self.transform = transform

        assert split in {"train", "val", "test"}
        split_id = {"train": 0, "val": 1, "test": 2}[split]

        # --------------------------------------------------
        # Load attributes
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Load split information
        # --------------------------------------------------
        split_path = os.path.join(celeba_root, "list_eval_partition.txt")
        split_map = {}
        with open(split_path, "r") as f:
            for line in f:
                fname, sid = line.strip().split()
                split_map[fname] = int(sid)

        # Filter by split
        split_fnames = [
            fname for fname in attr_data.keys()
            if split_map[fname] == split_id
        ]

        # --------------------------------------------------
        # Step 1: 10% random sample
        # --------------------------------------------------
        random.seed(seed)
        n_sample = int(sample_frac * len(split_fnames))
        sample_10_fnames = set(random.sample(split_fnames, n_sample))

        # --------------------------------------------------
        # Step 2: all necktie samples
        # --------------------------------------------------
        necktie_fnames = {
            fname for fname in split_fnames
            if attr_data[fname]["Wearing_Necktie"] == 1
        }

        # --------------------------------------------------
        # Final biased dataset
        # --------------------------------------------------
        final_fnames = sample_10_fnames.union(necktie_fnames)

        self.records = [
            (fname, attr_data[fname])
            for fname in sorted(final_fnames)
        ]

        # --------------------------------------------------
        # Sanity logging (keep this)
        # --------------------------------------------------
        n_total = len(self.records)
        n_necktie = sum(attrs["Wearing_Necktie"] == 1 for _, attrs in self.records)
        n_blonde = sum(attrs["Blond_Hair"] == 1 for _, attrs in self.records)

        print(
            f"[CelebA Biased | {split}] "
            f"Total={n_total} | "
            f"Necktie={n_necktie} ({n_necktie/n_total:.3f}) | "
            f"Blonde={n_blonde} ({n_blonde/n_total:.3f})"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, attrs = self.records[idx]

        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB")

        # Target: Blond_Hair ∈ {0,1}
        label = 1 if attrs["Blond_Hair"] == 1 else 0
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class CelebAGender(Dataset):
    def __init__(
        self,
        celeba_root="/ds/images/celeba",
        split="train",
        transform=None,
    ):
        """
        Gender classification on CelebA.
        Uses raw, distributions.
        Target: Male ∈ {0,1}
        """

        assert split in {"train", "val", "test"}

        self.celeba_root = celeba_root
        self.img_dir = os.path.join(celeba_root, "img_align_celeba")
        self.transform = transform

        split_id = {"train": 0, "val": 1, "test": 2}[split]

        # --------------------------------------------------
        # Load attributes
        # --------------------------------------------------
        attr_path = os.path.join(celeba_root, "list_attr_celeba.txt")
        with open(attr_path, "r") as f:
            lines = f.readlines()

        attr_names = lines[1].strip().split()
        assert "Male" in attr_names, "Male attribute not found in CelebA"

        attr_data = {}
        for line in lines[2:]:
            parts = line.strip().split()
            fname = parts[0]
            values = list(map(int, parts[1:]))
            attr_data[fname] = dict(zip(attr_names, values))

        # --------------------------------------------------
        # Load split info
        # --------------------------------------------------
        split_path = os.path.join(celeba_root, "list_eval_partition.txt")
        split_map = {}
        with open(split_path, "r") as f:
            for line in f:
                fname, sid = line.strip().split()
                split_map[fname] = int(sid)

        self.records = [
            (fname, attr_data[fname])
            for fname in attr_data
            if split_map[fname] == split_id
        ]

        # --------------------------------------------------
        # Logging: raw gender distribution
        # --------------------------------------------------
        n_total = len(self.records)
        n_male = sum(attrs["Male"] == 1 for _, attrs in self.records)

        print(
            f"[CelebA | {split}] "
            f"Total={n_total} | "
            f"Male={n_male} ({n_male/n_total:.3f}) | "
            f"Female={n_total - n_male} ({(n_total - n_male)/n_total:.3f})"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        fname, attrs = self.records[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Target: Male ∈ {0,1}
        label = 1 if attrs["Male"] == 1 else 0
        return img, label


if __name__ == "__main__":
    # train_ds =CelebABlondeNecktieBiased(split="train",alpha=0.3)

    # necktie = [attrs["Wearing_Necktie"] == 1 for _, attrs in train_ds.records]
    # blonde = [attrs["Blond_Hair"] == 1 for _, attrs in train_ds.records]


    # necktie = np.array(necktie)
    # blonde = np.array(blonde)

    # p_blond_necktie = blonde[necktie].mean()
    # p_blond_no_necktie = blonde[~necktie].mean()

    # print("P(Blond | Necktie):", p_blond_necktie)
    # print("P(Blond | ¬Necktie):", p_blond_no_necktie)

    train_ds = CelebAGender(split="train")
    