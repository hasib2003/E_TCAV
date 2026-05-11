import pandas as pd
from pathlib import Path
import shutil

CSV_PATH = "/netscratch/aslam/TCAV/derm7pt/meta/meta.csv"
IMAGE_ROOT = Path("/netscratch/aslam/TCAV/derm7pt/images")
OUTPUT_ROOT = Path("/netscratch/aslam/TCAV/derm7pt/concepts")

from pathlib import Path
import shutil

ROOT = Path(OUTPUT_ROOT)  # your current root directory

for concept_dir in ROOT.iterdir():
    if concept_dir.is_dir():
        concept_name = concept_dir.name

        # move positive images
        pos_dir = concept_dir / "positive"
        if pos_dir.exists():
            new_pos_dir = ROOT / f"positive_{concept_name}"
            new_pos_dir.mkdir(exist_ok=True)
            for img in pos_dir.iterdir():
                if img.is_file():
                    shutil.move(str(img), new_pos_dir / img.name)

        # move negative images
        neg_dir = concept_dir / "negative"
        if neg_dir.exists():
            new_neg_dir = ROOT / f"negative_{concept_name}"
            new_neg_dir.mkdir(exist_ok=True)
            for img in neg_dir.iterdir():
                if img.is_file():
                    shutil.move(str(img), new_neg_dir / img.name)

        # remove the now-empty concept folder
        try:
            concept_dir.rmdir()
        except OSError:
            print(f"Could not remove {concept_dir}, folder not empty")



# df = pd.read_csv(CSV_PATH)

# useful_cols = ["pigment_network","pigmentation","streaks","dots_and_globules","vascular_structures"]

# for col in df.columns:
#     unique_vals = df[col].unique()
#     print(f"{col}: {unique_vals}\n")

# exit()

# CONCEPT_LABEL_MAP = {
#     "regular_pigment_network":        ("pigment_network", ["typical"]),
#     "irregular_pigment_network":      ("pigment_network", ["atypical"]),

#     "typical_pigmentation":           ("pigmentation", ["diffuse regular","localized regular"]),
#     "atypical_pigmentation":          ("pigmentation", ["diffuse irregular", "localized irregular"]),

#     "regular_streaks":                ("streaks", ["regular"]),
#     "irregular_streaks":              ("streaks", ["irregular"]),

#     "regular_dots_and_globules":      ("dots_and_globules", ["regular"]),
#     "irregular_dots_and_globules":    ("dots_and_globules", ["irregular"]),

# }

# for concept, (column, positive_labels) in CONCEPT_LABEL_MAP.items():

#     concept_dir = OUTPUT_ROOT / concept
#     (concept_dir / "positive").mkdir(parents=True, exist_ok=True)
#     (concept_dir / "negative").mkdir(parents=True, exist_ok=True)

#     # select rows
#     pos_df = df[df[column].str.lower().isin([x.lower() for x in positive_labels])]
#     neg_df = df[df[column].str.lower() == "absent"]

#     # copy derm images only
#     for _, row in pos_df.iterrows():
#         img_path = IMAGE_ROOT / row["derm"]
#         if img_path.exists():
#             shutil.copy(img_path, concept_dir / "positive")

#     for _, row in neg_df.iterrows():
#         img_path = IMAGE_ROOT / row["derm"]
#         if img_path.exists():
#             shutil.copy(img_path, concept_dir / "negative")

#     print(f"{concept}: {len(pos_df)} positives, {len(neg_df)} negatives")



