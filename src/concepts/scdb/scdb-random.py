import argparse
import os
import random
import shutil
from pathlib import Path

def collect_samples(concept_dir):
    samples = []
    for root, _, files in os.walk(concept_dir):
        for f in files:
            samples.append(os.path.join(root, f))
    return samples


def main(args):
    random.seed(args.seed)

    concept_root = Path(args.concept_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    concept_dirs = [d for d in concept_root.iterdir() if d.is_dir()]
    concept_names = [d.name for d in concept_dirs]

    print(f"Found concepts: {concept_names}")

    # Preload samples per concept
    samples_per_concept = {
        d.name: collect_samples(d)
        for d in concept_dirs
    }

    for concept in concept_names:
        print(f"\nProcessing concept: {concept}")

        # Pool from all OTHER concepts
        pool = []
        for other_concept, samples in samples_per_concept.items():
            if other_concept == concept:
                continue
            pool.extend(samples)

        if len(pool) < args.samples_per_set:
            raise ValueError(
                f"Not enough samples to build random sets for concept '{concept}'"
            )

        for idx in range(args.num_random_sets):
            random_set_dir = output_root / f"{concept}_random_{idx}"
            random_set_dir.mkdir(parents=True, exist_ok=False)

            chosen = random.sample(pool, args.samples_per_set)

            for src in chosen:
                dst = random_set_dir / Path(src).name
                shutil.copy2(src, dst)

            print(f"  Created {random_set_dir} with {len(chosen)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create TCAV random concept directories"
    )
    parser.add_argument(
        "--concept-root",
        type=str,
        default="/netscratch/aslam/TCAV/SCDB/concept-dirs",
        help="Path to concept-dirs"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/netscratch/aslam/TCAV/SCDB/concept-dirs",
        help="Where to write random concept directories"
    )
    parser.add_argument(
        "--samples-per-set",
        type=int,
        default=200,
        help="Number of samples per random set"
    )
    parser.add_argument(
        "--num-random-sets",
        type=int,
        default=30,
        help="Number of random sets per concept"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    main(args)
