from pathlib import Path
import random
import shutil

# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Make paths relative to project root (robust)
BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "spectrograms"
OUTPUT_DIR = BASE_DIR / "data_split"

random.seed(SEED)


# -----------------------------
# HELPERS
# -----------------------------
def get_images(folder):
    return [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]


def split_files(files):
    random.shuffle(files)

    n = len(files)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = files[:train_end]
    val = files[train_end:val_end]
    test = files[val_end:]

    return train, val, test


def copy(files, split, class_name):
    out_dir = OUTPUT_DIR / split / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        shutil.copy2(f, out_dir / f.name)


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Missing dataset folder: {SOURCE_DIR}")

    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # create base folders
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    # process each class
    for class_dir in SOURCE_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        images = get_images(class_dir)

        if len(images) == 0:
            print(f"Skipping empty class: {class_name}")
            continue

        train, val, test = split_files(images)

        copy(train, "train", class_name)
        copy(val, "val", class_name)
        copy(test, "test", class_name)

        print(
            f"{class_name}: "
            f"{len(train)} train | {len(val)} val | {len(test)} test"
        )

    print("\nDone → dataset split created in data_split/")


if __name__ == "__main__":
    main()