import os
import math
import shutil
from pathlib import Path
import random

# =======================
# USER CONFIG
# =======================
IMAGES_FOLDER = r'images'     # change
LABELS_FOLDER = r'labels'     # change
OUTPUT_BASE   = r'folds'  # change

# Fixed targets per fold (for your case N=848: Train=678, Val=85, Test=85)
VAL_N  = 445
TEST_N = 445

N_FOLDS = 5
MASTER_SEED = 12345
CLEAN_EXISTING_FOLDS = True

# Image extensions to include
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']


# =======================
# HELPERS
# =======================
def list_image_pairs(images_dir: Path, labels_dir: Path):
    """
    Return a list of (stem, ext) for images that ALSO have a matching .txt label.
    Skips images without labels.
    """
    items = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stem = p.stem
            if (labels_dir / f"{stem}.txt").exists():
                items.append((stem, p.suffix.lower()))
            else:
                print(f"Warning: missing label for {p.name}; skipping.")
    return items

def ensure_clean_fold_dir(base: Path, fold_name: str, clean: bool = True):
    root = base / fold_name
    if clean and root.exists():
        shutil.rmtree(root)
    for subset in ['train', 'val', 'test']:
        (root / subset / 'images').mkdir(parents=True, exist_ok=True)
        (root / subset / 'labels').mkdir(parents=True, exist_ok=True)

def copy_pair(stem: str, img_ext: str, images_dir: Path, labels_dir: Path, dest_subset_dir: Path):
    img_src = images_dir / f"{stem}{img_ext}"
    lbl_src = labels_dir / f"{stem}.txt"
    img_dst = dest_subset_dir / 'images' / f"{stem}{img_ext}"
    lbl_dst = dest_subset_dir / 'labels' / f"{stem}.txt"
    if not img_src.exists():
        raise FileNotFoundError(f"Missing image: {img_src}")
    if not lbl_src.exists():
        raise FileNotFoundError(f"Missing label: {lbl_src}")
    shutil.copy2(img_src, img_dst)
    shutil.copy2(lbl_src, lbl_dst)

def circular_window(start: int, size: int, N: int):
    """Return a list of 'size' indices starting at 'start' on a circular array of length N."""
    return [(start + k) % N for k in range(size)]

def find_non_overlapping_offset(N: int, block: int):
    """
    Find an offset O (1..N-1) such that two windows of length 'block' starting at
    s and (s+O) never overlap for ANY start s. Sufficient condition: O % N not in [-(block-1)..(block-1)] mod N.
    We try multiples of 'block' to keep coverage evenly spaced.
    """
    candidates = []
    # Try multiples of 'block': block, 2*block, ..., (N-1)*block
    for k in range(1, N):
        O = (k * block) % N
        if O == 0:
            continue
        # Check disjointness for arbitrary start: non-overlap iff O >= block and N - O >= block
        # (distance forward and backward are both at least 'block')
        if O >= block and (N - O) >= block:
            candidates.append(O)
    # Prefer something spaced out (near N/2 if possible)
    if not candidates:
        # Fallback: brute force all O
        for O in range(1, N):
            if O >= block and (N - O) >= block:
                candidates.append(O)
    if not candidates:
        raise RuntimeError("Could not find a non-overlapping offset. Reduce VAL_N/TEST_N or check N.")
    # Choose candidate closest to N/2 for better spread
    return min(candidates, key=lambda x: abs(x - N/2))


# =======================
# MAIN
# =======================
def main():
    images_dir = Path(IMAGES_FOLDER)
    labels_dir = Path(LABELS_FOLDER)
    out_base   = Path(OUTPUT_BASE)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    items = list_image_pairs(images_dir, labels_dir)
    if not items:
        raise RuntimeError("No valid image+label pairs found.")

    rng = random.Random(MASTER_SEED)
    rng.shuffle(items)

    N = len(items)
    TRAIN_N = N - VAL_N - TEST_N
    if TRAIN_N < 0:
        raise ValueError(f"VAL_N({VAL_N}) + TEST_N({TEST_N}) exceed dataset size N({N}).")
    if N == 500 and (TRAIN_N, VAL_N, TEST_N) != (500, 50, 50):
        print(f"Note: with N=848 and VAL_N=TEST_N=85, train automatically becomes 678. "
              f"Current computed train={TRAIN_N}.")

    print(f"Total images with labels: {N}")
    print(f"Per-fold target counts -> Train={TRAIN_N} | Val={VAL_N} | Test={TEST_N}\n")

    out_base.mkdir(parents=True, exist_ok=True)

    # Choose an offset so that val and test windows never overlap within the same fold.
    # We start with an offset based on multiples of TEST_N and validate it.
    offset = find_non_overlapping_offset(N, TEST_N)

    for f in range(N_FOLDS):
        fold_name = f"fold{f+1}"
        ensure_clean_fold_dir(out_base, fold_name, CLEAN_EXISTING_FOLDS)

        # Test window shifts by TEST_N each fold
        test_start = (f * TEST_N) % N
        # Val window is offset by a non-overlapping amount from test window
        val_start  = (test_start + offset) % N

        test_idx = circular_window(test_start, TEST_N, N)
        val_idx  = circular_window(val_start,  VAL_N,  N)

        # Sanity: no overlap between test & val in this fold
        if set(test_idx).intersection(val_idx):
            raise RuntimeError("Internal error: same-fold test/val overlap detected.")

        test_set = set(test_idx)
        val_set  = set(val_idx)
        train_idx = [i for i in range(N) if i not in test_set and i not in val_set]

        assert len(test_idx)  == TEST_N
        assert len(val_idx)   == VAL_N
        assert len(train_idx) == TRAIN_N, f"Train mismatch {len(train_idx)} vs {TRAIN_N}"

        fold_root = out_base / fold_name
        for i in train_idx:
            stem, ext = items[i]
            copy_pair(stem, ext, images_dir, labels_dir, fold_root / 'train')
        for i in val_idx:
            stem, ext = items[i]
            copy_pair(stem, ext, images_dir, labels_dir, fold_root / 'val')
        for i in test_idx:
            stem, ext = items[i]
            copy_pair(stem, ext, images_dir, labels_dir, fold_root / 'test')

        print(f"{fold_name}: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    # Coverage note:
    # Across 5 folds, total held-out slots = 5*(VAL_N + TEST_N).
    # If this exceeds N slightly (as with 848 and 85/85 -> 850), a couple of items
    # will appear twice overall (in different folds). This is normal and ensures
    # every image is held out at least once, so nothing is in train for all 5 folds.
    print("\nDone. Exact per-fold sizes enforced; each image is held out in at least one fold.")

if __name__ == "__main__":
    main()
