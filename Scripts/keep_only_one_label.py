from pathlib import Path
import shutil

def clean_yolo_labels(folder: str, recursive: bool = False, make_backup: bool = True):
    folder_path = Path(folder)

    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pattern = "**/*.txt" if recursive else "*.txt"
    txt_files = list(folder_path.glob(pattern))

    total_files_changed = 0
    total_lines_removed = 0

    for txt_path in txt_files:
        try:
            original = txt_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        except Exception as e:
            print(f"[SKIP] {txt_path} (read error: {e})")
            continue

        # Keep lines that do NOT start with class 1 or 2.
        # We parse the first token robustly (handles leading spaces/tabs).
        kept = []
        removed_here = 0

        for line in original:
            stripped = line.strip()
            if not stripped:
                kept.append(line)  # keep empty lines (or you can drop them if you prefer)
                continue

            first_token = stripped.split()[0]
            if first_token in {"1", "2"}:
                removed_here += 1
            else:
                kept.append(line)

        if removed_here > 0:
            if make_backup:
                backup_path = txt_path.with_suffix(txt_path.suffix + ".bak")
                # Avoid overwriting an existing backup
                if not backup_path.exists():
                    shutil.copy2(txt_path, backup_path)

            txt_path.write_text("".join(kept), encoding="utf-8")

            total_files_changed += 1
            total_lines_removed += removed_here
            print(f"[OK] {txt_path.name}: removed {removed_here} line(s)")

    print(f"\nDone. Changed files: {total_files_changed}/{len(txt_files)}")
    print(f"Total lines removed: {total_lines_removed}")

if __name__ == "__main__":
    # EDIT THIS:
    FOLDER = r"labels"

    # Set recursive=True if labels are inside subfolders too.
    clean_yolo_labels(FOLDER, recursive=False, make_backup=True)