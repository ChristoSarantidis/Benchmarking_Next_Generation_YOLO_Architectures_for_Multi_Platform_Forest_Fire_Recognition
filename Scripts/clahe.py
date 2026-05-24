import os
import cv2

def apply_clahe(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(image)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, enhanced)

def process_folder(input_dir, output_dir):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for root, _, files in os.walk(input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()

            if ext in valid_extensions:
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)

                apply_clahe(input_path, output_path)

                print(f"CLAHE applied: {input_path} -> {output_path}")

# === USER CONFIGURATION ===
parent_folder = "images"
output_folder = os.path.join(os.path.dirname(parent_folder), "clahe_output")

process_folder(parent_folder, output_folder)