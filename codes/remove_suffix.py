import os

# === CONFIG (edit me) =========================================================
FOLDER = "data"
# ============================================================================

def remove_suffix_after_first_underscore(folder: str):
    for root, _, files in os.walk(folder):
        for fname in files:
            old_path = os.path.join(root, fname)
            name, ext = os.path.splitext(fname)

            if "_" in name:
                new_name = name.split("_", 1)[0] + ext
                new_path = os.path.join(root, new_name)

                # Avoid overwriting
                if os.path.exists(new_path):
                    print(f"Skip (would overwrite): {new_path}")
                    continue

                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    remove_suffix_after_first_underscore(FOLDER)
