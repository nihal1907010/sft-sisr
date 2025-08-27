import os

def remove_underscores_from_all_files(target_folder: str) -> None:
    """
    Recursively remove underscores '_' from all filenames in a given folder and its subfolders.

    This function searches through the given folder and all its subdirectories,
    finds files containing underscores in their names, and renames them by removing
    all underscores.

    Examples:
        - "my_file.png" -> "myfile.png"
        - "data_set_2025.csv" -> "dataset2025.csv"

    Parameters
    ----------
    target_folder : str
        The absolute or relative path to the root folder to process.

    Notes
    -----
    - Works recursively for all subfolders.
    - Skips renaming if the filename does not contain underscores.
    - Requires write permissions in the target folder and its subfolders.
    """

    if not os.path.isdir(target_folder):
        raise FileNotFoundError(f"The folder '{target_folder}' does not exist.")

    for root, _, files in os.walk(target_folder):
        for filename in files:
            if "_" in filename:
                old_path = os.path.join(root, filename)
                new_name = filename.replace("_", "")
                new_path = os.path.join(root, new_name)

                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed: '{old_path}' -> '{new_path}'")


# =============================
# CONFIGURATION SECTION
# =============================

# Change this to your main folder path
FOLDER_PATH = "datasets/Test"

# Run the function
remove_underscores_from_all_files(FOLDER_PATH)
