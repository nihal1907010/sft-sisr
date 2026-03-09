import os
import re

def remove_trailing_x_number_from_all_files(target_folder: str) -> None:
    """
    Recursively remove trailing 'x<number>' patterns from filenames in a given folder and all subfolders.

    This function searches through the given folder and all its subdirectories,
    finding files whose names end with the pattern 'x<number>' (before the extension),
    and renames them by removing that part.

    Examples:
        - "imagex2.png"  -> "image.png"
        - "documentx3.txt" -> "document.txt"
        - "videox4" -> "video"

    Parameters
    ----------
    target_folder : str
        The absolute or relative path to the root folder to process.

    Notes
    -----
    - Pattern is case-sensitive (only lowercase 'x' followed by digits).
    - Works on files with and without extensions.
    - All nested subfolders are processed.
    - Requires write permissions in the target folder and subfolders.
    """

    if not os.path.isdir(target_folder):
        raise FileNotFoundError(f"The folder '{target_folder}' does not exist.")

    # Walk through the folder tree
    for root, _, files in os.walk(target_folder):
        for filename in files:
            old_path = os.path.join(root, filename)
            name, extension = os.path.splitext(filename)

            # Match names ending with 'x' followed by digits
            if re.search(r"x\d+$", name):
                new_name = re.sub(r"x\d+$", "", name) + extension
                new_path = os.path.join(root, new_name)

                os.rename(old_path, new_path)
                print(f"Renamed: '{old_path}' -> '{new_path}'")


# =============================
# CONFIGURATION SECTION
# =============================

# Change this to your main folder path
FOLDER_PATH = "/home/nihal/sft-sisr/datasets/train/div2k"

# Run the function
remove_trailing_x_number_from_all_files(FOLDER_PATH)
