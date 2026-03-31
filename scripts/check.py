import os

folder_path = "datasets/train/div2k/hr"

existing = set()

# collect existing numbers
for file in os.listdir(folder_path):
    if file.endswith(".png"):
        name = file.split(".")[0]
        if name.isdigit():
            existing.add(int(name))

if not existing:
    print("No valid files found")
    exit()

start = min(existing)
end = max(existing)

missing = []

for i in range(start, end + 1):
    if i not in existing:
        missing.append(f"{i:07d}.png")  # 7 digits like your data

# output
if missing:
    print("Missing files:")
    for f in missing:
        print(f)
else:
    print("No missing files 🎉")

print("Total missing:", len(missing))