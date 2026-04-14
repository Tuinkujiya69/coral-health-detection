import os

for split in ["train", "val"]:
    print(f"\n{split.upper()}:")
    for cls in os.listdir(f"dataset/{split}"):
        count = len(os.listdir(f"dataset/{split}/{cls}"))
        print(f"  {cls}: {count} images")