import os, shutil, random

SOURCE_DIR = "C:/Users/maria/Downloads/BHD-Corals"

CLASS_MAP = {
    "Bleached": "Bleached Coral",
    "Healthy":  "Healthy Coral",
    "Dead":     "Dead Coral"
}

DEST_DIR = "dataset"
SPLIT    = 0.8
random.seed(42)

for src_name, dest_name in CLASS_MAP.items():
    src_path = os.path.join(SOURCE_DIR, src_name)
    images   = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(images)

    split_idx  = int(len(images) * SPLIT)
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        dest = os.path.join(DEST_DIR, split, dest_name)
        os.makedirs(dest, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(src_path, img), os.path.join(dest, img))

    print(f"{dest_name}: {len(train_imgs)} train, {len(val_imgs)} val")

print("\n✅ Dataset ready!")