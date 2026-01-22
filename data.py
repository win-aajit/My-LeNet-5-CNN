import os
import pandas as pd
from PIL import Image
import io

splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

def save_split(df, split_name):
    out_root = f"mnist_images/{split_name}"
    os.makedirs(out_root, exist_ok=True)

    for d in range(10):
        os.makedirs(os.path.join(out_root, str(d)), exist_ok=True)

    print(f"Processing {split_name}...")

    for i, row in df.iterrows():
        label = int(row["label"])

        # Extract PNG bytes from the parquet row
        png_bytes = row["image"]["bytes"]

        # Convert bytes → PIL Image
        img = Image.open(io.BytesIO(png_bytes)).convert("L")  # ensure grayscale

        # Resize to 32×32 for LeNet-5
        img32 = img.resize((32, 32), Image.BILINEAR)

        # Save as PNG into digit folder
        out_path = os.path.join(out_root, str(label), f"{i}.png")
        img32.save(out_path)

    print(f"Done {split_name}")

save_split(df_train, "train")
save_split(df_test, "test")