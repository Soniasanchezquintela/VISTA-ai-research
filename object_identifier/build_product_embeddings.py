from pathlib import Path
import json
import sqlite3

import numpy as np
import pandas as pd
import torch
from PIL import Image
import open_clip


DB_PATH = "products.sqlite"
METADATA_CSV = "product_db/metadata.csv"
IMAGE_DIR = Path("product_db/images")
EMBEDDINGS_PATH = "product_db/embeddings/product_embeddings.npy"
IDS_PATH = "product_db/embeddings/product_ids.json"

DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def create_database(metadata_csv: str, db_path: str) -> None:
    df = pd.read_csv(metadata_csv)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS product_images;
        DROP TABLE IF EXISTS products;

        CREATE TABLE products (
            sku_id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            category TEXT NOT NULL
        );

        CREATE TABLE product_images (
            image_id TEXT PRIMARY KEY,
            sku_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            embedding_index INTEGER NOT NULL,
            FOREIGN KEY (sku_id) REFERENCES products(sku_id)
        );
        """
    )

    for sku_id, description, category in df[["sku_id", "description", "category"]].itertuples(
        index=False, name=None
    ):
        print(f"[SQL] Inserting product: {sku_id}, {description}, {category}")
        cur.execute(
            """
            INSERT INTO products (sku_id, description, category)
            VALUES (?, ?, ?)
            """,
            (sku_id, description, category),
        )

    conn.commit()
    conn.close()


def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess


def image_to_embedding(model, preprocess, image_path: Path) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    # Normalize so dot product becomes cosine similarity.
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0].astype("float32")


def infer_sku_id_from_filename(image_path: Path) -> str:
    # Example:
    # soy_milk_low_sugar_001_01.jpg -> soy_milk_low_sugar_001
    parts = image_path.stem.split("_")
    return "_".join(parts[:-1])


def main() -> None:
    Path("product_db/embeddings").mkdir(parents=True, exist_ok=True)

    create_database(METADATA_CSV, DB_PATH)

    model, preprocess = load_model()

    embeddings = []
    product_image_records = []

    image_paths = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.jpeg")) + sorted(IMAGE_DIR.glob("*.png"))

    for embedding_index, image_path in enumerate(image_paths):
        sku_id = infer_sku_id_from_filename(image_path)
        image_id = image_path.stem

        emb = image_to_embedding(model, preprocess, image_path)
        embeddings.append(emb)

        product_image_records.append(
            {
                "image_id": image_id,
                "sku_id": sku_id,
                "image_path": str(image_path),
                "embedding_index": embedding_index,
            }
        )

    embeddings_array = np.vstack(embeddings)
    np.save(EMBEDDINGS_PATH, embeddings_array)

    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(product_image_records, f, indent=2)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for rec in product_image_records:
        cur.execute(
            """
            INSERT INTO product_images (
                image_id, sku_id, image_path, embedding_index
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                rec["image_id"],
                rec["sku_id"],
                rec["image_path"],
                rec["embedding_index"],
            ),
        )

    conn.commit()
    conn.close()

    print(f"Saved {len(embeddings_array)} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
