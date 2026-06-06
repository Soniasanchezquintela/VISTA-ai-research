import json
import sqlite3
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "products.sqlite"
EMBEDDINGS_PATH = BASE_DIR / "product_db/embeddings/product_embeddings.npy"
IDS_PATH = BASE_DIR / "product_db/embeddings/product_ids.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_product_from_db(sku_id: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT *
            FROM products
            WHERE sku_id = ?
            """,
            (sku_id,),
        ).fetchone()

        if row is None:
            return {
                "sku_id": sku_id,
                "description": "unknown",
                "category": "unknown",
            }

        return dict(row)

    finally:
        conn.close()



class ObjectIdentifier:
    MIN_SCORE: float = 0.70
    MIN_CONFIDENCE: float = 0.80

    def __init__(self):
        self.device = get_device()
        self.model, self.preprocess = self.load_model(self.device)
        self.reference_embeddings, self.product_entries = self.load_product_index()


        print(f"Using device: {self.device}")
        print("Loading model: open_clip ViT-B-32 laion2b_s34b_b79k")
        print(f"Loading embeddings: {EMBEDDINGS_PATH}")
        print(f"Loading IDs: {IDS_PATH}")
        print(f"Using DB: {DB_PATH}")
        print()
        
    def load_model(self, device: torch.device):
        """
        IMPORTANT:
        This must be the same model used when creating product_embeddings.npy.
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )

        model = model.to(device)
        model.eval()

        return model, preprocess

    def load_product_index(self):
        """
        Loads:
        - product embeddings from product_embeddings.npy
        - image/SKU metadata from product_ids.json

        product_ids.json entries are expected to look like:

        {
            "image_id": "yosoy_arroz_coco_1l_001",
            "sku_id": "yosoy_arroz_coco_1l",
            "image_path": "product_db/images/yosoy_arroz_coco_1l_001.jpg",
            "embedding_index": 0
        }
        """
        embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

        with open(IDS_PATH, "r", encoding="utf-8") as f:
            product_entries = json.load(f)

        if len(product_entries) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(product_entries)} JSON entries but "
                f"{len(embeddings)} embeddings."
            )

        embeddings = torch.from_numpy(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.to(self.device)

        return embeddings, product_entries

    def _to_pil_image(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image).convert("RGB")

            if image.ndim == 3:
                if image.shape[2] == 1:
                    return Image.fromarray(image[:, :, 0]).convert("RGB")

                if image.shape[2] == 3:
                    # OpenCV uses BGR by default; convert to RGB for CLIP preprocessing.
                    return Image.fromarray(image[:, :, ::-1]).convert("RGB")

                if image.shape[2] == 4:
                    # Keep alpha-aware conversion explicit when RGBA/BGRA appears.
                    return Image.fromarray(image, mode="RGBA").convert("RGB")

        raise TypeError(f"Unsupported image type for identification: {type(image)}")

    def get_image_embedding(self, image) -> torch.Tensor:
        pil_image = self._to_pil_image(image)
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)

        embedding = torch.nn.functional.normalize(embedding, dim=-1)

        return embedding.squeeze(0)

    def identify_product(self,
        image,
        temperature: float = 0.03,
    ) -> dict:

        query_embedding = self.get_image_embedding(image)

        similarities = self.reference_embeddings @ query_embedding

        best_idx = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_idx].item())

        best_entry = self.product_entries[best_idx]
        best_sku_id = best_entry["sku_id"]

        # With only one product in the database, this will always be 1.0.
        # It only becomes meaningful when you have multiple candidate embeddings.
        probabilities = torch.softmax(similarities / temperature, dim=0)
        confidence = float(probabilities[best_idx].item())

        product = get_product_from_db(best_sku_id)

        return {
            #"image": image_path.name,
            "sku_id": best_sku_id,
            "reference_image_id": best_entry.get("image_id", "unknown"),
            "reference_image_path": best_entry.get("image_path", "unknown"),
            "score": best_score,
            "confidence": confidence,
            "product": product,
        }
