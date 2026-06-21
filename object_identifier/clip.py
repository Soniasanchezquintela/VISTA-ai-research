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

    def compute_best_as_single(self, query_embedding: torch.Tensor, temperature: float = 0.03, verbose: bool = True) -> dict:
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

    def compute_best_top_k(self, query_embedding: torch.Tensor, k: int = 5, temperature: float = 0.03, verbose: bool = True) -> dict:
        # These are initial operating points; tune them with labelled known and
        # unknown product crops from the target camera.
        min_score = 0.70
        min_sku_margin = 0.02
        min_same_sku_in_top_k = 2

        if k < 1:
            raise ValueError("k must be at least 1")

        similarities = self.reference_embeddings @ query_embedding
        if similarities.numel() == 0:
            raise ValueError("The product index contains no reference embeddings")

        top_k_count = min(k, similarities.numel())
        top_scores, top_indices = torch.topk(similarities, k=top_k_count)

        top_candidates = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            entry = self.product_entries[index]
            top_candidates.append(
                {
                    "sku_id": entry["sku_id"],
                    "reference_image_id": entry.get("image_id", "unknown"),
                    "reference_image_path": entry.get("image_path", "unknown"),
                    "score": float(score),
                }
            )

        # Print the top-k candidates for debugging and analysis.
        if verbose:
            print("Top-k candidates:")
            i = 0
            for candidate in top_candidates:
                print(
                    f"[{i}] SKU: {candidate['sku_id']}, "
                    f"score: {candidate['score']:.4f}, "
                    f"image_id: {candidate['reference_image_id']}, "
                )
                i += 1
            print()

        best_index = top_indices[0].item()
        best_entry = self.product_entries[best_index]
        best_sku_id = best_entry["sku_id"]
        best_score = float(top_scores[0].item())
        same_sku_in_top_k = sum(
            candidate["sku_id"] == best_sku_id for candidate in top_candidates
        )

        # Compare the winner to the strongest *different SKU*, rather than to
        # another reference image of the winning SKU.
        runner_up_index = None
        for index in torch.argsort(similarities, descending=True).tolist():
            if self.product_entries[index]["sku_id"] != best_sku_id:
                runner_up_index = index
                break

        if runner_up_index is None:
            runner_up_sku_id = None
            runner_up_score = None
            sku_margin = float("inf")
        else:
            runner_up_sku_id = self.product_entries[runner_up_index]["sku_id"]
            runner_up_score = float(similarities[runner_up_index].item())
            sku_margin = best_score - runner_up_score

        # This remains useful for diagnostics, but is not used as the primary
        # decision signal because it depends on the number of references and
        # the softmax temperature.
        probabilities = torch.softmax(similarities / temperature, dim=0)
        confidence = float(probabilities[best_index].item())

        accepted = best_score >= min_score and (
            sku_margin >= min_sku_margin
            or same_sku_in_top_k >= min_same_sku_in_top_k
        )

        return {
            "sku_id": best_sku_id,
            "reference_image_id": best_entry.get("image_id", "unknown"),
            "reference_image_path": best_entry.get("image_path", "unknown"),
            "score": best_score,
            "confidence": confidence,
            "product": get_product_from_db(best_sku_id),
            "accepted": accepted,
            "sku_margin": sku_margin,
            "runner_up_sku_id": runner_up_sku_id,
            "runner_up_score": runner_up_score,
            "same_sku_in_top_k": same_sku_in_top_k,
            "top_candidates": top_candidates,
        }

    def identify_product(self,
        image,
        temperature: float = 0.03,
        k: int = 1,
        verbose: bool = True
    ) -> dict:

        query_embedding = self.get_image_embedding(image)

        if k == 1:
            return self.compute_best_as_single(query_embedding, temperature=temperature, verbose=verbose)

        return self.compute_best_top_k(query_embedding, k=k, temperature=temperature, verbose=verbose)
