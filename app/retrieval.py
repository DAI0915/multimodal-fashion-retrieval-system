from pathlib import Path
import json
import numpy as np


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CATALOG_EMBEDDINGS_PATH = (
    ARTIFACTS_DIR / "catalog_embeddings.npy"
)

CATALOG_IDS_PATH = (
    ARTIFACTS_DIR / "catalog_ids.json"
)


# =========================================================
# Load catalog once
# =========================================================

catalog_embeddings = np.load(
    CATALOG_EMBEDDINGS_PATH
).astype(np.float32)

with open(
    CATALOG_IDS_PATH,
    "r"
) as f:
    catalog_ids = json.load(f)


print(
    f"Loaded catalog: "
    f"{catalog_embeddings.shape}"
)


# =========================================================
# Search
# =========================================================

def search_catalog(
    query_embedding: np.ndarray,
    top_k: int = 5
):

    # [1, 256] -> [256]
    query_embedding = query_embedding.squeeze(0)

    # Dot-product similarity
    similarities = (
        catalog_embeddings @ query_embedding
    )

    # Highest similarity first
    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "image_id": catalog_ids[index],
            "score": float(
                similarities[index]
            )
        })

    return results