from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoTokenizer


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ONNX_PATH = ARTIFACTS_DIR / "query_encoder_dynamic.onnx"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer"


# =========================================================
# Load model / tokenizer once
# =========================================================

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"]
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)


# =========================================================
# Image preprocessing
#
# Same as training:
# Resize(256)
# CenterCrop(224)
# ToTensor()
# Normalize(ImageNet mean/std)
# =========================================================

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_image(image: Image.Image) -> np.ndarray:

    image = image.convert("RGB")

    width, height = image.size

    # Resize shorter side to 256
    if width < height:
        new_width = 256
        new_height = round(height * 256 / width)
    else:
        new_height = 256
        new_width = round(width * 256 / height)

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    # CenterCrop(224)
    left = (new_width - 224) // 2
    top = (new_height - 224) // 2

    image = image.crop(
        (
            left,
            top,
            left + 224,
            top + 224
        )
    )

    # HWC -> float [0, 1]
    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # Normalize
    image_array = (
        image_array - IMAGENET_MEAN
    ) / IMAGENET_STD

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array.astype(np.float32)


# =========================================================
# Query encoder
# =========================================================

def create_query_embedding(
    image: Image.Image,
    text: str
) -> np.ndarray:

    image_input = preprocess_image(image)

    encoded_text = tokenizer(
        text,
        return_tensors="np",
        padding=True,
        truncation=True
    )

    inputs = {
        "reference_image": image_input,
        "input_ids": encoded_text[
            "input_ids"
        ].astype(np.int64),
        "attention_mask": encoded_text[
            "attention_mask"
        ].astype(np.int64)
    }

    query_embedding = session.run(
        ["query_embedding"],
        inputs
    )[0]

    return query_embedding