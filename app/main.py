from io import BytesIO

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app.inference import create_query_embedding
from app.retrieval import search_catalog
from app.catalog_images import get_image_path


app = FastAPI(
    title="Multimodal Fashion Retrieval API",
    description=(
        "Retrieve fashion items using a reference image "
        "and a natural-language modification."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Multimodal Fashion Retrieval API is running."
    }


@app.get("/catalog-image/{image_id}")
def catalog_image(image_id: str):

    image_path = get_image_path(image_id)

    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail="Image not found."
        )

    return FileResponse(image_path)


@app.post("/search")
async def search(
    image: UploadFile = File(...),
    text: str = Form(...),
    top_k: int = Form(5)
):

    # Validate top_k
    if top_k < 1 or top_k > 50:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 50."
        )

    # Read image
    try:
        image_bytes = await image.read()

        pil_image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # ONNX inference
    try:
        query_embedding = create_query_embedding(
            image=pil_image,
            text=text
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

    # Top-K retrieval
    results = search_catalog(
        query_embedding=query_embedding,
        top_k=top_k
    )

    # Add image URLs
    for result in results:
        result["image_url"] = (
            f"http://127.0.0.1:8000/"
            f"catalog-image/{result['image_id']}"
        )

    return {
        "query": {
            "filename": image.filename,
            "text": text,
            "top_k": top_k
        },
        "results": results
    }