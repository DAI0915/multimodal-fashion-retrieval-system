from pathlib import Path
import kagglehub


_image_map = None
LOCAL_IMAGE_DIR = Path("artifacts/catalog_images")


def get_image_map():

    global _image_map

    if _image_map is None:

        if LOCAL_IMAGE_DIR.exists():
            print("Using local catalog image directory:", LOCAL_IMAGE_DIR)

            _image_map = {
                path.stem: path
                for path in LOCAL_IMAGE_DIR.rglob("*")
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            }

            print(f"Loaded local image paths: {len(_image_map)}")
            return _image_map

        dataset_root = Path(
            kagglehub.dataset_download(
                "hoangkim14/fashion-iq-dataset"
            )
        )

        print("FashionIQ path:", dataset_root)
        print("Building image map from Kaggle dataset...")

        _image_map = {
            path.stem: path
            for path in dataset_root.rglob("*.jpg")
        }

        print(
            f"Loaded image paths: {len(_image_map)}"
        )

    return _image_map


def get_image_path(image_id: str):

    image_map = get_image_map()

    return image_map.get(image_id)