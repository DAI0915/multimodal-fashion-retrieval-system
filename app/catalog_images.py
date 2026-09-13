from pathlib import Path
import kagglehub


_image_map = None


def get_image_map():

    global _image_map

    if _image_map is None:

        dataset_root = Path(
            kagglehub.dataset_download(
                "hoangkim14/fashion-iq-dataset"
            )
        )

        print("FashionIQ path:", dataset_root)
        print("Building image map...")

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