import requests
import streamlit as st
import base64
from pathlib import Path
import streamlit as st
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/search"
)

print("Using API URL:", API_URL)

st.set_page_config(
    page_title="Multimodal Fashion Retrieval",
    layout="wide"
)


def set_background(image_path):

    image_bytes = Path(image_path).read_bytes()

    encoded = base64.b64encode(
        image_bytes
    ).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(
                    rgba(0, 0, 0, 0.10),
                    rgba(0, 0, 0, 0.10)
                ),
                url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, p, label {{
            color: white !important;
            text-shadow: 0 2px 6px rgba(0, 0, 0, 0.6);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background("assets/background.jpg")


st.title("Multimodal Fashion Retrieval")

st.write(
    "Upload a reference image and describe "
    "how you want the item to change."
)


uploaded_file = st.file_uploader(
    "Reference image",
    type=["jpg", "jpeg", "png"]
)


text = st.text_input(
    "Modification",
    placeholder="e.g. make it red with short sleeves"
)


top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=10,
    value=5
)


if uploaded_file is not None:

    st.subheader("Reference")

    st.image(
        uploaded_file,
        width=300
    )


if st.button("Search"):

    if uploaded_file is None:
        st.error("Please upload an image.")

    elif not text.strip():
        st.error("Please enter a modification.")

    else:

        with st.spinner("Searching..."):

            files = {
                "image": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            data = {
                "text": text,
                "top_k": top_k
            }

            response = requests.post(
                API_URL,
                files=files,
                data=data
            )

        if response.status_code != 200:

            st.error(
                f"API error: {response.text}"
            )

        else:

            response_data = response.json()

            results = response_data["results"]

            st.subheader("Top Results")

            columns = st.columns(
                min(len(results), 5)
            )

            for i, result in enumerate(results):

                column = columns[
                    i % len(columns)
                ]

                with column:

                    st.image(
                        result["image_url"],
                        use_container_width=True
                    )

                    st.write(
                        f"**Rank {i + 1}**"
                    )

                    st.caption(
                        result["image_id"]
                    )

                    st.write(
                        f"Score: "
                        f"{result['score']:.4f}"
                    )