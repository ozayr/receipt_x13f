import io
import streamlit as st
from doctr.io import DocumentFile
from PIL import Image
from pi_heif import register_heif_opener
import numpy as np
import cv2
import pandas as pd

from doctr.models import ocr_predictor
from utils.ocr_formatter import format_document
from utils.data_processsor import add_one_row_to_dataframe
import streamlit_scrollable_textbox as stx
from utils.gpt_receipt_processor import GPTReceiptProcessor
from keys import openai_api_key
import time

register_heif_opener()


@st.cache_resource
def load_ocr_model():
    model = ocr_predictor(
        det_arch="fast_small",
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
        assume_straight_pages=False,
        detect_orientation=True,
    )

    model.det_predictor.model.postprocessor.bin_thresh = 0.3
    model.det_predictor.model.postprocessor.box_thresh = 0.1
    return model


@st.cache_resource
def load_receipt_processor():
    return GPTReceiptProcessor(api_key=openai_api_key, model_name="gpt-4o")


def load_dataframe():
    return pd.read_pickle("receipts.pkl").sort_values("datetime")


def submit_receipt():
    df = load_dataframe()
    data_extract = st.session_state.data_extract
    df = add_one_row_to_dataframe(df=df, data=data_extract, image_path="")
    df.to_pickle("receipts.pkl")
    st.success("Receipt submitted successfully")
    time.sleep(2)
    st.balloons()


image_cols = st.columns((1, 1), vertical_alignment="bottom")
image_cols[0].subheader("Input Receipt")
data_cols = st.columns((1, 1))
# Sidebar
# File selection
st.sidebar.title("Receipt Upload")
model = None
with st.spinner("Loading model..."):
    model = load_ocr_model()
with st.spinner("Loading receipt processor..."):
    receipt_processor = load_receipt_processor()

uploaded_file = st.sidebar.file_uploader(
    "Upload files", type=["png", "jpeg", "jpg", "heic"]
)
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    image = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB))[
        1
    ].tobytes()
    receipt_image = DocumentFile.from_images(image)

    image_cols[0].image(receipt_image)
side_bar_btn_bar = st.sidebar.columns(2)
data_extract = None
if side_bar_btn_bar[0].button("Analyze Receipt"):
    if uploaded_file is None:
        st.sidebar.write("Please upload a document")

    else:
        with st.spinner("Analyzing..."):
            inference_result = model(receipt_image)
            formatted_result = format_document(
                inference_result.pages[0].export()["blocks"][0]["lines"]
            )
            image_cols[1].image(inference_result.pages[0].synthesize(), clamp=True)
            data_cols[0].text_area("OCR Output", formatted_result, height=500)
            data_extract = receipt_processor.extract_data_from_text(formatted_result)
            data_cols[1].json(data_extract, expanded=True)
            st.session_state.data_extract = data_extract

if data_extract is not None:
    st.button("Submit Receipt", on_click=submit_receipt)
