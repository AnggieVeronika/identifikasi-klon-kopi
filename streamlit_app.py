from huggingface_hub import hf_hub_download
from pathlib import Path
import sys

import streamlit as st
import torch
from PIL import Image
import timm
from timm.data.transforms_factory import create_transform

APP_TITLE = "Dashboard Klasifikasi Daun Kopi Robusta"
APP_SUBTITLE = "Upload gambar daun kopi robusta untuk diprediksi menggunakan model DenseNet yang sudah dilatih sebelumnya."
MODEL_PATH = hf_hub_download(
    repo_id="anggiii/densenet-kopi",
    filename="densenet_kopi_best.pth"
)
MAX_TOP_K = 5


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__" and not _running_under_streamlit():
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(stcli.main())


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(48, 97, 60, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(198, 140, 36, 0.16), transparent 22%),
                linear-gradient(180deg, #f6f2ea 0%, #eef4ee 45%, #f9fbf7 100%);
        }
        .hero {
            padding: 2rem 2rem 1.5rem 2rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(23, 74, 42, 0.95), rgba(52, 93, 52, 0.92));
            color: #f7f5ef;
            box-shadow: 0 24px 60px rgba(18, 45, 26, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            line-height: 1.1;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin-top: 0.75rem;
            margin-bottom: 0;
            max-width: 920px;
            font-size: 1.02rem;
            color: rgba(247, 245, 239, 0.88);
        }
        .info-card {
            padding: 1.15rem 1.2rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(26, 63, 39, 0.1);
            box-shadow: 0 10px 28px rgba(40, 63, 46, 0.08);
        }
        .info-card h3 {
            margin: 0 0 0.5rem 0;
            color: #183c25;
            font-size: 1.02rem;
        }
        .muted {
            color: #4f5f54;
            font-size: 0.95rem;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
        }
        .badge {
            display: inline-block;
            padding: 0.42rem 0.72rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            color: #f6f2ea;
            border: 1px solid rgba(255, 255, 255, 0.18);
            font-size: 0.86rem;
        }
        .result-card {
            padding: 1.1rem 1.15rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(27, 67, 40, 0.1);
            box-shadow: 0 14px 34px rgba(38, 58, 42, 0.08);
            margin-bottom: 1rem;
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #5a6b60;
            margin-bottom: 0.35rem;
        }
        .result-label {
            font-size: 1.7rem;
            line-height: 1.1;
            color: #173a24;
            font-weight: 800;
            margin: 0;
        }
        .result-score {
            color: #355540;
            margin-top: 0.45rem;
            font-size: 1rem;
        }
        .topk-item {
            margin-bottom: 0.8rem;
        }
        .topk-label {
            color: #111111;
            font-weight: 700;
            font-size: 0.98rem;
            margin-bottom: 0.15rem;
        }
        .topk-score {
            color: #111111;
            font-size: 0.92rem;
            margin-bottom: 0.35rem;
        }
        .footer-note {
            margin-top: 1.25rem;
            color: #5c6b61;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_and_transform(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu")
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    data_config = checkpoint["data_config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = create_transform(**data_config, is_training=False)
    return {
        "checkpoint": checkpoint,
        "model": model,
        "transform": transform,
        "device": device,
        "class_names": class_names,
    }


def preprocess_image(uploaded_image: Image.Image, transform, device):
    image = uploaded_image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def predict(uploaded_image: Image.Image, model, transform, class_names, device, top_k: int):
    tensor = preprocess_image(uploaded_image, transform, device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))

    return [
        {
            "label": class_names[int(index)],
            "score": float(value),
        }
        for value, index in zip(values, indices)
    ]


st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
        <div class="badge-row">
            <span class="badge">DenseNet checkpoint</span>
            <span class="badge">Upload gambar</span>
            <span class="badge">Top-k prediction</span>
            <span class="badge">Streamlit dashboard</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

try:
    resources = load_model_and_transform(MODEL_PATH)
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()

checkpoint = resources["checkpoint"]
model = resources["model"]
transform = resources["transform"]
device = resources["device"]
class_names = resources["class_names"]

with st.sidebar:
    st.markdown("### Pengaturan")
    top_k = st.slider("Top-k hasil", min_value=1, max_value=min(MAX_TOP_K, len(class_names)), value=min(3, len(class_names)))
    confidence_threshold = st.slider("Ambang keyakinan", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.markdown("### Informasi model")
    st.caption(f"Model: {checkpoint.get('model_name', 'unknown')}")
    st.caption(f"Device: {device}")
    st.caption(f"Jumlah kelas: {len(class_names)}")
    st.caption("Kelas yang dikenali:")
    for class_name in class_names:
        st.caption(f"- {class_name}")

left_column, right_column = st.columns([1.15, 0.95], gap="large")

with left_column:
    st.markdown(
        """
        <div class="info-card">
            <h3>Upload gambar daun kopi robusta</h3>
            <p class="muted">
                Gunakan gambar JPG, PNG, atau WEBP. Dashboard ini akan memproses gambar
                dengan preprocessing yang sama seperti notebook pelatihan, lalu menampilkan
                prediksi teratas dari model DenseNet.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Pilih gambar",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

with right_column:
    st.markdown(
        """
        <div class="info-card">
            <h3>Catatan</h3>
            <p class="muted">
                Model yang digunakan adalah <strong>densenet_kopi_best.pth</strong>.
                Jadi dashboard ini dipakai untuk inferensi pada gambar baru, bukan melatih ulang model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card" style="margin-top: 1rem;">
            <h3>Alur kerja</h3>
            <p class="muted">
                1. Upload gambar daun kopi.<br/>
                2. Model membaca gambar dan menghitung probabilitas kelas.<br/>
                3. Hasil prediksi ditampilkan sebagai label teratas dan daftar top-k.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if uploaded_files:
    results = []

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file).convert("RGB")
        predictions = predict(image, model, transform, class_names, device, top_k)
        top_prediction = predictions[0]
        results.append({
            "file": uploaded_file.name,
            "label": top_prediction["label"],
            "score": top_prediction["score"],
        })

        st.markdown(
            f"<div class='result-card'><div class='result-title'>Gambar {index}</div><div class='result-label'>{top_prediction['label']}</div><div class='result-score'>Confidence: {top_prediction['score']:.2%}</div></div>",
            unsafe_allow_html=True,
        )

        preview_column, detail_column = st.columns([1, 1], gap="large")
        with preview_column:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with detail_column:
            if top_prediction["score"] < confidence_threshold:
                st.warning(
                    f"Confidence {top_prediction['score']:.2%} berada di bawah ambang {confidence_threshold:.2%}."
                )
            else:
                st.success(
                    f"Confidence {top_prediction['score']:.2%} melewati ambang {confidence_threshold:.2%}."
                )

            st.markdown("**Top-k prediction**")
            for rank, prediction in enumerate(predictions, start=1):
                st.markdown(
                    f"""
                    <div class="topk-item">
                        <div class="topk-label">{rank}. {prediction['label']}</div>
                        <div class="topk-score">Confidence: {prediction['score']:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(prediction["score"], 1.0))

        st.write("")

    if len(results) > 1:
        st.markdown("### Ringkasan upload")
        st.dataframe(results, use_container_width=True)
else:
    st.info("Silakan upload satu atau beberapa gambar daun kopi robusta untuk melihat hasil prediksi.")

st.markdown(
    "<div class='footer-note'>Dashboard ini mengikuti checkpoint DenseNet yang sudah dilatih sebelumnya. Jika Anda ingin, saya juga bisa tambahkan fitur simpan hasil prediksi ke CSV atau mode batch folder upload.</div>",
    unsafe_allow_html=True,
)
