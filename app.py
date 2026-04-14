"""
Streamlit Demo App — Real Estate Watermark Removal
Drag-and-drop upload with before/after comparison.
Run: streamlit run app.py
"""

import io
import time
import zipfile
from pathlib import Path
from typing import List

import requests
import streamlit as st
from PIL import Image

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PropClear — AI Watermark Removal",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0e0e11;
    color: #e8e6e1;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #2a2a4a;
}
.hero h1 { font-size: 2.8rem; margin: 0; color: #f0ebe3; }
.hero p { font-size: 1.05rem; color: #9e9db5; margin: 0.5rem 0 0; }

.badge {
    display: inline-block;
    background: #e8c96d22;
    color: #e8c96d;
    border: 1px solid #e8c96d44;
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
    letter-spacing: 0.03em;
}

.stat-card {
    background: #16161e;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.stat-card .number { font-size: 2rem; font-weight: 700; color: #e8c96d; }
.stat-card .label { font-size: 0.8rem; color: #7a7a8e; margin-top: 4px; }

.result-box {
    background: #13131b;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1rem;
}

div[data-testid="stSidebar"] {
    background: #0c0c12 !important;
    border-right: 1px solid #1e1e2e;
}
.stButton > button {
    background: linear-gradient(135deg, #e8c96d, #d4a843);
    color: #0e0e11;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover { filter: brightness(1.1); transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    sharpen = st.toggle("Sharpen output", value=True)
    color_correct = st.toggle("Color correction", value=True)
    manual_mask = st.toggle("Force center mask (skip OCR)", value=False)
    show_mask = st.toggle("Show detected mask", value=False)

    st.divider()
    st.markdown("## 📡 Backend Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        h = r.json()
        st.success(f"✅ API Connected")
        st.code(f"Model:   {h.get('inpainter','?')}\nOCR:     {h.get('ocr_backend','?')}", language="")
    except Exception:
        st.error("❌ API Offline\nRun: `uvicorn main:app`")

    st.divider()
    st.markdown("""
**How it works:**
1. OCR detects watermark text region
2. Binary mask generated + dilated
3. LaMa inpainting fills region
4. Post-processing for quality
    """)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="badge">AI-Powered</span>
  <span class="badge">LaMa Inpainting</span>
  <span class="badge">Real Estate</span>
  <h1>PropClear</h1>
  <p>Remove watermarks from property images with 90–98% reconstruction quality — no training data needed.</p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🖼  Single Image", "📦 Batch Processing"])


# ── Single Image Tab ──────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Drop a property image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible"
    )

    if uploaded:
        original = Image.open(uploaded).convert("RGB")
        col_a, col_b = st.columns(2, gap="medium")

        with col_a:
            st.markdown("**Original (with watermark)**")
            st.image(original, use_container_width=True)

        if st.button("🪄 Remove Watermark", key="single"):
            with st.spinner("Detecting watermark and inpainting…"):
                t0 = time.time()

                uploaded.seek(0)
                params = {
                    "sharpen": sharpen,
                    "color_correct": color_correct,
                    "manual_mask_center": manual_mask
                }
                resp = requests.post(
                    f"{API_URL}/remove-watermark",
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                    params=params,
                    timeout=120
                )

            if resp.status_code == 200:
                elapsed = time.time() - t0
                result_img = Image.open(io.BytesIO(resp.content))

                with col_b:
                    st.markdown("**Result (watermark removed)**")
                    st.image(result_img, use_container_width=True)

                # Stats
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class="stat-card"><div class="number">{elapsed:.1f}s</div><div class="label">Processing Time</div></div>""", unsafe_allow_html=True)
                with c2:
                    w, h = result_img.size
                    st.markdown(f"""<div class="stat-card"><div class="number">{w}×{h}</div><div class="label">Output Resolution</div></div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="stat-card"><div class="number">LaMa</div><div class="label">Inpainting Model</div></div>""", unsafe_allow_html=True)

                # Download
                st.download_button(
                    "⬇️ Download Cleaned Image",
                    data=resp.content,
                    file_name=f"clean_{uploaded.name.rsplit('.', 1)[0]}.png",
                    mime="image/png",
                    use_container_width=True
                )

                # Optional mask preview
                if show_mask:
                    uploaded.seek(0)
                    mask_resp = requests.post(
                        f"{API_URL}/remove-watermark/mask-preview",
                        files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                        timeout=60
                    )
                    if mask_resp.status_code == 200:
                        st.markdown("**Detected Watermark Mask**")
                        mask_img = Image.open(io.BytesIO(mask_resp.content))
                        st.image(mask_img, use_container_width=True, clamp=True)
            else:
                st.error(f"API Error {resp.status_code}: {resp.text[:300]}")


# ── Batch Tab ─────────────────────────────────────────────────────────────────
with tab2:
    batch_files = st.file_uploader(
        "Upload multiple property images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if batch_files and st.button("🚀 Process Batch", key="batch"):
        files_data = [(f.name, f.read(), f.type) for f in batch_files]

        with st.spinner(f"Submitting {len(files_data)} images…"):
            resp = requests.post(
                f"{API_URL}/remove-watermark/batch",
                files=[("files", (name, data, mime)) for name, data, mime in files_data],
                params={"sharpen": sharpen, "color_correct": color_correct},
                timeout=30
            )

        if resp.status_code == 200:
            job = resp.json()
            job_id = job["job_id"]
            total = job["total"]
            st.info(f"Job ID: `{job_id}` — {total} images queued.")

            progress = st.progress(0, text="Processing…")
            status_placeholder = st.empty()

            while True:
                time.sleep(2)
                sr = requests.get(f"{API_URL}/batch-status/{job_id}", timeout=10).json()
                done = sr["done"]
                progress.progress(done / total, text=f"{done}/{total} complete")
                if sr["status"] == "done":
                    break

            st.success("✅ All images processed!")

            # Collect results into ZIP
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, (name, _, _) in enumerate(files_data):
                    r = requests.get(f"{API_URL}/batch-result/{job_id}/{i}", timeout=30)
                    if r.status_code == 200 and r.content:
                        stem = Path(name).stem
                        zf.writestr(f"clean_{stem}.png", r.content)

            st.download_button(
                "⬇️ Download All Results (ZIP)",
                data=zip_buf.getvalue(),
                file_name="cleaned_images.zip",
                mime="application/zip",
                use_container_width=True
            )

            # Preview grid (first 6)
            st.markdown("### Preview (first 6 results)")
            cols = st.columns(3)
            for i, (name, _, _) in enumerate(files_data[:6]):
                r = requests.get(f"{API_URL}/batch-result/{job_id}/{i}", timeout=30)
                if r.status_code == 200 and r.content:
                    with cols[i % 3]:
                        st.image(Image.open(io.BytesIO(r.content)), caption=f"clean_{name}", use_container_width=True)
        else:
            st.error(f"Batch submission failed: {resp.status_code}")