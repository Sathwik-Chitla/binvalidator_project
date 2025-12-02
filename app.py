# --- FIX FOR STREAMLIT OPENCV PROBLEM ---
import os
import sys

# Prevent Streamlit's auto-installed opencv-python (GUI) from loading
if "cv2" in sys.modules:
    del sys.modules["cv2"]

# Environment variables to help cv2-headless
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Force OpenCV headless to load
try:
    import cv2
except:
    pass
# ----------------------------------------------------

import streamlit as st
from ultralytics import YOLO
import torch
import clip
from PIL import Image
import numpy as np
import pickle

# -----------------------------
# CONFIG / PATHS (edit if needed)
# -----------------------------
YOLO_MODEL_PATH = "best.pt"                 # your trained YOLO model
BIN_IMAGES_DIR = "data/bin-images"          # contains 00001.jpg, ...
CROPS_DIR = "data/crops"                    # will be created if missing
EMBEDDINGS_PATH = "data/asin_clip_embeddings.pkl"  # dict {asin: (512,)}

# Optional similarity threshold (0..1). If you want to ignore very-low-score matches set e.g. 0.25.
SIMILARITY_THRESHOLD = None  # or e.g. 0.25

# -----------------------------
# LOAD MODEL: YOLO + CLIP
# -----------------------------
st.set_page_config(page_title="Amazon Bin Validator", layout="wide")

@st.cache_resource
def load_models():
    # YOLO
    yolo = YOLO(YOLO_MODEL_PATH)

    # CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    return yolo, clip_model, preprocess, device

model, clip_model, preprocess, device = load_models()

# -----------------------------
# Load embeddings dict and convert to matrix
# -----------------------------
with open(EMBEDDINGS_PATH, "rb") as f:
    asin_dict = pickle.load(f)   # expected: dict {asin: vector}

# Build ordered lists/matrix from the dict
asin_list = np.array(list(asin_dict.keys()), dtype=str)  # (N,)
asin_matrix = np.array(list(asin_dict.values()), dtype="float32")  # (N,512)
# Normalize embeddings for cosine similarity
asin_matrix = asin_matrix / np.linalg.norm(asin_matrix, axis=1, keepdims=True)

# -----------------------------
# Utilities
# -----------------------------
def get_bin_image_path(bin_id):
    path = os.path.join(BIN_IMAGES_DIR, f"{bin_id}.jpg")
    return path if os.path.exists(path) else None

def save_crop(img_pil, crop_path):
    os.makedirs(os.path.dirname(crop_path), exist_ok=True)
    img_pil.save(crop_path)

def run_yolo_and_save_crops(image_path):
    """Runs YOLO on image_path and returns list of crop file paths."""
    results = model(image_path)
    os.makedirs(CROPS_DIR, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    crops = []
    # results[0].boxes may be empty
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)
        crop = img.crop((x1, y1, x2, y2))
        crop_path = os.path.join(CROPS_DIR, f"crop_{i}.jpg")
        save_crop(crop, crop_path)
        crops.append(crop_path)

    return crops

def get_clip_embedding_from_path(image_path):
    """Return L2-normalized CLIP embedding vector (shape (512,))."""
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(img_t).cpu().numpy().reshape(-1)
    # guard against zero norm
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

def match_best_asin_for_crop(crop_path):
    """Return (best_asin, similarity_score). Matches across ALL ASINs (per your choice)."""
    emb = get_clip_embedding_from_path(crop_path)  # (512,)
    sims = asin_matrix.dot(emb)  # (N,)
    if sims.size == 0:
        return "UNKNOWN", 0.0
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_asin = str(asin_list[best_idx])
    # optionally apply threshold
    if SIMILARITY_THRESHOLD is not None and best_score < SIMILARITY_THRESHOLD:
        return "UNKNOWN", best_score
    return best_asin, best_score

def compute_detection_summary(detected_asins, user_request):
    """Return dict summary and total count."""
    freq = {}
    for a in detected_asins:
        freq[a] = freq.get(a, 0) + 1

    summary = {}
    for asin, count in freq.items():
        summary[asin] = {
            "detections": int(count),
            "quantity_per_item_requested": int(user_request.get(asin, 1))
        }
    total_detected = sum(freq.values())
    return summary, total_detected

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📦 Amazon Bin Validator")

st.markdown("Enter the ASINs and quantities you expect to find in the bin, then enter a bin ID and run validation.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Requested ASINs (one per line)")
    asin_text = st.text_area("Format: ASIN QTY (e.g. B00NJ008MI 1)", height=180, key="asin_input")
    user_request = {}
    if asin_text and asin_text.strip():
        for ln in asin_text.splitlines():
            parts = ln.strip().split()
            if len(parts) >= 2:
                a = parts[0].strip()
                try:
                    q = int(parts[1])
                except:
                    q = 1
                user_request[a] = q

with col2:
    st.subheader("2) Bin ID")
    bin_id = st.text_input("Bin ID (filename without extension)", value="00001")
    run_button = st.button("Run Validation")

# Run the pipeline
if run_button:
    if not user_request:
        st.warning("No requested ASINs provided. Please enter at least one ASIN and quantity.")
    img_path = get_bin_image_path(bin_id)
    if img_path is None:
        st.error(f"Bin image not found: {bin_id}.jpg in `{BIN_IMAGES_DIR}`")
    else:
        st.image(img_path, caption=f"Bin {bin_id}", use_column_width=True)

        # 1) YOLO detection & crops
        st.info("Running YOLO to detect objects and create crops...")
        crop_paths = run_yolo_and_save_crops(img_path)
        st.success(f"YOLO found {len(crop_paths)} object(s).")

        if len(crop_paths) == 0:
            st.warning("No detections from YOLO. Cannot validate the bin.")
            # Final big FAIL
            st.markdown("---")
            st.markdown("<div style='padding:20px;background:#dc3545;color:white;border-radius:10px;font-size:24px;text-align:center;'>"
                        "❌ FAIL — No items detected in bin"
                        "</div>", unsafe_allow_html=True)
        else:
            st.subheader("Detected crops")
            crop_cols = st.columns(min(4, len(crop_paths)))
            for i, cpath in enumerate(crop_paths):
                with crop_cols[i % len(crop_cols)]:
                    st.image(cpath, caption=os.path.basename(cpath), width=180)

            # 2) CLIP matching (across ALL ASINs)
            st.info("Running CLIP to find best matching ASIN for each crop (searching all ASINs)...")
            detected_asins = []
            details = []
            for cpath in crop_paths:
                best_asin, score = match_best_asin_for_crop(cpath)
                detected_asins.append(best_asin)
                details.append((os.path.basename(cpath), best_asin, score))
                st.write(f"• **{os.path.basename(cpath)}** → **{best_asin}** (score={score:.4f})")

            # 3) Compute detection summary
            summary, total_detected = compute_detection_summary(detected_asins, user_request)
            st.subheader("📦 Quantity Verification")
            st.json(summary)
            st.success(f"📦 FINAL DETECTED ITEM COUNT = {total_detected}")

            # 4) Validation vs user request
            st.subheader("📊 Final Validation Report")
            all_good = True
            # If none of the detected ASINs intersect user requested, we should FAIL (per your choice)
            intersection = set(detected_asins).intersection(set(user_request.keys()))
            if len(intersection) == 0:
                st.error("❌ None of the detected ASINs match the requested ASINs.")
                all_good = False

            for asin, req_qty in user_request.items():
                found_qty = detected_asins.count(asin)
                if found_qty == req_qty:
                    st.success(f"✔ {asin}: Expected {req_qty}, Found {found_qty}")
                else:
                    st.error(f"❌ {asin}: Expected {req_qty}, Found {found_qty}")
                    all_good = False

            extras = [a for a in detected_asins if a not in user_request]
            if extras:
                st.warning(f"⚠ Unexpected items detected (ASINs): {extras}")
                all_good = False

            # 5) Big PASS/FAIL box
            st.markdown("---")
            st.subheader("🏁 FINAL RESULT")
            if all_good:
                st.markdown(
                    "<div style='padding:20px;background:#28a745;color:white;border-radius:10px;font-size:24px;text-align:center;'>"
                    "✅ PASS — Bin Is Correct"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div style='padding:20px;background:#dc3545;color:white;border-radius:10px;font-size:24px;text-align:center;'>"
                    "❌ FAIL — Bin Does NOT Match Order"
                    "</div>",
                    unsafe_allow_html=True
                )

# Optional bottom notes
st.markdown("---")
st.caption("Notes: Matching uses CLIP (ViT-B/32) embeddings produced earlier. Matching searches across all available ASIN embeddings. "
           "If many false positives occur, consider re-generating better product embeddings, adding a similarity threshold, or limiting search to requested ASINs.")
