"""BeCompliant — AGCO beverage cooler compliance auditor.

Mobile-first Streamlit app. Uses the phone camera via st.camera_input,
calls a Roboflow workflow to detect and classify beverage cans, then
computes the Ontario-craft share locally (the workflow's own percentage
is unreliable) and renders a clean verdict for store owners.

Workflow: eigentiki / beverage-cooler-compliance-audit-v2
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page configuration — mobile-first
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BeCompliant — AGCO Cooler Auditor",
    page_icon="🍻",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Aesthetic: dark metallic + purple liquid glass
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, rgba(155, 120, 255, 0.18), transparent 60%),
          radial-gradient(900px 500px at 110% 10%, rgba(90, 60, 200, 0.20), transparent 60%),
          linear-gradient(160deg, #0b0b14 0%, #11111d 50%, #0a0a12 100%);
        color: #ececf1;
      }
      .block-container { padding-top: 1.4rem; padding-bottom: 4rem; max-width: 720px; }
      h1 { font-size: 2rem; margin-bottom: 0.2rem; }
      h1, h2, h3 { color: #f3f0ff; letter-spacing: -0.01em; }

      .glass {
        background: rgba(40, 30, 70, 0.38);
        border: 1px solid rgba(180, 150, 255, 0.18);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        backdrop-filter: blur(14px) saturate(140%);
        -webkit-backdrop-filter: blur(14px) saturate(140%);
        box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
      }

      .pill {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em;
        border: 1px solid rgba(255,255,255,0.12);
      }
      .pill-ok  { background: rgba(80, 200, 140, 0.18); color: #8ef2b8; border-color: rgba(80,200,140,0.4); }
      .pill-bad { background: rgba(240, 90, 110, 0.18); color: #ff9aa8; border-color: rgba(240,90,110,0.4); }

      .metric-row { display: flex; gap: 10px; margin-top: 12px; }
      .metric {
        flex: 1; padding: 12px 14px; border-radius: 12px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06); text-align: center;
      }
      .metric .label { font-size: 0.7rem; color: #b7b2d6; text-transform: uppercase; letter-spacing: 0.1em; }
      .metric .value { font-size: 1.5rem; font-weight: 700; color: #f3f0ff; margin-top: 2px; }

      [data-testid="stFileUploader"] section,
      [data-testid="stCameraInput"] > div {
        background: rgba(40, 30, 70, 0.25);
        border: 1.5px dashed rgba(180, 150, 255, 0.35);
        border-radius: 16px;
      }

      .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #7a4bff 0%, #a77bff 100%);
        color: white; border: 0; border-radius: 12px;
        padding: 0.7rem 1.2rem; font-weight: 600; width: 100%;
        box-shadow: 0 8px 24px rgba(122,75,255,0.35);
      }
      .stButton > button:hover { filter: brightness(1.08); }

      /* Tab styling */
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04); border-radius: 10px;
        padding: 8px 16px; color: #b7b2d6;
      }
      .stTabs [aria-selected="true"] {
        background: rgba(122,75,255,0.25) !important; color: #f3f0ff !important;
      }

      .footnote { color: #8b87a8; font-size: 0.78rem; margin-top: 1.5rem; text-align: center; }

      @media (max-width: 600px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        h1 { font-size: 1.6rem; }
        .metric .value { font-size: 1.2rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Config — prefer st.secrets (Streamlit Cloud) then env vars (.env / host)
# ---------------------------------------------------------------------------
THRESHOLD = 20.0


def _cfg(name: str, default: str | None = None) -> str | None:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


@st.cache_resource
def get_client() -> InferenceHTTPClient:
    api_key = _cfg("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("ROBOFLOW_API_KEY is not set. Configure it in .env or Streamlit secrets.")
        st.stop()
    return InferenceHTTPClient(
        api_url=_cfg("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
        api_key=api_key,
    )


def run_workflow(image_bytes: bytes) -> dict[str, Any]:
    client = get_client()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        raw = client.run_workflow(
            workspace_name=_cfg("ROBOFLOW_WORKSPACE", "eigentiki"),
            workflow_id=_cfg(
                "ROBOFLOW_WORKFLOW_ID",
                "beverage-cooler-compliance-audit-v2-1776699170107",
            ),
            images={"image": tmp_path},
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return raw[0] if isinstance(raw, list) and raw else raw


def pick(result: dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in result and result[k] is not None:
            return result[k]
    return default


def decode_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value") or value.get("image") or value.get("base64")
    if not isinstance(value, str):
        return None
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return Image.open(io.BytesIO(base64.b64decode(payload)))
    except Exception:
        return None


def compress_for_upload(image_bytes: bytes, max_dim: int = 1600, quality: int = 85) -> bytes:
    """Shrink large phone photos before shipping to the Roboflow API.

    Phone cameras emit 10+ MP JPEGs. The model doesn't need that; compressing
    cuts upload time dramatically on cellular and avoids serverless timeouts.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.markdown("# 🍻 BeCompliant")
st.markdown(
    '<div class="glass">'
    "<b>AGCO cooler auditor for Ontario convenience stores.</b><br>"
    f"Scan your cooler — we verify that at least <b>{THRESHOLD:.0f}%</b> of the "
    "display is Ontario craft, the AGCO minimum."
    "</div>",
    unsafe_allow_html=True,
)

st.write("")

tab_camera, tab_upload = st.tabs(["📷 Camera", "📁 Upload"])

image_bytes: bytes | None = None
with tab_camera:
    cam = st.camera_input(
        "Point at the cooler and snap",
        label_visibility="collapsed",
        key="camera",
    )
    if cam is not None:
        image_bytes = cam.getvalue()

with tab_upload:
    up = st.file_uploader(
        "Upload a cooler shelf photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="uploader",
    )
    if up is not None and image_bytes is None:
        image_bytes = up.getvalue()

if image_bytes is None:
    st.markdown(
        '<p class="footnote">Stand back far enough to capture the whole cooler door in one frame.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

if not st.button("🔍 Run compliance audit", type="primary"):
    st.image(image_bytes, caption="Ready to audit", use_column_width=True)
    st.stop()

with st.spinner("Auditing shelf compliance…"):
    try:
        prepared = compress_for_upload(image_bytes)
        result = run_workflow(prepared)
    except Exception as exc:
        st.error(f"Workflow call failed: {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Parse response — recompute percentage locally (workflow math is unreliable)
# ---------------------------------------------------------------------------
craft_count = pick(result, "craft_count", default=0) or 0
macro_count = pick(result, "macro_count", default=0) or 0
total_count = pick(result, "total_count", "total", default=None)
if total_count is None:
    total_count = craft_count + macro_count

try:
    craft_pct = round((craft_count / total_count) * 100, 1) if total_count else 0.0
except Exception:
    craft_pct = 0.0

is_compliant = craft_pct >= THRESHOLD
pill_class = "pill-ok" if is_compliant else "pill-bad"
pill_text = "COMPLIANT" if is_compliant else "NON-COMPLIANT"

annotated = decode_image(
    pick(result, "output_image", "annotated_image", "visualization", "image")
)

st.markdown(
    f"""
    <div class="glass" style="margin-top: 1rem;">
      <span class="pill {pill_class}">{pill_text}</span>
      <h2 style="margin: 8px 0 0 0;">Ontario craft share: {craft_pct:.1f}%</h2>
      <div style="color:#b7b2d6; margin-top:4px;">Threshold: {THRESHOLD:.0f}% minimum</div>
      <div class="metric-row">
        <div class="metric"><div class="label">Craft</div><div class="value">{craft_count}</div></div>
        <div class="metric"><div class="label">Macro</div><div class="value">{macro_count}</div></div>
        <div class="metric"><div class="label">Total</div><div class="value">{total_count}</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_compliant and total_count > 0:
    needed = max(0, int(((THRESHOLD / 100.0) * total_count) - craft_count) + 1)
    st.warning(
        f"Swap approximately **{needed}** macro can(s) for Ontario craft to reach the {THRESHOLD:.0f}% minimum."
    )

if annotated is not None:
    st.image(annotated, caption="Model detections", use_column_width=True)

if st.button("📸 Scan another cooler"):
    st.rerun()

with st.expander("Raw workflow response (debug)"):
    safe = {
        k: (f"<{len(v)}-char blob>" if isinstance(v, str) and len(v) > 400 else v)
        for k, v in result.items()
    }
    st.json(safe)

st.markdown(
    '<p class="footnote">BeCompliant · Unofficial · Always verify with AGCO for final compliance.</p>',
    unsafe_allow_html=True,
)
