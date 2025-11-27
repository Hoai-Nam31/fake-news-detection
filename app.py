import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit.components.v1 as components

# ====== THÊM CAPTUM (Integrated Gradients) ======
try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    raise ImportError(
        "Bạn cần cài captum trước:\n"
        "    pip install captum"
    )

# ============================================================
# 1. CẤU HÌNH TRANG
# ============================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="centered"
)

st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = r"D:\University\year_4\Semester_1\Natural Language Processing\my_fakenews_app\distilbert_final"
MAX_LENGTH = 128


# ============================================================
# 2. LOAD MODEL (CPU)
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# ============================================================
# 3. HÀM DỰ ĐOÁN
# ============================================================
def predict_proba(text: str):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs, enc


# ============================================================
# 4. GIẢI THÍCH BẰNG INTEGRATED GRADIENTS (IG)
# ============================================================
def explain_with_ig(text: str):
    # Xác suất gốc
    base_probs, enc = predict_proba(text)

    input_ids = enc["input_ids"]        # (1, L)
    attention_mask = enc["attention_mask"]

    # Hàm forward trả về xác suất lớp Fake News (index 1)
    def forward_func(input_ids_, attention_mask_):
        outputs = model(input_ids=input_ids_, attention_mask=attention_mask_)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        # trả về batch vector [batch_size], mỗi phần tử là prob của class 1
        return probs[:, 1]

    # LayerIntegratedGradients trên embedding layer
    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())

    # Baseline: toàn bộ là token [PAD] (nếu không có thì dùng CLS)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0

    baseline_ids = torch.full_like(input_ids, pad_id)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=50,
        return_convergence_delta=True
    )

    # Gộp attribution trên chiều embedding → 1 giá trị / token
    attributions = attributions.sum(dim=-1).squeeze(0)  # (L,)
    attributions = attributions.detach().cpu().numpy().tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, attributions, base_probs


def build_html_explanation(tokens, importances):
    # Bỏ phần padding phía sau
    cleaned_tokens = []
    cleaned_importances = []
    for tok, imp in zip(tokens, importances):
        if tok == tokenizer.pad_token:
            break
        cleaned_tokens.append(tok)
        cleaned_importances.append(imp)

    tokens = cleaned_tokens
    importances = cleaned_importances

    if not importances:
        return "<p>Không tạo được giải thích.</p>"

    # Chuẩn hóa theo trị tuyệt đối lớn nhất
    max_abs = max(abs(x) for x in importances) or 1e-6

    spans = []
    for tok, imp in zip(tokens, importances):
        # Bỏ luôn [CLS], [SEP], [PAD] nếu còn
        if tok in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        strength = min(1.0, max(0.15, abs(imp) / max_abs))

        if imp > 0:
            # Đẩy về Fake News → đỏ
            color = f"rgba(255, 0, 0, {strength:.2f})"
        elif imp < 0:
            # Đẩy về Real News → xanh
            color = f"rgba(0, 120, 255, {strength:.2f})"
        else:
            color = "rgba(0,0,0,0)"

        display_tok = tok
        if tok.startswith("##"):
            display_tok = tok[2:]
            space = ""
        else:
            space = " "

        span = (
            f"<span style='background-color:{color}; "
            f"padding:2px 3px; border-radius:3px; margin:1px;'>{display_tok}</span>{space}"
        )
        spans.append(span)

    html = f"""
    <div style="font-family: monospace; line-height: 1.8; font-size: 14px;">
        {''.join(spans)}
    </div>
    """

    return html


# ============================================================
# 5. GIAO DIỆN
# ============================================================
st.title("🕵️ Phát hiện Tin Giả & Giải Thích")
st.markdown("---")
st.write("Nhập nội dung cần kiểm tra bằng **Tiếng Anh!!!**")

input_text = st.text_area(
    "Nội dung tin tức",
    height=150,
    placeholder="Paste your text here..."
)


# ============================================================
# 6. BUTTON PHÂN TÍCH
# ============================================================
if st.button("🔍 Phân tích & Giải thích", type="primary"):

    if not input_text.strip():
        st.warning("⚠ Vui lòng nhập nội dung!")
        st.stop()

    # 6.1. Dự đoán
    with st.spinner("🔎 AI đang đọc hiểu văn bản..."):
        probs, _ = predict_proba(input_text)
        real_score, fake_score = probs
        pred_idx = int(np.argmax(probs))

    st.markdown("### 1. Kết quả Dự đoán")
    col1, col2 = st.columns(2)
    col1.metric("Real News", f"{real_score:.1%}")
    col2.metric("Fake News", f"{fake_score:.1%}")

    if pred_idx == 1:
        st.error(f"🟥 Kết luận: **FAKE NEWS** ({fake_score:.1%})")
    else:
        st.success(f"🟩 Kết luận: **REAL NEWS** ({real_score:.1%})")

    # 6.2. Giải thích bằng IG
    st.markdown("---")
    st.markdown("### 2. Tại sao mô hình dự đoán như vậy?")
    st.info("🔴 Đỏ = Từ làm tăng xác suất Fake News — 🔵 Xanh = Từ kéo về Real News")

    with st.spinner("🧠 Đang tính mức độ ảnh hưởng của từng từ (Integrated Gradients)..."):
        try:
            tokens, importances, _ = explain_with_ig(input_text)
            html = build_html_explanation(tokens, importances)
            components.html(html, height=400, scrolling=True)
        except Exception as e:
            st.error(f"❌ Không thể tạo giải thích: {e}")


# ============================================================
# 7. FOOTER
# ============================================================
st.markdown("---")
st.caption("✨ Powered by DistilBERT + Integrated Gradients (Captum) + Streamlit")