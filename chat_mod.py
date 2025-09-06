import streamlit as st
from PIL import Image
import easyocr
from transformers import pipeline
import pandas as pd
import numpy as np
import io
import re

st.set_page_config(page_title="Chat Screenshot -> Meaning", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def get_ocr_reader(use_gpu: bool = False):
    return easyocr.Reader(["en"], gpu=use_gpu)  # English only

# ---------------- Classifier ----------------
@st.cache_resource
def get_emotion_classifier():
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=False,
    )

# ---------------- Flirtish Detector ----------------
def detect_flirtish(text: str) -> bool:
    flirt_words = ["😘", "😍", "❤️", "baby", "babe", "cutie", "hot", "sexy",
                   "sweetheart", "miss you", "love you"]
    text_lower = text.lower()
    return any(word in text_lower for word in flirt_words)

# ---------------- Preprocess OCR lines ----------------
def preprocess_ocr_lines(lines):
    out = []
    for l in lines:
        if not l:
            continue
        l = l.strip()
        l = re.sub(r"\d{1,2}:\d{2}\s?(AM|PM|am|pm)?", "", l)
        l = re.sub(r"^\W+", "", l)  # remove leading non-alphanumeric
        if l:
            out.append(l)
    return out

# ---------------- Summary ----------------
def create_summary(predictions):
    if not predictions:
        return "No messages to summarize."

    label_counts = {}
    for p in predictions:
        label = p["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    top_label, top_count = sorted_labels[0]

    label_explanations = {
        "joy": "happy/excited — positive, enthusiastic tone",
        "love": "affectionate/romantic — warm or loving messages",
        "optimism": "optimistic/hopeful — forward-looking or encouraging",
        "anger": "angry/annoyed — upset or frustrated tone",
        "disgust": "disgusted — strong negative reaction",
        "sadness": "sad/upset — disappointed or hurt tone",
        "fear": "anxious/worried — expressing concern or fear",
        "surprise": "surprised — unexpected reaction",
        "neutral": "neutral/informative — straightforward, non-emotional",
        "embarrassment": "embarrassed — awkward or apologetic tone",
        "confusion": "confused — unclear or seeking clarification",
        "flirtish": "romantic/flirty — playful or intimate tone 😉",
    }

    top_expl = label_explanations.get(top_label.lower(), f"{top_label} (tone)")

    summary_lines = []
    summary_lines.append(f"Overall dominant tone: **{top_label}** ({top_count} message(s)).")
    summary_lines.append(f"Suggested interpretation: *{top_expl}*.")

    if len(sorted_labels) > 1:
        others = ", ".join([f"{lbl}({cnt})" for lbl, cnt in sorted_labels[1:4]])
        summary_lines.append(f"Other tones observed: {others}.")

    advice_map = {
        "joy": "Reply in kind — positive engagement (emojis, plans, appreciation).",
        "love": "Be warm and reciprocate if appropriate.",
        "anger": "Acknowledge feelings, don't escalate. Short apology or clarification may help.",
        "sadness": "Show empathy, ask if they're okay, offer support.",
        "confusion": "Clarify and ask what they mean.",
        "neutral": "Respond as usual with info or logistics.",
        "flirtish": "Play along if you like them 😉, or keep it light if unsure.",
    }
    advice = advice_map.get(top_label.lower(), "Use a calm, clarifying reply.")
    summary_lines.append(f"Quick suggestion: {advice}")

    return "\n\n".join(summary_lines)

# ---------------- Streamlit UI ----------------
st.title("📸 Chat Screenshot → What is she trying to say?")
st.markdown(
    "Upload a chat screenshot (WhatsApp / Telegram / SMS). The app will extract text, classify each message's emotion (English only), detect flirty tone, and give a short summary."
)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload chat screenshot (PNG / JPG)", type=["png", "jpg", "jpeg"])
    use_gpu = st.checkbox("Use GPU for OCR (if available)", value=False)
    run_ocr_btn = st.button("Extract & Analyze")

with col2:
    st.write("Model & tools used:")
    st.write("- OCR: EasyOCR (English only)")
    st.write("- Classifier: bhadresh-savani/distilbert-base-uncased-emotion")
    st.write("- App: Streamlit")

if uploaded_file is None:
    st.info("Upload a screenshot to get started.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded screenshot", use_column_width=True)

    # ---------------- OCR Step ----------------
    if run_ocr_btn:
        with st.spinner("Running OCR..."):
            reader = get_ocr_reader(use_gpu=use_gpu)
            img_np = np.array(image)
            ocr_results = reader.readtext(img_np, detail=0, paragraph=False)
            extracted_lines = preprocess_ocr_lines(ocr_results)
            st.session_state["extracted_lines"] = extracted_lines

    # ---------------- Editable DataFrame ----------------
    extracted_lines = st.session_state.get("extracted_lines", [])
    if extracted_lines:
        editable_df = pd.DataFrame({"text": extracted_lines})
        edited = st.data_editor(editable_df, num_rows="dynamic")

        # ---------------- Classification ----------------
        if st.button("Classify messages") and extracted_lines:
            with st.spinner("Classifying..."):
                classifier = get_emotion_classifier()

                preds = []
                for txt in edited["text"].tolist():
                    if not str(txt).strip():
                        continue

                    # Classification
                    try:
                        res = classifier(str(txt))
                        if isinstance(res, list):
                            r = res[0]
                        else:
                            r = res
                        label = r.get("label")
                        score = float(r.get("score", 0.0))

                        # Flirtish override
                        if detect_flirtish(txt):
                            label = "flirtish"
                            score = 0.99
                    except Exception:
                        label = "error"
                        score = 0.0

                    preds.append({
                        "text": txt,
                        "translated": txt,
                        "label": label,
                        "score": round(score, 3)
                    })

            preds_df = pd.DataFrame(preds)

            st.markdown("### Classified messages")
            st.dataframe(preds_df)

            summary = create_summary(preds)
            st.markdown("### Short summary / interpretation")
            st.markdown(summary)

            csv_buf = io.StringIO()
            preds_df.to_csv(csv_buf, index=False)
            csv_contents = csv_buf.getvalue().encode("utf-8")
            st.download_button("Download CSV", csv_contents, file_name="chat_predictions.csv", mime="text/csv")

            st.markdown("### Full extracted text")
            st.text("\n".join(edited["text"].tolist()))
    else:
        st.info("Run OCR first to see extracted messages.")

st.markdown("---")
st.markdown("**Notes:**\n"
            "- OCR may fail on blurry images.\n"
            "- Emotion classifier is open-source; fine-tuning on your data improves accuracy.\n"
            "- Runs locally — your images/text stay private.")
st.markdown("---")
st.markdown("Made by Vaibhav singh ❤️")
