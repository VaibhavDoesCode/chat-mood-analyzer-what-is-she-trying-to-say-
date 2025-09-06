import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px # Using Plotly for interactive and engaging charts

st.set_page_config(page_title="Chat Screenshot -> Meaning", layout="wide", initial_sidebar_state="expanded")

# ---------------- OCR ----------------
@st.cache_resource
def get_ocr_reader(use_gpu: bool = False):
    """Initializes and caches the EasyOCR reader."""
    return easyocr.Reader(["en"], gpu=use_gpu)  # English only

# ---------------- Classifier ----------------
@st.cache_resource
def get_emotion_classifier():
    """Initializes and caches the Hugging Face emotion classifier pipeline."""
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=False,
    )

# ---------------- Flirtish Detector ----------------
def detect_flirtish(text: str) -> bool:
    """Detects if a message contains flirtatious keywords or emojis."""
    flirt_words = ["😘", "😍", "❤️", "baby", "babe", "cutie", "hot", "sexy",
                   "sweetheart", "miss you", "love you", "xoxo", "honey", "darling"]
    text_lower = text.lower()
    return any(word in text_lower for word in flirt_words)

# ---------------- Preprocess OCR lines ----------------
def preprocess_ocr_lines(lines):
    """Cleans and preprocesses OCR extracted lines."""
    out = []
    for l in lines:
        if not l:
            continue
        l = l.strip()
        # Remove common chat timestamps (e.g., "10:30 AM", "1:05pm")
        l = re.sub(r"\d{1,2}:\d{2}\s?(AM|PM|am|pm)?", "", l)
        # Remove leading non-alphanumeric characters (e.g., bullet points, special symbols)
        l = re.sub(r"^\W+", "", l)
        # Remove common chat metadata like "You" or contact names if they appear at the start
        l = re.sub(r"^(You|Me|[\w\s]+?):\s*", "", l, flags=re.IGNORECASE)
        if l:
            out.append(l)
    return out

# ---------------- Summary ----------------
def create_summary(predictions):
    """
    Generates a summary of the chat's emotional tone and provides advice.
    Returns the summary string and a dictionary of label counts.
    """
    if not predictions:
        return "No messages to summarize.", {}

    label_counts = {}
    for p in predictions:
        label = p["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    # Sort labels by count for summary and chart
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    top_label, top_count = sorted_labels[0] if sorted_labels else ("neutral", 0)

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
    summary_lines.append(f"### Overall Chat Tone Analysis")
    summary_lines.append(f"The dominant tone detected is: **{top_label.upper()}** (found in {top_count} message(s)).")
    summary_lines.append(f"This suggests an interpretation of: *{top_expl}*.")

    if len(sorted_labels) > 1:
        other_tones = ", ".join([f"{lbl} ({cnt})" for lbl, cnt in sorted_labels[1:4]])
        summary_lines.append(f"Other significant tones observed include: {other_tones}.")

    advice_map = {
        "joy": "Reply in kind — positive engagement (emojis, plans, appreciation).",
        "love": "Be warm and reciprocate if appropriate.",
        "anger": "Acknowledge feelings, don't escalate. A short apology or clarification may help.",
        "sadness": "Show empathy, ask if they're okay, offer support.",
        "confusion": "Clarify and ask what they mean.",
        "neutral": "Respond as usual with info or logistics.",
        "flirtish": "Play along if you like them 😉, or keep it light if unsure.",
        "disgust": "Address the source of disgust carefully, or change the topic.",
        "fear": "Reassure them and offer support.",
        "surprise": "Acknowledge their surprise and ask for more context if needed.",
        "embarrassment": "Offer reassurance or change the subject to ease their discomfort.",
        "optimism": "Encourage their positive outlook and share in their hope."
    }
    advice = advice_map.get(top_label.lower(), "Consider a calm, clarifying reply based on context.")
    summary_lines.append(f"\n**Quick Suggestion for your reply:** {advice}")

    return "\n\n".join(summary_lines), label_counts

# ---------------- Streamlit UI ----------------
st.title("📸 Chat Screenshot → What are they trying to say?")
st.markdown(
    "Upload a chat screenshot (WhatsApp / Telegram / SMS). The app will extract text, classify each message's emotion (English only), detect flirty tone, and give a short summary."
)

# Sidebar for settings and info
with st.sidebar:
    st.header("Settings & Info")
    st.markdown("---")
    st.subheader("Upload Options")
    uploaded_file = st.file_uploader("Upload chat screenshot (PNG / JPG)", type=["png", "jpg", "jpeg"])
    use_gpu = st.checkbox("Use GPU for OCR (if available)", value=False, help="Check this if you have a compatible GPU for faster OCR.")
    run_ocr_btn = st.button("Extract & Analyze Screenshot", type="primary")

    st.markdown("---")
    st.subheader("About the App")
    st.info(
        """
        This app uses:
        - **OCR:** EasyOCR (English only) for text extraction.
        - **Emotion Classifier:** `bhadresh-savani/distilbert-base-uncased-emotion` from Hugging Face.
        - **Flirt Detector:** Custom keyword matching.
        - **Framework:** Streamlit for the interactive UI.
        """
    )
    st.markdown("**Notes:**\n"
                "- OCR may struggle with blurry or low-resolution images.\n"
                "- The emotion classifier is a general model; fine-tuning on specific chat data could improve accuracy.\n"
                "- All processing runs locally within your browser session — your images/text stay private.")
    st.markdown("---")
    st.markdown("Made with ❤️ by Vaibhav Singh")


# Main content area
if uploaded_file is None:
    st.info("⬆️ Upload a screenshot from the sidebar to get started!")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded screenshot", use_column_width=True)

    # ---------------- OCR Step ----------------
    if run_ocr_btn:
        with st.spinner("Extracting text with OCR... This might take a moment."):
            reader = get_ocr_reader(use_gpu=use_gpu)
            img_np = np.array(image)
            ocr_results = reader.readtext(img_np, detail=0, paragraph=False)
            extracted_lines = preprocess_ocr_lines(ocr_results)
            st.session_state["extracted_lines"] = extracted_lines
        st.success("Text extraction complete!")

    # ---------------- Editable DataFrame ----------------
    extracted_lines = st.session_state.get("extracted_lines", [])
    if extracted_lines:
        st.subheader("Review & Edit Extracted Messages")
        st.info("You can edit the extracted text below if OCR made mistakes. Add/remove rows as needed.")
        editable_df = pd.DataFrame({"text": extracted_lines})
        edited = st.data_editor(editable_df, num_rows="dynamic", use_container_width=True)

        # ---------------- Classification ----------------
        if st.button("Classify Emotions & Summarize", type="primary"):
            with st.spinner("Analyzing emotions..."):
                classifier = get_emotion_classifier()

                preds = []
                for txt in edited["text"].tolist():
                    if not str(txt).strip(): # Skip empty lines
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

                        # Flirtish override for specific keywords/emojis
                        if detect_flirtish(txt):
                            label = "flirtish"
                            score = 0.99 # High confidence for flirtish if detected
                    except Exception as e:
                        st.warning(f"Could not classify message: '{txt[:50]}...' Error: {e}")
                        label = "error"
                        score = 0.0

                    preds.append({
                        "text": txt,
                        "label": label,
                        "score": round(score, 3)
                    })

            preds_df = pd.DataFrame(preds)

            st.markdown("---")
            st.subheader("Detailed Message Analysis")
            st.dataframe(preds_df, use_container_width=True)

            summary, label_counts = create_summary(preds)
            st.markdown("---")
            st.subheader("Chat Summary & Interpretation")
            st.markdown(summary)

            # --- Engaging Pie Chart ---
            if label_counts:
                st.markdown("---")
                st.subheader("Emotional Breakdown of the Chat")
                chart_data_plotly = pd.DataFrame(label_counts.items(), columns=['Emotion', 'Count'])

                # Sort for consistent pie chart slices
                chart_data_plotly = chart_data_plotly.sort_values(by='Count', ascending=False)

                fig_pie = px.pie(
                    chart_data_plotly,
                    values='Count',
                    names='Emotion',
                    title='Distribution of Emotions in Messages',
                    hole=0.4, # Creates a donut chart
                    color_discrete_sequence=px.colors.qualitative.Pastel, # A nice color palette
                    labels={'Emotion':'Emotion Type', 'Count':'Number of Messages'}, # Custom hover labels
                    hover_data=['Count']
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#000000', width=1)) # Add border to slices
                )
                fig_pie.update_layout(
                    showlegend=True,
                    title_x=0.5, # Center title
                    font=dict(size=14)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            # --- End Engaging Pie Chart ---

            st.markdown("---")
            st.subheader("Download Results")
            csv_buf = io.StringIO()
            preds_df.to_csv(csv_buf, index=False)
            csv_contents = csv_buf.getvalue().encode("utf-8")
            st.download_button(
                "Download Classified Messages as CSV",
                csv_contents,
                file_name="chat_predictions.csv",
                mime="text/csv",
                key="download_csv_btn"
            )

            st.markdown("---")
            st.subheader("Raw Extracted Text")
            st.text_area("Full Text (for reference)", "\n".join(edited["text"].tolist()), height=200, disabled=True)
    else:
        st.info("Click 'Extract & Analyze Screenshot' to process the image first.")



