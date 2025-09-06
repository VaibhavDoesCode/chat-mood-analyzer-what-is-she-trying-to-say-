# chat-mood-analyzer-what-is-she-trying-to-say

📸 Streamlit app that extracts chat messages from screenshots using OCR, classifies emotions with HuggingFace, and generates a quick summary with reply suggestions.
## 🚀 Overview
This project is a **Streamlit-based NLP + OCR application** that analyzes **chat screenshots** (WhatsApp / Telegram / SMS).  
It extracts text from screenshots, classifies the **emotional tone** of each message using a HuggingFace transformer, and provides a **short summary with suggestions**.

---

## ⚡ Features
- Upload **chat screenshots** (PNG / JPG).
- Extract text using **EasyOCR**.
- Edit extracted messages before analysis.
- Classify emotions with HuggingFace (`distilbert-base-uncased-emotion`).
- Summarize conversation with **dominant tone** + **reply suggestion**.
- Export results as **CSV**.

---

## 🛠️ Tech Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **OCR**: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- **NLP/Classification**: [HuggingFace Transformers](https://huggingface.co/)
- **Data Handling**: Pandas, NumPy
- **Deployment**: Streamlit Cloud / Render

---

## 📂 Workflow
1. **Upload Screenshot** → `st.file_uploader`
2. **OCR Extraction** → EasyOCR → text lines
3. **Preprocessing** → Regex cleanup
4. **Editable Messages** → `st.data_editor`
5. **Emotion Classification** → HuggingFace model
6. **Summary** → Dominant tone + suggestions
7. **Download CSV** → Predictions exported

---

## 📷 Demo
*(Add a sample screenshot here if available)*

---

## ⚙️ Installation & Usage

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/chat-screenshot-analyzer.git
cd chat-screenshot-analyzer
