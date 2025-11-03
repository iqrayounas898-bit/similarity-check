# app.py
import io
import os
import tempfile
import pandas as pd
import numpy as np
# Ensure python-docx is installed for Streamlit app execution
!pip install streamlit
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Similarity Checker", layout="wide")

st.title("📄 Document Similarity Checker (Streamlit)")
st.markdown("Upload multiple `.docx` or `.txt` files. The app computes TF-IDF cosine similarity, shows a heatmap, pairwise table, and lets you download the results.")

uploaded_files = st.file_uploader("Upload files (multiple):", accept_multiple_files=True, type=['docx', 'txt'])

def read_docx_bytes(file_bytes):
    # file_bytes: bytes or BytesIO
    # Save to a temporary file then read with python-docx (which expects a path or file-like)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    try:
        tmp.write(file_bytes)
        tmp.flush()
        tmp.close()
        doc = Document(tmp.name)
        text = "\n".join([p.text for p in doc.paragraphs])
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return text

def read_txt_bytes(file_bytes):
    try:
        return file_bytes.decode('utf-8', errors='replace')
    except:
        return file_bytes.decode('latin-1', errors='replace')

def extract_text_from_uploaded(file):
    # file is UploadedFile (has .name and .read())
    name = file.name
    content = file.read()
    if name.lower().endswith('.docx'):
        return read_docx_bytes(content)
    else:
        return read_txt_bytes(content)

if uploaded_files and len(uploaded_files) >= 1:
    with st.spinner("Reading files and computing similarity..."):
        file_names = [f.name for f in uploaded_files]
        docs = []
        for f in uploaded_files:
            txt = extract_text_from_uploaded(f)
            docs.append(txt if txt.strip() != "" else " ")  # avoid empty doc errors

        # Vectorize
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
        tfidf = vectorizer.fit_transform(docs)

        # Similarity matrix
        sim_matrix = cosine_similarity(tfidf)
        sim_df = pd.DataFrame(sim_matrix, index=file_names, columns=file_names)

    st.success("✅ Computed similarity for {} files".format(len(file_names)))

    # Show top pairs (excluding self)
    def top_pairs(df, top_n=10):
        pairs = []
        names = df.index.tolist()
        n = len(names)
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((names[i], names[j], float(df.iloc[i, j])))
        pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
        return pairs_sorted[:top_n]

    st.subheader("Top similar document pairs")
    tp = top_pairs(sim_df, top_n=10)
    if tp:
        top_df = pd.DataFrame(tp, columns=["Document A", "Document B", "Similarity"])
        top_df['Similarity'] = top_df['Similarity'].round(3)
        st.table(top_df)
    else:
        st.write("Not enough documents to show pairs.")

    # Show heatmap
    st.subheader("Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(sim_df.values, annot=True, fmt=".2f", xticklabels=file_names, yticklabels=file_names, ax=ax, cmap="YlGnBu")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    # Allow downloading heatmap as PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="Download heatmap (PNG)",
        data=buf,
        file_name="similarity_heatmap.png",
        mime="image/png"
    )

    # Show similarity matrix table and allow CSV download
    st.subheader("Full Similarity Matrix (CSV Download)")
    st.dataframe(sim_df.style.format("{:.4f}"), height=300)

    csv = sim_df.to_csv()
    st.download_button("Download similarity CSV", data=csv, file_name="similarity_matrix.csv", mime="text/csv")

    # Simple summaries (first 3 non-empty lines)
    st.subheader("Document previews / simple summaries")
    cols = st.columns(2)
    for i, name in enumerate(file_names):
        preview = docs[i].strip()
        # pick first 3 non-empty lines
        lines = [ln.strip() for ln in preview.splitlines() if ln.strip() != ""]
        summary = "\n".join(lines[:3]) if lines else "(No extractable text)"
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            st.write(summary)

    # Optional: threshold filtering to flag possible copies
    st.subheader("Flag likely duplicates (threshold)")
    threshold = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    flagged = []
    names = sim_df.index.tolist()
    n = len(names)
    for i in range(n):
            for j in range(i+1, n):
                if sim_df.iloc[i, j] >= threshold:
                    flagged.append((names[i], names[j], float(sim_df.iloc[i, j])))
    if flagged:
        flag_df = pd.DataFrame(flagged, columns=["Doc A", "Doc B", "Similarity"])
        flag_df['Similarity'] = flag_df['Similarity'].round(3)
        st.table(flag_df)
        st.warning(f"{len(flag_df)} pair(s) above threshold {threshold}")
    else:
        st.info(f"No pairs with similarity >= {threshold}")

else:
    st.info("Upload at least one `.docx` or `.txt` file to start. Example: student submissions, essays, notes.")

st.markdown("---")
st.caption("Built with Streamlit. TF-IDF + Cosine similarity. If you want plagiarism-style detailed reporting (sentence-level matching, highlighting, or support for PDFs), tell me and I can extend this.")
