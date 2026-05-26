import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# ── Page Setup ───────────────────────────────────────────────
st.set_page_config(page_title="NER Engine", page_icon="🏷️", layout="centered")

# ── Colors ──────────────────────────────────────────────────
ENTITY_COLORS = {
    "PER":  {"bg": "#D4EDDA", "border": "#28A745", "label": "Person"},
    "ORG":  {"bg": "#CCE5FF", "border": "#004085", "label": "Organization"},
    "LOC":  {"bg": "#FFF3CD", "border": "#856404", "label": "Location"},
    "DATE": {"bg": "#E8D7FF", "border": "#6F42C1", "label": "Date"},
    "MISC": {"bg": "#F8D7DA", "border": "#721C24", "label": "Misc"},
}
DEFAULT_COLOR = {"bg": "#E2E3E5", "border": "#6C757D", "label": "???"}

# ── Cached Model Loader (Lightweight) ────────────────────────
@st.cache_resource
def load_ner_pipeline():
    # dslim/bert-base-NER is fast (40MB vs 1.3GB)
    return pipeline(
        task="ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

# ── Date Extractor ──────────────────────────────────────────
def extract_dates(text: str):
    patterns = [r'\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}', r'\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b', r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s+\d{4})?\b']
    dates_found = []
    for p in patterns:
        for match in re.finditer(p, text, re.IGNORECASE):
            dates_found.append({"entity_group": "DATE", "word": match.group(), "start": match.start(), "end": match.end(), "score": 1.0})
    return dates_found

# ── HTML Builder ────────────────────────────────────────────
def render_annotated_html(text, entities):
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    for ent in sorted_entities:
        s, e = ent["start"], ent["end"]
        etype = ent["entity_group"]
        col = ENTITY_COLORS.get(etype, DEFAULT_COLOR)
        badge = f'<span style="background:{col["bg"]}; border:1px solid {col["border"]}; padding:2px 5px; border-radius:4px; font-weight:bold;">{text[s:e]} <sub>{col["label"]}</sub></span>'
        text = text[:s] + badge + text[e:]
    return f'<div style="line-height:2.5;">{text}</div>'

# ── UI ──────────────────────────────────────────────────────
st.title("🏷️ Fast NER Engine")
sample_text = "Elon Musk announced that Tesla will hold a major event in Washington D.C. on October 24, 2026."
user_input = st.text_area("Input Text:", value=sample_text, height=100)

if st.button("Extract Entities"):
    with st.spinner("Analyzing..."):
        nlp = load_ner_pipeline()
        entities = nlp(user_input) + extract_dates(user_input)
        
        # Cleanup overlaps
        entities = sorted(entities, key=lambda x: x['start'])
        final = []
        last_end = -1
        for ent in entities:
            if ent['start'] >= last_end:
                final.append(ent)
                last_end = ent['end']
        
        st.markdown(render_annotated_html(user_input, final), unsafe_allow_html=True)
        st.subheader("Summary Table")
        st.table(pd.DataFrame([{"Entity": e["word"], "Label": ENTITY_COLORS.get(e["entity_group"], DEFAULT_COLOR)["label"]} for e in final]))