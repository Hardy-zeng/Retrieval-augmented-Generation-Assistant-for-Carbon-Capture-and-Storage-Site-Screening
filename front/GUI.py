import os
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
for env_var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']:
    os.environ.pop(env_var, None)
    os.environ.pop(env_var.upper(), None)

import json
import base64 
import pandas as pd
import streamlit as st
import altair as alt
import requests
import torch
import numpy as np
import time
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
PROJECT_ROOT = "/home/hardy/ccs_project"

@st.cache_resource
def load_rag_database():
    try:
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "vectorstore")
        
        vs = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
        return vs, True
    except Exception as e:
        print(f"RAG Database Loading Error: {e}")
        return None, False

vectorstore, RAG_AVAILABLE = load_rag_database()
try:
    from pytorch_tabular import TabularModel
except ImportError:
    st.error("PyTorch Tabular is not installed. Manual live prediction will not work.")
if not hasattr(torch, '_original_load_saved'):
    torch._original_load_saved = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return torch._original_load_saved(*args, **kwargs)
    torch.load = _patched_load
try:
    from pytorch_lightning.callbacks import Callback
except ImportError:
    class Callback: pass 

class LossHistoryTracker(Callback):
    def __init__(self):
        self.history = []
    def on_train_epoch_end(self, trainer, pl_module):
        pass

# Page Configuration
st.set_page_config(
    page_title="RAG ASSISTANCE FOR CCS SITE SCREENING",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

def navigate_to(page):
    st.session_state.current_page = page

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

bg_base64 = get_base64_image("background.jpg")

if st.session_state.current_page == "Home":
    st.markdown(f"""
    <style>
        header[data-testid="stHeader"] {{ display: none !important; }}
        .block-container {{
            padding: 0rem !important; max-width: 100% !important; margin-top: 0px !important;
        }}
        .stApp {{
            background-image: linear-gradient(rgba(15, 23, 42, 0.55), rgba(15, 23, 42, 0.85)), url("data:image/jpeg;base64,{bg_base64}");
            background-size: cover; background-position: center; background-attachment: fixed;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            height: 100vh !important; overflow: hidden !important;
        }}
        @keyframes fadeUp {{
            0% {{ opacity: 0; transform: translateY(30px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        .animate-fade-up {{ animation: fadeUp 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards; }}
        .delay-1 {{ animation-delay: 0.2s; opacity: 0; }}
        div[data-testid="stVerticalBlock"] > div:first-child {{ padding-top: 0 !important; margin-top: 0 !important; }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        header[data-testid="stHeader"] { display: block !important; }
        .block-container { padding: 2rem 3rem !important; max-width: auto !important; }
        .stApp { background-color: #fcfcfd; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 21px;
    }
    .stDataFrame div {
    font-size: 1.25rem !important;
    }
    .stDataFrame thead tr th {
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    }
    .stMarkdown p, .stMarkdown li { font-size: 1.2rem !important; line-height: 1.7 !important; }
    .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label, .stMultiSelect label { font-size: 1.2rem !important; font-weight: 700 !important; color: #1e293b !important; }
    div[data-baseweb="select"], input[type="text"], input[type="number"] { font-size: 1.15rem !important; }
    .stButton button { font-size: 1.25rem !important; font-weight: 700 !important; padding: 12px 24px !important; }
    .stAlert div { font-size: 1.15rem !important; }
    div[role="radiogroup"] label { font-size: 1.25rem !important; font-weight: 700 !important; color: #1e293b !important; }
    .streamlit-expanderHeader { font-size: 1.25rem !important; font-weight: 700 !important; color: #1e293b !important; }
    .formal-footer { text-align: center; margin-top: 60px; padding-top: 20px; border-top: 1px solid #e2e8f0; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 10px; padding-bottom: 30px; }
    .footer-line-1 { font-weight: 700; color: #475569; font-size: 1.2rem; }
    .footer-line-2 { color: #64748b; font-size: 1.15rem; }
    @keyframes pulse-border {
        0% { border-color: rgba(37,99,235,0.3); box-shadow: 0 0 0 0 rgba(37,99,235,0.15); }
        50% { border-color: rgba(37,99,235,0.8); box-shadow: 0 0 0 6px rgba(37,99,235,0); }
        100% { border-color: rgba(37,99,235,0.3); box-shadow: 0 0 0 0 rgba(37,99,235,0); }
    }
    .pulse-card {
        background-color: #f8fbff; border: 2px solid #3b82f6; border-radius: 12px;
        padding: 24px; animation: pulse-border 3s infinite; margin-top: 16px;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0284c7; border-radius: 8px; padding: 20px 26px;
        margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .stVegaLiteChart { margin-top: -10px; }

    .journal-header { padding: 10px 0 30px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 24px; }
    .j-title { font-size: 3.2rem; font-weight: 900; color: #0f172a; letter-spacing: -0.03em; margin-bottom: 16px; line-height: 1.1; }
    .j-desc { font-size: 1.3rem; color: #475569; max-width: 1000px; line-height: 1.6; margin-bottom: 28px; }
    .j-meta-container { display: flex; gap: 48px; }
    .j-meta-block { display: flex; flex-direction: column; gap: 6px; }
    .j-meta-label { color: #94a3b8; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase; font-size: 1.0rem; }
    .j-meta-value { color: #0f172a; font-weight: 800; font-size: 1.3rem; }
    
    .clean-pipeline { display: flex; justify-content: space-between; align-items: center; padding: 0 10px; margin-bottom: 32px; font-size: 1.1rem; }
    .cp-step { font-weight: 700; color: #64748b; text-align: center; }
    .cp-step.active { color: #2563eb; font-weight: 900; }
    .cp-arrow { color: #cbd5e1; font-weight: 300; font-size: 1.5rem; }
    
    .kpi-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .kpi-title { font-size: 1.0rem; font-weight: 800; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 2.2rem; font-weight: 900; color: #0f172a; margin-top: 8px; line-height: 1.1; }
    .kpi-badge-wrap { margin-top: 12px; display: inline-block; }
    
    .panel-header { background: transparent; border-bottom: 2px solid #f1f5f9; padding: 14px 0; margin-bottom: 24px; margin-top: 24px; }
    .panel-title { font-size: 1.5rem; font-weight: 900; color: #0f172a; margin-bottom: 6px; }
    .panel-subtitle { font-size: 1.15rem; color: #64748b; line-height: 1.5; }
    
    .evidence-card { border-left: 5px solid #3b82f6; background: #ffffff; padding: 18px 20px; margin-bottom: 16px; border-radius: 4px 8px 8px 4px; border: 1px solid #e2e8f0; }
    .evidence-source { font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; line-height: 1.4; }
    .evidence-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; gap: 8px; flex-wrap: wrap; }
    .evidence-page { font-size: 1.0rem; color: #475569; background: #f1f5f9; padding: 4px 10px; border-radius: 4px; font-weight: 700; border: 1px solid #e2e8f0; }
    .evidence-tag { font-size: 0.95rem; color: #059669; font-weight: 800; display: flex; align-items: center; gap: 4px;}
    .evidence-snippet { font-size: 1.1rem; color: #475569; font-style: italic; border-top: 1px dashed #e2e8f0; padding-top: 10px; margin-top: 6px; line-height: 1.5; }
    
    .badge { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 6px; font-weight: 800; font-size: 1.1rem; letter-spacing: 0.3px; }
    .badge-high { background-color: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; } 
    .badge-medium-high { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; } 
    .badge-medium { background-color: #fef08a; color: #a16207; border: 1px solid #fde047; } 
    .badge-medium-low { background-color: #ffedd5; color: #c2410c; border: 1px solid #fed7aa; } 
    .badge-low { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; } 
    .badge-unknown { background-color: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; } 
    
    .stTabs [data-baseweb="tab-list"] { display: flex; width: 100%; justify-content: space-between; gap: 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 0; }
    .stTabs [data-baseweb="tab"] { flex: 1; display: flex; justify-content: center; align-items: center; height: 60px; font-weight: 700; font-size: 1.4rem; color: #64748b; background-color: transparent; }
    .stTabs [aria-selected="true"] { color: #1d4ed8 !important; border-bottom: 4px solid #1d4ed8 !important; background-color: #f8fbff !important; }
    
    .factor-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin-top: 16px; }
    .factor-item { background: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
    .factor-label { font-size: 1.0rem; color: #64748b; font-weight: 800; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.5px; }
    .factor-val { font-size: 1.3rem; color: #0f172a; font-weight: 900; line-height: 1.2; }
    
    .manual-input-group { background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
    .manual-group-title { font-size: 1.2rem; font-weight: 900; color: #2563eb; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; border-bottom: 2px dashed #e2e8f0; padding-bottom: 10px; }
    
    [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)

if st.session_state.current_page != "Home":
    nav_home = "#dc2626" if st.session_state.current_page == "Home" else "#64748b"
    nav_basin = "#dc2626" if st.session_state.current_page == "Basin Overview" else "#64748b"
    nav_target = "#dc2626" if st.session_state.current_page == "Target Deep Dive" else "#64748b"
    nav_engine = "#dc2626" if st.session_state.current_page == "Engine Validation" else "#64748b"
    
    st.markdown(f"""
    <style>
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button {{
            background-color: transparent !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            font-weight: 900 !important;
            font-size: 1.4rem !important; 
            padding: 12px 0 !important;
            border-radius: 0px !important;
            transition: color 0.2s ease;
        }}
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button:hover,
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button:focus,
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button:active {{
            background-color: transparent !important; color: #dc2626 !important;
        }}
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button::before,
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type div[data-testid="stButton"] > button::after {{ display: none !important; }}
        
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(1) button {{ color: {nav_home} !important; border-bottom: {f"4px solid {nav_home}" if st.session_state.current_page == "Home" else "4px solid transparent"} !important; }}
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(2) button {{ color: {nav_basin} !important; border-bottom: {f"4px solid {nav_basin}" if st.session_state.current_page == "Basin Overview" else "4px solid transparent"} !important; }}
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(3) button {{ color: {nav_target} !important; border-bottom: {f"4px solid {nav_target}" if st.session_state.current_page == "Target Deep Dive" else "4px solid transparent"} !important; }}
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(4) button {{ color: {nav_engine} !important; border-bottom: {f"4px solid {nav_engine}" if st.session_state.current_page == "Engine Validation" else "4px solid transparent"} !important; }}
    </style>
    """, unsafe_allow_html=True)
    
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
    with col_nav1: st.button("Home", on_click=navigate_to, args=("Home",), use_container_width=True)
    with col_nav2: st.button("Basin Overview", on_click=navigate_to, args=("Basin Overview",), use_container_width=True)
    with col_nav3: st.button("Target Deep Dive", on_click=navigate_to, args=("Target Deep Dive",), use_container_width=True)
    with col_nav4: st.button("Engine Validation", on_click=navigate_to, args=("Engine Validation",), use_container_width=True)
    
    st.markdown("<hr style='margin-top: 0px; margin-bottom: 30px; border-color: rgba(226, 232, 240, 0.8);'>", unsafe_allow_html=True)


REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
FIG_DIR_EXP = os.path.join(PROJECT_ROOT, "figures", "explainability")
FIG_DIR_GEN = os.path.join(PROJECT_ROOT, "figures", "tarim_generalization") 

EXCEL_PATH = os.path.join(REPORT_DIR, "thesis_gradient_summary_table.xlsx")
JSON_PATH = os.path.join(REPORT_DIR, "thesis_gradient_ccs_reports.json") 
FORCE_SUMMARY_PATH = os.path.join(REPORT_DIR, "model_final_summary.xlsx") if os.path.exists(os.path.join(REPORT_DIR, "model_final_summary.xlsx")) else os.path.join(REPORT_DIR, "model_final_summary.csv")
TARIM_SUMMARY_PATH = os.path.join(REPORT_DIR, "tarim_blind_test_results.csv") if os.path.exists(os.path.join(REPORT_DIR, "tarim_blind_test_results.csv")) else os.path.join(REPORT_DIR, "tarim_blind_test_results.xlsx")

SHAP_SUMMARY_PATH = os.path.join(FIG_DIR_EXP, "9b_shap_bar_tarim.png")
SHAP_DEP_PATH = os.path.join(FIG_DIR_EXP, "10_shap_dependence_tarim.png")
SHAP_WATERFALL_PATH = os.path.join(FIG_DIR_EXP, "11a_shap_waterfall_tarim.png")
SHAP_BAR_PATH = os.path.join(FIG_DIR_EXP, "9b_shap_bar_tarim.png")
SHAP_FORCE_PATH = os.path.join(FIG_DIR_EXP, "11b_shap_force_tarim.png")
LEARNING_CURVE_PATH = os.path.join(FIG_DIR_GEN, "8_0_final_learning_curve.png")
CONF_MATRIX_PATH = os.path.join(FIG_DIR_GEN, "8_1_tarim_confusion_matrix.png")
ROC_CURVE_PATH = os.path.join(FIG_DIR_GEN, "8_2_tarim_roc_curve.png")
PR_CURVE_PATH = os.path.join(FIG_DIR_GEN, "8_3_tarim_pr_curve.png")

DEPTH_PLOT_PATH = os.path.join(PROJECT_ROOT, "figures", "13_tarim_depth_plot.png")
if not os.path.exists(DEPTH_PLOT_PATH):
    DEPTH_PLOT_PATH = os.path.join(FIG_DIR_EXP, "13_tarim_depth_plot.png")

VERDICT_ORDER = {"High": 5, "Medium-High": 4, "Medium": 3, "Medium-Low": 2, "Low": 1}
ALTAIR_COLORS = {"High": "#065f46", "Medium-High": "#15803d", "Medium": "#a16207", "Medium-Low": "#c2410c", "Low": "#b91c1c"}

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    try: return pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    except Exception: return pd.DataFrame()

@st.cache_data
def load_json_reports(json_path: str) -> dict:
    if not os.path.exists(json_path): return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
        reports = data.get("gradient_candidates_evaluation", data.get("top_candidates_evaluation", []))
        result = {}
        for item in reports:
            interval_id = item.get("interval_id")
            if interval_id:
                result[interval_id] = {
                    "report_text": item.get("rag_geological_evaluation", ""),
                    "evidence": item.get("retrieved_evidence", [])
                }
        return result
    except Exception: return {}

def render_badge(verdict: str) -> str:
    mapping = {"High": "badge-high", "Medium-High": "badge-medium-high", "Medium": "badge-medium", "Medium-Low": "badge-medium-low", "Low": "badge-low"}
    v = str(verdict).strip()
    return f'<span class="badge {mapping.get(v, "badge-unknown")}">{v}</span>'

def extract_reason(text: str) -> str:
    if not text: return ""
    for line in text.splitlines():
        if line.startswith("Reason:"): return line.replace("Reason:", "").strip()
    return text

def extract_recommendation(text: str) -> str:
    if not text: return ""
    for line in text.splitlines():
        if line.startswith("Recommendation:"): return line.replace("Recommendation:", "").strip()
    return ""

def parse_to_percentage(x):
    if x is None or pd.isna(x): return None
    try:
        val = float(x)
        return val * 100 if val <= 1.0 else val
    except Exception: return None

def get_f1_score(df: pd.DataFrame):
    if df.empty: return None
    for col in df.columns:
        if "f1" in col.lower() or "f1-score" in col.lower():
            try: return float(df[col].iloc[0])
            except Exception: pass
    return None

def short_snippet(text, limit=160) -> str:
    if not text: return ""
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."

def format_docs_with_citations(docs):
    if not docs:
        return "No relevant geological evidence retrieved."

    formatted = []
    for i, doc in enumerate(docs, 1):
        raw_source = doc.metadata.get("source", f"Document {i}") if hasattr(doc, "metadata") else f"Document {i}"
        source = os.path.basename(str(raw_source))
        page = doc.metadata.get("page", "N/A") if hasattr(doc, "metadata") else "N/A"
        content = doc.page_content.strip() if hasattr(doc, "page_content") else str(doc)

        formatted.append(f"[Source: {source}, Page: {page}]\n{content}")

    return "\n\n".join(formatted)

def parse_llm_output(live_result: str) -> dict:
    parsed = {
        "Evidence Type": "Unknown",
        "Caprock Evidence": "Unknown",
        "Structural Risk Evidence": "Unknown",
        "Screening Verdict": "Unknown",
        "Reason": "",
        "Recommendation": ""
    }
    
    text = live_result.replace("**", "")
    for line in text.split('\n'):
        line_c = line.strip()
        if line_c.startswith("Evidence Type:"): parsed["Evidence Type"] = line_c.split(":", 1)[1].strip()
        elif line_c.startswith("Caprock Evidence:"): parsed["Caprock Evidence"] = line_c.split(":", 1)[1].strip()
        elif line_c.startswith("Structural Risk Evidence:"): parsed["Structural Risk Evidence"] = line_c.split(":", 1)[1].strip()
        elif line_c.startswith("Screening Verdict:"): parsed["Screening Verdict"] = line_c.split(":", 1)[1].strip()
    if "Reason:" in text and "Recommendation:" in text:
        parts = text.split("Recommendation:")
        parsed["Recommendation"] = parts[1].strip()
        reason_section = parts[0].split("Reason:")
        if len(reason_section) > 1:
            parsed["Reason"] = reason_section[1].strip()
    elif "Reason:" in text:
        parsed["Reason"] = text.split("Reason:")[1].strip()
    else:
        parsed["Reason"] = text.strip() 
    parsed["Reason"] = parsed["Reason"].replace('\n', '<br>')
    parsed["Recommendation"] = parsed["Recommendation"].replace('\n', '<br>')
    
    return parsed

def get_evidence_weight(evidence_type: str) -> float:
    e = str(evidence_type).strip().lower()

    if "robust" in e:
        return 1.0
    elif "regional" in e or "analog" in e:
        return 0.75
    elif "insufficient" in e:
        return 0.5
    else:
        return 0.6  # fallback

def build_metrics_comparison_df(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    possible_pairs = [("Accuracy (%)", "Accuracy (%)"), ("Precision (%)", "Precision (%)"), ("Recall (%)", "Recall (%)"), ("F1-Score (%)", "F1-Score (%)"), ("ROC-AUC (%)", "ROC-AUC (%)")]
    rows = []
    for s_col, t_col in possible_pairs:
        if s_col in source_df.columns and t_col in target_df.columns:
            metric_name = s_col.replace(" (%)", "")
            try:
                s_val = float(source_df[s_col].iloc[0])
                t_val = float(target_df[t_col].iloc[0])
                rows.append({"Metric": metric_name, "Domain": "Source (North Sea)", "Score": s_val})
                rows.append({"Metric": metric_name, "Domain": "Target (Tarim Basin)", "Score": t_val})
            except Exception: pass
    return pd.DataFrame(rows)

def build_metrics_retention_df(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    possible_pairs = [("Accuracy (%)", "Accuracy (%)"), ("Precision (%)", "Precision (%)"), ("Recall (%)", "Recall (%)"), ("F1-Score (%)", "F1-Score (%)"), ("ROC-AUC (%)", "ROC-AUC (%)")]
    rows = []
    for s_col, t_col in possible_pairs:
        if s_col in source_df.columns and t_col in target_df.columns:
            metric_name = s_col.replace(" (%)", "")
            try:
                s_val = float(source_df[s_col].iloc[0])
                t_val = float(target_df[t_col].iloc[0])
                retention = (t_val / s_val) * 100 if s_val > 0 else 0
                rows.append({"Metric": metric_name, "Retention": retention})
            except Exception: pass
    return pd.DataFrame(rows)

def call_deepseek_manual(basin, conf, thick, gr, rhob, nphi, pef, dtc, context_str=""):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-c4972c3ebfae4c748195fccf12231d87"
    }
    
    prompt = f"""You are a Principal Lead Geologist at an international energy corporation, specializing in Carbon Capture and Storage (CCS) site pre-screening.

Your task is to produce a highly professional, evidence-aware technical evaluation for a candidate geological interval in the {basin}.

==================================================
[UNIVERSAL SCREENING PHILOSOPHY]
This is a PRE-SCREENING system:
- Petrophysical quality and volumetric capacity determine reservoir potential.
- Geological containment risk (caprock + structure) determines viability.
- Safety and uncertainty must be explicitly acknowledged.

If local basin-specific evidence is available → use it strictly.
If local evidence is missing → DO NOT reject automatically. Instead, perform an "Analog-based" assessment using global standards (IPCC / IEAGHG), but clearly state the limitations.

==================================================
[DUAL VETO POWER & ENGINEERING RULES — CRITICAL]
You MUST act as a strict engineering guardrail. Do NOT compromise or average out critical failures.

1. THICKNESS VETO (The RAG Guardrail):
- Thickness is measured in LOG SAMPLES. Approximation: meters ≈ samples / 65 × 10.
- VETO RULE: If Thickness is LESS THAN 15 samples (~2.3 meters), it is GEOLOGICALLY USELESS for commercial CCS injection, NO MATTER HOW HIGH the ML Confidence or porosity is.
- IF TRIGGERED: Force "Screening Verdict" to "Low" or "Medium-Low". Explicitly state in the Reason that the severe lack of thickness overrides the AI's high prediction.

2. PETROPHYSICS VETO (The Tight Rock Baseline):
- VETO RULE: If ML Prediction Confidence is LESS THAN 0.30, the rock is inherently tight (e.g., mudstone/seal) with virtually zero storage capacity.
- IF TRIGGERED: Force "Screening Verdict" to "Low" or "Medium-Low". Explicitly state in the Reason that the interval is rejected due to poor petrophysical quality.

==================================================
INPUT DATA PROFILE
Target Basin: {basin}
ML Prediction Confidence: {conf:.4f}
Thickness: {thick} samples

Petrophysical Logs:
- GR: {gr} API
- RHOB: {rhob} g/cm3
- NPHI: {nphi} v/v
- PEF: {pef} b/e
- DTC: {dtc} us/ft

==================================================
RETRIEVED GEOLOGICAL CONTEXT
{context_str}

==================================================
[EVIDENCE & RISK RULES]
1. Evidence Type:
   - "Robust": Basin-specific geological evidence exists.
   - "Regional/Analog": Based on global CCS standards (no local data).
   - "Insufficient": No meaningful context retrieved.

2. Risk Evaluation:
   - With Robust evidence → evaluate directly.
   - With Analog evidence → assign Moderate or High uncertainty risk, NEVER Low Risk.
   - IF ANY VETO APPLIES (<15 samples OR ML < 0.30) → MUST assign High Risk.

==================================================
[SCREENING DECISION RULE (STRICT LOGIC)]
Evaluate strictly in this order:

- Medium-Low / Low:
  Triggers THICKNESS VETO (<15 samples) OR triggers PETROPHYSICS VETO (ML < 0.30) OR High Risk. (NO EXCEPTIONS).

- Medium:
  Moderate ML (0.30 - 0.60) AND Thickness >= 15 AND moderate uncertainty/analog evidence.

- Medium-High:
  Strong ML (>0.60) AND adequate thickness (>= 15) AND moderate risk (can be analog-based but MUST state uncertainty).

- High:
  Strong ML (>0.70) AND sufficient thickness (>30 samples) AND LOW structural risk WITH Robust basin-specific evidence.

==================================================
[OUTPUT FORMAT – STRICT]
Evidence Type: [Robust / Regional/Analog / Insufficient]
Caprock Evidence: [Strong / Moderate / Weak]
Structural Risk Evidence: [Low Risk / Moderate Risk / High Risk]
Screening Verdict: [High / Medium-High / Medium / Medium-Low / Low]

Reason:
(180–250 words)
MUST include:
1. Petrophysical + ML interpretation.
2. Thickness assessment (Apply explicit VETO if <15 samples).
3. Explicit statement of evidence type (Robust vs Analog).
4. Geological risk with citation (e.g., [Source: ..., Page: X]).
5. If ML < 0.30, clearly state the rock is too tight for storage.

Recommendation:
(1–2 actionable engineering steps focused on validation, e.g., seismic / seal verification)
"""
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a rigid, uncompromising geological AI engineering assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0  # 设置为0.0，消除所有随机性，让它严格执行规则
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"LLM Generation Error: {str(e)}"

def extract_probabilities(pred_df):
    candidates = ["1_probability", "target_1_probability", "class_1_probability", "prediction_probability", "probability"]
    for c in candidates:
        if c in pred_df.columns:
            probs = pred_df[c].values.astype(float)
            if np.nanmin(probs) < -0.01 or np.nanmax(probs) > 1.01:
                probs = 1 / (1 + np.exp(-probs))
            return probs
    prob_cols = [c for c in pred_df.columns if c.endswith("_probability")]
    not_zero = [c for c in prob_cols if not (c.startswith("0_") or "class_0" in c or "0_probability" in c)]
    prefer = [c for c in not_zero if "1_" in c or "target_1" in c or "class_1" in c]
    prob_col = prefer[0] if prefer else (not_zero[0] if not_zero else prob_cols[-1])
    probs = pred_df[prob_col].values.astype(float)
    if np.nanmin(probs) < -0.01 or np.nanmax(probs) > 1.01: 
        probs = 1 / (1 + np.exp(-probs))
    return probs

df = load_data(EXCEL_PATH)
report_dict = load_json_reports(JSON_PATH)

if df.empty:
    st.error("Data source unavailable. Please ensure ML & RAG pipeline has been executed.")
    st.stop()

RISK_COL_NAME = "Risk Evidence" if "Risk Evidence" in df.columns else "Structural Risk Evidence" if "Structural Risk Evidence" in df.columns else None

if st.session_state.current_page != "Home":
    with st.sidebar:
        st.markdown("### Screening Filters")
        st.markdown("<p style='color:#64748b; font-size:1.1rem;'>Configure parameters to filter basin-scale candidates.</p>", unsafe_allow_html=True)

        valid_verdicts = df["Screening Verdict"].dropna().unique().tolist() if "Screening Verdict" in df.columns else []
        selected_verdicts = st.multiselect("Geological Verdict", options=valid_verdicts, default=valid_verdicts)

        st.divider()

        filtered_df = df[df["Screening Verdict"].isin(selected_verdicts)].copy()
        filtered_df["__rank"] = filtered_df["Screening Verdict"].map(VERDICT_ORDER).fillna(0) if "Screening Verdict" in filtered_df.columns else 0
        ranked_df = filtered_df.sort_values(by=["__rank", "ML Confidence"], ascending=[False, False])

        if not ranked_df.empty:
            csv_export = ranked_df.drop(columns=["__rank"], errors="ignore").to_csv(index=False).encode("utf-8")
            st.download_button(label="Export Candidate Assets", data=csv_export, file_name="ccs_screening_assets.csv", mime="text/csv")
else:
    filtered_df = df.copy()
    filtered_df["__rank"] = filtered_df["Screening Verdict"].map(VERDICT_ORDER).fillna(0) if "Screening Verdict" in filtered_df.columns else 0
    ranked_df = filtered_df.sort_values(by=["__rank", "ML Confidence"], ascending=[False, False])

avg_conf = ranked_df["ML Confidence"].mean() if not ranked_df.empty else 0.0
best_target = ranked_df.iloc[0]["Interval ID"] if not ranked_df.empty else "N/A"
top_tier = ranked_df.iloc[0]["Screening Verdict"] if not ranked_df.empty else "N/A"


# PAGE: HOME 
if st.session_state.current_page == "Home":
    st.markdown("<div style='height: 25vh;'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="text-align: center; max-width: 1100px; margin: 0 auto;">
        <h1 class="animate-fade-up" style="font-size: 5.5rem; font-weight: 900; color: #38bdf8; letter-spacing: -0.02em; margin-bottom: 30px; text-shadow: 2px 4px 12px rgba(0,0,0,0.8);">
            RAG Assistance for<br><span style="color: #ffffff;"> CCS Site Screening</span>
        </h1>
        <p class="animate-fade-up delay-1" style="font-size: 1.8rem; font-weight: 400; color: #ffffff; line-height: 1.8; margin-bottom: 50px; text-shadow: 1px 2px 8px rgba(0,0,0,0.8);">
            Industrial emissions pose an unprecedented threat to global climate stability.<br>
            The critical challenge in modern environmental engineering lies in rapidly and safely identifying optimal geological sites for permanent CO₂ sequestration.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_c1, col_c2, col_c3 = st.columns([1.5, 1, 1.5])
    with col_c2:
        action_placeholder = st.empty()
        
        if action_placeholder.button(" Launch Screening Engine", use_container_width=True, type="primary"):
            action_placeholder.markdown("""
            <div style="padding: 15px 20px; background: rgba(15, 23, 42, 0.85); border-radius: 8px; border: 1px solid #38bdf8; box-shadow: 0 0 20px rgba(56, 189, 248, 0.5); backdrop-filter: blur(8px);">
                <div style="color: #38bdf8; font-size: 1.05rem; font-weight: 700; margin-bottom: 8px; font-family: monospace;">
                    🌟Initializing Engine Clusters... <br>
                    🌟Loading Petrophysical Weights...
                </div>
                <div style="width: 100%; background-color: #1e293b; border-radius: 4px; height: 6px; overflow: hidden;">
                    <div style="width: 100%; height: 100%; background-color: #38bdf8; animation: load-bar 1.5s ease-in-out forwards;"></div>
                </div>
            </div>
            <style>
                @keyframes load-bar { 0% { width: 0%; } 50% { width: 70%; } 100% { width: 100%; } }
            </style>
            """, unsafe_allow_html=True)
            
            time.sleep(1.6)  
            st.session_state.current_page = "Basin Overview"
            st.rerun()

# PAGE: Basin Overview
elif st.session_state.current_page == "Basin Overview":
    st.markdown("""
    <div class="journal-header">
    <div class="j-title">RAG ASSISTANCE FOR CCS SITE SCREENING</div>
    <div class="j-desc">An interpretable decision-support framework for geological carbon storage, integrating representation learning (FTTransformer) with LLM-assisted geological verification (DeepSeek-RAG).</div>
    <div class="j-meta-container">
    <div class="j-meta-block"><span class="j-meta-label">Study Area</span><span class="j-meta-value">Tarim Basin</span></div>
    <div class="j-meta-block"><span class="j-meta-label">Core Engine</span><span class="j-meta-value">FTTransformer + DeepSeek-RAG</span></div>
    <div class="j-meta-block"><span class="j-meta-label">Evaluation Mode</span><span class="j-meta-value">Blind-Test Validation</span></div>
    </div>
    </div>
    <div class="clean-pipeline">
    <div class="cp-step">1. Datase Prep</div><div class="cp-arrow">›</div>
    <div class="cp-step">2. Model Training & Best Model</div><div class="cp-arrow">›</div>
    <div class="cp-step">3. Interval Ranking</div><div class="cp-arrow">›</div>
    <div class="cp-step active">4. RAG Verification</div><div class="cp-arrow">›</div>
    <div class="cp-step active">5. Final Recommendation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel-header" style="margin-top: 0;"><div class="panel-title">Executive Summary & Screening Policy</div></div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-title">Filtered Candidates</div><div class="kpi-value">{len(ranked_df)}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-title">Mean Confidence</div><div class="kpi-value">{avg_conf:.4f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-title">Top Recommended Interval</div><div class="kpi-value">{best_target}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-title">Highest Screening Grade</div><div class="kpi-badge-wrap">{render_badge(str(top_tier))}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    high_count = len(ranked_df[ranked_df["Screening Verdict"].isin(["High", "Medium-High"])]) if not ranked_df.empty else 0

    st.markdown("""
<div style="margin-top: 16px; margin-bottom: 16px; padding-left: 18px; border-left: 5px solid #3b82f6;">
<div style="font-size: 1.25rem; font-weight: 900; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Screening Decision Policy</div>
<div style="font-size: 1.25rem; color:#475569; line-height: 1.7;">
<b>High / Medium-High:</b> strong ML reservoir indication, adequate thickness proxy, caprock support, and low structural uncertainty.<br>
<b>Medium:</b> partial geological support or moderate uncertainty; requires site-specific validation.<br>
<b>Medium-Low / Low:</b> insufficient volume proxy, weak ML confidence, or elevated structural risk.
</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="panel-header"><div class="panel-title">Asset Distribution & Prioritized Database</div></div>', unsafe_allow_html=True)

    main_left, main_right = st.columns([1.25, 1.55])

    with main_left:
        st.markdown('<div class="panel-header"><div class="panel-title">Screening Verdict Distribution</div><div class="panel-subtitle">Basin-scale distribution of integrated geological verdicts.</div></div>', unsafe_allow_html=True)
        if not ranked_df.empty and "Screening Verdict" in ranked_df.columns:
            vc = ranked_df["Screening Verdict"].value_counts().reindex(list(VERDICT_ORDER.keys())).fillna(0)
            dist_df = vc[vc > 0].reset_index()
            dist_df.columns = ["Verdict", "Count"]
            bar_chart = alt.Chart(dist_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=36).encode(
                x=alt.X("Verdict:N", sort=list(VERDICT_ORDER.keys()), axis=alt.Axis(labelAngle=-35, title=None, labelFontSize=18)),
                y=alt.Y("Count:Q", axis=alt.Axis(title="Candidate Count", titleFontSize=18)),
                color=alt.Color("Verdict:N", scale=alt.Scale(domain=list(ALTAIR_COLORS.keys()), range=list(ALTAIR_COLORS.values())), legend=None),
                tooltip=["Verdict", "Count"]
            )
            bar_text = alt.Chart(dist_df).mark_text(dy=-6, fontWeight="bold", fontSize=20).encode(x=alt.X("Verdict:N", sort=list(VERDICT_ORDER.keys())), y="Count:Q", text="Count:Q")
            st.altair_chart((bar_chart + bar_text).properties(height=300).configure_view(strokeWidth=0), use_container_width=True)

        st.markdown('<div class="panel-header" style="margin-top: 30px;"><div class="panel-title">Verdict vs Confidence Distribution</div><div class="panel-subtitle">Relationship between FTTransformer confidence and final geological grading.</div></div>', unsafe_allow_html=True)
        if not ranked_df.empty and "Screening Verdict" in ranked_df.columns and "ML Confidence" in ranked_df.columns:
            scatter_chart = alt.Chart(ranked_df).mark_circle(size=120, opacity=0.8).encode(
                x=alt.X("Screening Verdict:N", sort=list(VERDICT_ORDER.keys()), axis=alt.Axis(labelAngle=-35, title=None, labelFontSize=18)),
                y=alt.Y("ML Confidence:Q", axis=alt.Axis(title="FTTransformer Confidence", titleFontSize=16), scale=alt.Scale(domain=[0.5, 1.0])),
                color=alt.Color("Screening Verdict:N", scale=alt.Scale(domain=list(ALTAIR_COLORS.keys()), range=list(ALTAIR_COLORS.values())), legend=None),
                tooltip=["Interval ID", "Screening Verdict", "ML Confidence", "Thickness"]
            ).properties(height=320).configure_view(strokeWidth=0)
            st.altair_chart(scatter_chart, use_container_width=True)

        st.markdown("<p style='font-size:1.1rem; color:#64748b; font-style:italic; margin-top:12px; line-height:1.6;'>* The combined plots show both basin-scale verdict composition and how raw ML confidence is modulated by geological risk-aware screening logic.</p>", unsafe_allow_html=True)

    with main_right:
        st.markdown('<div class="panel-header"><div class="panel-title">Prioritized Candidate Database</div><div class="panel-subtitle">Top-ranked candidates under the current screening filters.</div></div>', unsafe_allow_html=True)
        display_df = ranked_df.copy()
        core_cols = ["Interval ID", "Screening Verdict", "ML Confidence", "Thickness", "Caprock Evidence"]
        if RISK_COL_NAME: core_cols.append(RISK_COL_NAME)
        actual_cols = [c for c in core_cols if c in display_df.columns]
        if actual_cols:
            st.dataframe(
                display_df[actual_cols],
                column_config={"ML Confidence": st.column_config.ProgressColumn("ML Confidence", format="%.3f", min_value=0.0, max_value=1.0), "Thickness": st.column_config.NumberColumn("Thickness", format="%.1f")},
                height=780, use_container_width=True, hide_index=True
            )

# PAGE: Target Deep Dive
elif st.session_state.current_page == "Target Deep Dive":
    st.markdown("""
    <div class="journal-header">
    <div class="j-title">RAG ASSISTANCE FOR CCS SITE SCREENING</div>
    <div class="j-desc">An interpretable decision-support framework for geological carbon storage, integrating representation learning (FTTransformer) with LLM-assisted geological verification (DeepSeek-RAG).</div>
    <div class="j-meta-container">
    <div class="j-meta-block"><span class="j-meta-label">Study Area</span><span class="j-meta-value">Tarim Basin</span></div>
    <div class="j-meta-block"><span class="j-meta-label">Core Engine</span><span class="j-meta-value">FTTransformer + DeepSeek-RAG</span></div>
    <div class="j-meta-block"><span class="j-meta-label">Evaluation Mode</span><span class="j-meta-value">Blind-Test Validation</span></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
        div[role="radiogroup"] { padding: 10px 0 20px 0; }
        div[role="radiogroup"] label p, 
        div[role="radiogroup"] label div { 
            font-weight: 800 !important; 
            color: #FF0000 !important; 
            font-size: 1.35rem !important; 
            line-height: 1.5 !important;
        }
        
        div[role="radiogroup"] label span[data-baseweb="radio"] {
            transform: scale(1.2);
            margin-right: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    eval_mode = st.radio("Select Operation Mode", ["Precomputed Case Study (Tarim Basin)", "Interactive AI Screening Tool"], horizontal=True, label_visibility="collapsed")
    st.markdown("<hr style='margin-top: 0px; border-color: #e2e8f0; border-width: 2px;'>", unsafe_allow_html=True)

    if eval_mode == "Precomputed Case Study (Tarim Basin)":
        if ranked_df.empty: st.warning("No candidates available for analysis.")
        else:
            st.markdown("<div style='font-size: 1.3rem; font-weight: 800; color: #1e293b; margin-bottom: 12px;'>Select Target Interval for Deep Dive</div>", unsafe_allow_html=True)
            selected_id = st.selectbox("", ranked_df["Interval ID"].tolist(), label_visibility="collapsed")
            sel_row = ranked_df[ranked_df["Interval ID"] == selected_id].iloc[0]
            sel_report = report_dict.get(selected_id, {})
            report_full_text = sel_report.get("report_text", "")
            reasoning = extract_reason(report_full_text)
            recommendation = extract_recommendation(report_full_text)
            evidence_data = sel_report.get("evidence", [])

            st.markdown('<div class="panel-header"><div class="panel-title">Integrated Geological Assessment</div></div>', unsafe_allow_html=True)
            risk_info = str(sel_row.get(RISK_COL_NAME, "N/A") if RISK_COL_NAME else "N/A")
            petro_raw = str(sel_row.get("Petrophysics", "nan"))
            petro_status = "Available (Integrated)" if (petro_raw != "nan" and petro_raw != "Not available" and petro_raw.strip() != "") else "Not Available"

            st.markdown(f"""
<div class="factor-grid">
<div class="factor-item"><div class="factor-label">ML Confidence</div><div class="factor-val">{sel_row.get('ML Confidence', 0):.4f}</div></div>
<div class="factor-item"><div class="factor-label">Relative Thickness</div><div class="factor-val">{sel_row.get('Thickness', 'N/A')}</div></div>
<div class="factor-item"><div class="factor-label">Caprock Integrity</div><div class="factor-val">{sel_row.get('Caprock Evidence', 'N/A')}</div></div>
<div class="factor-item"><div class="factor-label">Petrophysics</div><div class="factor-val">{petro_status}</div></div>
<div class="factor-item"><div class="factor-label">Final Verdict</div><div class="kpi-badge-wrap" style="margin-top:2px;">{render_badge(str(sel_row.get('Screening Verdict', 'N/A')))}</div></div>
</div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            risk_text_lower = risk_info.lower()
            if any(kw in risk_text_lower for kw in ["high risk", "moderate", "fault", "active", "uncertain", "limited seal"]):
                st.warning(f"**Key Structural Uncertainty:** {risk_info}", icon="⚠️")
            else:
                st.info(f"**Structural Assessment:** {risk_info}", icon="🛡️")

            st.markdown("""
<div style="margin-top: 16px; margin-bottom: 18px; padding: 20px 24px; background: #f8fbff; border-left: 5px solid #2563eb; border-radius: 0 8px 8px 0;">
<div style="font-size: 1.2rem; font-weight: 900; color: #1d4ed8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">RAG Reasoning Constraints</div>
<div style="font-size: 1.25rem; color: #334155; line-height: 1.7;">
<b>Evidence-only policy:</b> reasoning must be grounded in retrieved geological context.<br>
<b>Unit discipline:</b> thickness proxy is not interpreted as absolute depth.<br>
<b>Risk-first grading:</b> structural uncertainty must be evaluated before assigning favorable verdicts.<br>
<b>Verdict consistency:</b> grading must align with CCS screening criteria.
</div></div>
            """, unsafe_allow_html=True)

            if reasoning or recommendation:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("RAG-Based Geological Reasoning (Evidence-Constrained)", expanded=True):
                    content_html = ""
                    if reasoning: content_html += f"<div style='color:#1e293b; font-size:1.3rem; line-height:1.7;'>{reasoning}</div>"
                    if recommendation:
                        content_html += f"""
<hr style='margin: 24px 0; border: none; border-top: 2px dashed #cbd5e1;'>
<div style='background-color: #f0f9ff;'>
<div style='font-size: 1.15rem; font-weight: 900; color: #0369a1; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;'>Actionable Recommendation</div>
<div style='font-size: 1.25rem; font-weight: 600; color: #0f172a; line-height: 1.7;'>{recommendation}</div></div>
                        """
                    st.markdown(content_html, unsafe_allow_html=True)

            st.markdown('<div class="panel-header"><div class="panel-title">Interpretability & Knowledge Traceability</div></div>', unsafe_allow_html=True)
            col_trust1, col_trust2 = st.columns(2)
            with col_trust1:
                st.markdown("<div style='font-size: 1.3rem; font-weight: 800; color: #1e293b; margin-bottom: 12px;'>Local SHAP Attribution</div>", unsafe_allow_html=True)
                if os.path.exists(SHAP_WATERFALL_PATH):
                    st.image(SHAP_WATERFALL_PATH, use_container_width=True)
                    st.markdown("<p style='font-size:1.15rem; color:#64748b; line-height:1.5; border-top:1px solid #e2e8f0; padding-top:10px;'><i>* This plot explains which petrophysical features most strongly pushed the selected interval toward a favorable or unfavorable screening outcome.</i></p>", unsafe_allow_html=True)
                else: st.caption("Local Waterfall plot unavailable.")

            with col_trust2:
                st.markdown("<div style='font-size: 1.3rem; font-weight: 800; color: #1e293b; margin-bottom: 12px;'>RAG Literature Evidence</div>", unsafe_allow_html=True)
                if evidence_data:
                    with st.expander(f"📚 Evidence Coverage: {len(evidence_data)} Sources Retrieved", expanded=False):
                        st.markdown("<div style='max-height: 400px; overflow-y: auto; padding-right: 12px;'>", unsafe_allow_html=True)
                        for ev in evidence_data:
                            source = ev.get("source", "Unknown Reference")
                            page = ev.get("page", "N/A")
                            snippet = short_snippet(ev.get("snippet", ev.get("excerpt", ev.get("summary", ""))))
                            snippet_html = f'<div class="evidence-snippet" style="font-size: 1.1rem; line-height: 1.6;">"{snippet}"</div>' if snippet else ""
                            st.markdown(f"""
<div class="evidence-card">
<div class="evidence-source" style="font-size: 1.2rem;">{source}</div>
<div class="evidence-meta"><span class="evidence-page" style="font-size: 1.0rem;">Page / Section: {page}</span><span class="evidence-tag" style="font-size: 1.0rem;">✓ Geological Context</span></div>
{snippet_html}</div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                else: st.caption("No citation metadata found for this target.")


    # Mode B: End-to-end real-time prediction (Model + RAG)
    else:
        st.markdown('<div class="panel-header" style="margin-top: 0;"><div class="panel-title">End-to-End Live Screening</div><div class="panel-subtitle">Input core petrophysical parameters to trigger real-time ML prediction and RAG synthesis.</div></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <style>
            /* 针对文本输入框和数字输入框的标签 */
            div[data-testid="stTextInput"] label p, 
            div[data-testid="stNumberInput"] label p {
                font-size: 1.15rem !important; 
                font-weight: 700 !important;   
                color: #1e293b !important;     
            }
        </style>
        """, unsafe_allow_html=True)

        col_m_left, col_m_right = st.columns([1, 1], gap="large")
        
        with col_m_left:
            st.markdown("<div class='manual-group-title' style='color: #2563eb; font-size: 1.2rem; margin-bottom: 15px;'> Global Constraints & Acoustics</div>", unsafe_allow_html=True)
            m_basin = st.text_input("Target Basin", value="Sichuan Basin")
            m_thick = st.number_input("Thickness Proxy (samples)", value=70)
            st.markdown("<hr style='margin: 14px 0; border: none; border-top: 2px dashed #e2e8f0;'>", unsafe_allow_html=True)
            m_dtc = st.number_input("Compressional Sonic (DTC) [us/ft]", value=85.0)
            m_pef = st.number_input("Photoelectric Factor (PEF) [b/e]", value=2.6)

        with col_m_right:
            st.markdown("<div class='manual-group-title' style='color: #2563eb; font-size: 1.2rem; margin-bottom: 15px;'> Core Petrophysics</div>", unsafe_allow_html=True)
            m_gr = st.number_input("Gamma Ray (GR) [API]", value=60.0)
            m_rhob = st.number_input("Bulk Density (RHOB) [g/cm3]", value=2.35)
            m_nphi = st.number_input("Neutron Porosity (NPHI) [v/v]", value=0.22)

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button(" Execute End-to-End Evaluation (ML + RAG)", type="primary", use_container_width=True):
            
            is_real_calc = False
            real_ml_conf = 0.0 
            
            with st.status(" Initializing Deep AI Engine...", expanded=True) as status:
                st.write(" Step 1/2: Ingesting petrophysical well-logs & Calculating ML Probability...")
                time.sleep(0.8) 
                
                try:
                    excel_path = os.path.join(REPORT_DIR, "model_final_summary.xlsx")
                    if os.path.exists(excel_path):
                        final_excel_df = pd.read_excel(excel_path)
                        core_results = final_excel_df[final_excel_df['FeatureSet'] == 'Core']
                        champ_model_name = core_results.sort_values('F1-Score (%)', ascending=False).iloc[0]['Model']
                    else: champ_model_name = "FTTransformer"


                    target_folder_1 = os.path.join(PROJECT_ROOT, "notebook", "saved_models", "my_true_champion")
                    target_folder_2 = os.path.join(PROJECT_ROOT, "saved_models", "my_true_champion")
                    
                    valid_model_path = None
                    if os.path.exists(os.path.join(target_folder_1, "config.yml")):
                        valid_model_path = target_folder_1
                    elif os.path.exists(os.path.join(target_folder_2, "config.yml")):
                        valid_model_path = target_folder_2
                    
                    input_data = pd.DataFrame([{'GR': m_gr, 'RHOB': m_rhob, 'NPHI': m_nphi, 'PEF': m_pef, 'DTC': m_dtc}])
                    
                    if valid_model_path:
                        loaded_model = TabularModel.load_model(valid_model_path)
                        real_ml_conf = extract_probabilities(loaded_model.predict(input_data))[0]
                        is_real_calc = True
                        st.write(f" Champion Model ({champ_model_name}) Loaded. Conf: {real_ml_conf:.4f}")
                    else:
                        st.write(f" Weights missing. Engaging heuristic fallback...")
                        pseudo_score = 0.5
                        if m_gr < 80: pseudo_score += 0.2
                        if m_rhob < 2.55: pseudo_score += 0.15
                        if m_nphi > 0.12: pseudo_score += 0.1
                        real_ml_conf = min(0.98, pseudo_score * (1 + (np.random.rand() * 0.05)))
                        
                except Exception as e:
                    real_ml_conf = 0.42 
                    is_real_calc = False
                    st.write(f" Error: {e}")

                st.write(" Step 2/2: Querying ChromaDB and Triggering DeepSeek RAG...")
                time.sleep(0.5)
                context_str = ""
                if RAG_AVAILABLE:
                    try:
                        query_str = f"CCS site screening criteria {m_basin} caprock seal structural fault risk global standard requirements"
                        docs = vectorstore.similarity_search(query_str, k=5)
                        context_str = format_docs_with_citations(docs)
                        st.write(f" Retrieved {len(docs)} relevant geological documents.")
                    except Exception as e:
                        st.write(f"RAG Retrieval failed: {e}. Falling back to LLM base knowledge.")
                
                live_result = call_deepseek_manual(m_basin, real_ml_conf, m_thick, m_gr, m_rhob, m_nphi, m_pef, m_dtc, context_str=context_str)

                status.update(label="Evaluation Complete!", state="complete", expanded=False)
            st.markdown("### Integrated AI Verification Report")
            parsed_data = parse_llm_output(live_result)
            evidence_weight = get_evidence_weight(parsed_data.get("Evidence Type", "Unknown"))
            final_confidence = real_ml_conf * evidence_weight
            verdict_badge_html = render_badge(parsed_data["Screening Verdict"])
            margin_bottom = "20px" if parsed_data["Recommendation"] else "0px"

            col_res_left, col_res_right = st.columns([1, 2.5])
            
            with col_res_left:
                gauge_color = "#10b981" if real_ml_conf >= 0.8 else ("#f59e0b" if real_ml_conf >= 0.6 else "#ef4444")
                start_angle = -np.pi / 2
                end_angle = start_angle + (real_ml_conf * np.pi)
                
                bg_arc = alt.Chart(pd.DataFrame({'v': [1]})).mark_arc(
                    innerRadius=60, outerRadius=80, color='#e2e8f0', cornerRadius=4
                ).encode(theta=alt.value(start_angle), theta2=alt.value(np.pi / 2))
                
                arc = alt.Chart(pd.DataFrame({'v': [1]})).mark_arc(
                    innerRadius=60, outerRadius=80, color=gauge_color, cornerRadius=4
                ).encode(theta=alt.value(start_angle), theta2=alt.value(end_angle))
                
                text_df = pd.DataFrame({'value': [real_ml_conf]})
                text = alt.Chart(text_df).mark_text(align='center', baseline='middle', dy=0, fontSize=30, fontWeight=900, color='#0f172a').encode(text=alt.Text('value:Q', format='.2f'))
                label = alt.Chart(text_df).mark_text(align='center', baseline='middle', dy=30, fontSize=13, fontStyle='italic', color='#64748b').encode(text=alt.value('ML Confidence'))
                gauge_chart = (bg_arc + arc + text + label).properties(width=220, height=200).configure_view(strokeWidth=0).configure(padding=10)
                st.altair_chart(gauge_chart, use_container_width=True)
                
                if not is_real_calc:
                    st.markdown("<p style='font-size:0.95rem; color:#94a3b8; text-align:center; margin-top:-20px;'>*(Simulated proxy score)*</p>", unsafe_allow_html=True)

            with col_res_right:
                unified_html = f"""
<div class="pulse-card" style="margin-top: 0px; box-shadow: 0 4px 15px rgba(37,99,235,0.08);">
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
<span style="font-size: 1.4rem;">🤖</span>
<span style="font-size: 1.25rem; font-weight: 900; color: #1e293b; letter-spacing: 0.5px;"> DeepSeek Synthesized Reasoning</span>
</div>

<div style="display: flex; gap: 15px; margin-bottom: 24px; flex-wrap: wrap;">
<div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 12px 18px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); flex: 1;">
<div style="font-size: 0.85rem; color: #64748b; font-weight: 800; text-transform: uppercase;">Final Verdict</div>
<div style="margin-top: 8px;">{verdict_badge_html}</div>
</div>

<div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 12px 18px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); flex: 1;">
<div style="font-size: 0.85rem; color: #64748b; font-weight: 800; text-transform: uppercase;">Evidence Type</div>
<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 6px;">{parsed_data['Evidence Type']}</div>
</div>

<div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 12px 18px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); flex: 1;">
<div style="font-size: 0.85rem; color: #64748b; font-weight: 800; text-transform: uppercase;">Final Confidence</div>
<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 6px;">{final_confidence:.2f}</div>
<div style="font-size: 0.82rem; color: #64748b; margin-top: 4px;">ML × Evidence Weight ({evidence_weight:.2f})</div>
</div>

<div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 12px 18px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); flex: 1;">
<div style="font-size: 0.85rem; color: #64748b; font-weight: 800; text-transform: uppercase;">Caprock Integrity</div>
<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 6px;">{parsed_data['Caprock Evidence']}</div>
</div>

<div style="background: #ffffff; border: 1px solid #e2e8f0; padding: 12px 18px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.02); flex: 1;">
<div style="font-size: 0.85rem; color: #64748b; font-weight: 800; text-transform: uppercase;">Structural Risk</div>
<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 6px;">{parsed_data['Structural Risk Evidence']}</div>
</div>
</div>

<div style="color: #334155; font-size: 1.15rem; line-height: 1.8; margin-bottom: {margin_bottom}; text-align: justify; padding: 0 5px;">
{parsed_data['Reason']}
</div>
"""
                
                if parsed_data["Recommendation"]:
                    unified_html += f"""
<hr style='margin: 24px 0; border: none; border-top: 2px dashed #cbd5e1;'>
<div style='background-color: #f0f9ff; border-left: 5px solid #0284c7; padding: 18px 24px;'>
<div style='font-size: 1.15rem; font-weight: 900; color: #0369a1; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;'>
Actionable Recommendation
</div>
<div style='font-size: 1.25rem; font-weight: 600; color: #0f172a; line-height: 1.7;'>
{parsed_data['Recommendation']}
</div>
</div>
"""
                
                unified_html += """</div>"""
                st.markdown(unified_html, unsafe_allow_html=True)
                
            final_verdict = str(parsed_data.get("Screening Verdict", "Unknown")).strip()

            if final_verdict in ["High", "Medium-High"]:
                st.toast("Target Verified: Favorable.", icon="🎯")
                st.balloons()
                st.success(
                    f"🎯 **Favorable Screening Outcome.** "
                    f"The integrated ML + RAG assessment suggests this interval is promising, "
                    f"with an adjusted final confidence of {final_confidence:.2f}. "
                    f"Further basin-specific validation remains necessary where evidence is analog-based."
                )
            elif final_verdict == "Medium":
                st.toast("Target Verified: Requires Review.", icon="📊")
                st.info(
                    f"📊 **Moderate Candidate.** "
                    f"The interval shows partial support, with an adjusted final confidence of {final_confidence:.2f}. "
                    f"Additional basin-specific geological validation is required."
                )
            elif final_verdict in ["Medium-Low", "Low"]:
                st.toast("Target Rejected: Elevated Risk.", icon="🚫")
                st.error(
                    f"⚠️ **Elevated Geological Risk.** "
                    f"The integrated assessment does not currently support this interval as a favorable CCS candidate. "
                    f"Adjusted final confidence: {final_confidence:.2f}."
            )

            st.markdown('<div class="panel-header" style="margin-top: 30px;"><div class="panel-title">Interpretability & Knowledge Traceability</div></div>', unsafe_allow_html=True)
            
            st.markdown("<div style='font-size: 1.3rem; font-weight: 800; color: #1e293b; margin-bottom: 12px;'>Live RAG Literature Evidence</div>", unsafe_allow_html=True)
            
            if 'docs' in locals() and docs:
                with st.expander(f"📚 Evidence Coverage: {len(docs)} Sources Retrieved for {m_basin}", expanded=False):
                    st.markdown("<div style='max-height: 400px; overflow-y: auto; padding-right: 12px;'>", unsafe_allow_html=True)
                    
                    for d in docs:
                        source = os.path.basename(str(d.metadata.get("source", "Unknown Reference")))
                        page = d.metadata.get("page", "N/A")
                        raw_content = d.page_content.strip()
                        snippet = (raw_content[:160] + "...") if len(raw_content) > 160 else raw_content
                        snippet_html = f'<div class="evidence-snippet" style="font-size: 1.1rem; line-height: 1.6;">"{snippet}"</div>'
                        
                        st.markdown(f"""
<div class="evidence-card">
<div class="evidence-source" style="font-size: 1.2rem;">{source}</div>
<div class="evidence-meta">
<span class="evidence-page" style="font-size: 1.0rem;">Page / Section: {page}</span>
<span class="evidence-tag" style="font-size: 1.0rem;">✓ Real-time Retrieved Context</span>
</div>
{snippet_html}
</div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.caption("No citation metadata found for this live query. (Check RAG database connection)")

elif st.session_state.current_page == "Engine Validation":
    st.markdown("""
    <div class="journal-header">
        <div class="j-title">RAG ASSISTANCE FOR CCS SITE SCREENING</div>
        <div class="j-desc">An interpretable decision-support framework for geological carbon storage, integrating representation learning (FTTransformer) with LLM-assisted geological verification (DeepSeek-RAG).</div>
        <div class="j-meta-container">
            <div class="j-meta-block"><span class="j-meta-label">Study Area</span><span class="j-meta-value">Tarim Basin</span></div>
            <div class="j-meta-block"><span class="j-meta-label">Core Engine</span><span class="j-meta-value">FTTransformer + DeepSeek-RAG</span></div>
            <div class="j-meta-block"><span class="j-meta-label">Evaluation Mode</span><span class="j-meta-value">Blind-Test Validation</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    f_df = load_data(FORCE_SUMMARY_PATH)
    t_df = load_data(TARIM_SUMMARY_PATH)

    f1_source_raw = get_f1_score(f_df)
    f1_target_raw = get_f1_score(t_df)

    if f1_source_raw is not None and f1_target_raw is not None:
        f1_source = parse_to_percentage(f1_source_raw)
        f1_target = parse_to_percentage(f1_target_raw)
        gap = f1_target - f1_source

        if gap >= -5: 
            transfer_status, status_desc = "Stable Transfer", "Minimal degradation under blind-test transfer."
        elif gap >= -15: 
            transfer_status, status_desc = "Moderate Drop", "Acceptable performance loss across basins."
        else: 
            transfer_status, status_desc = "Substantial Drop", "Significant reduction suggests limited transferability."

        gap_color = "normal" if gap >= 0 else "inverse"

        st.markdown('<div class="panel-header" style="margin-bottom: 24px;"><div class="panel-title">Cross-Basin Zero-Shot Generalization Performance</div><div class="panel-subtitle">Testing FTTransformer representation transfer from North Sea (Source) to Tarim Basin (Target).</div></div>', unsafe_allow_html=True)

        c_empty1, b1, b2, b3, c_empty2 = st.columns([1, 2, 2, 2, 1])
        b1.metric("Source Domain F1 (North Sea)", f"{f1_source:.2f}%")
        b2.metric("Target Domain F1 (Tarim Basin)", f"{f1_target:.2f}%", f"{gap:+.2f}% vs Source", delta_color=gap_color)
        b3.metric("Transfer Status", transfer_status)
        with b3: 
            st.caption(status_desc)

        st.markdown("<br>", unsafe_allow_html=True)
        st.success("**Research Takeaway:** Cross-basin zero-shot results suggest that the FTTransformer representation captures transferable reservoir signatures rather than only basin-specific patterns.", icon="✅")
        st.divider()

        comp_df = build_metrics_comparison_df(f_df, t_df)
        retention_df = build_metrics_retention_df(f_df, t_df)

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.markdown("#### Generalization Metrics Comparison")
            if not comp_df.empty:
                bars = alt.Chart(comp_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X("Domain:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                    y=alt.Y("Score:Q", title="Score (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelFontSize=22, titleFontSize=20)),
                    color=alt.Color("Domain:N", scale=alt.Scale(domain=["Source (North Sea)", "Target (Tarim Basin)"], range=["#94a3b8", "#2563eb"]), legend=alt.Legend(orient="bottom", title=None, labelFontSize=20)),
                    column=alt.Column("Metric:N", title=None, spacing=14, header=alt.Header(labelFontSize=20, labelFontWeight="bold")),
                    tooltip=["Metric", "Domain", "Score"]
                ).properties(width=100, height=280)
                st.altair_chart(bars.configure_view(stroke="transparent"), use_container_width=False)
            else: 
                st.info("Metric comparison plot unavailable.")

        with gcol2:
            st.markdown("#### Performance Preservation (Metric Retention Rate)")
            if not retention_df.empty:
                max_retention = max(100, retention_df['Retention'].max() + 5)
                retention_chart = alt.Chart(retention_df).mark_bar(cornerRadiusEnd=4).encode(
                    y=alt.Y("Metric:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=20)),
                    x=alt.X("Retention:Q", title="Retention Rate (%)", scale=alt.Scale(domain=[0, max_retention]), axis=alt.Axis(labelFontSize=20, titleFontSize=18)),
                    color=alt.Color("BarColor:N", scale=None),
                    tooltip=["Metric", alt.Tooltip("Retention:Q", format=".1f", title="Retention (%)")]
                ).transform_calculate(BarColor="datum.Retention >= 90 ? '#10b981' : (datum.Retention >= 75 ? '#eab308' : '#ef4444')").properties(height=280)
                
                baseline_rule = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(color="#cbd5e1", strokeDash=[4, 4]).encode(x="x:Q")
                st.altair_chart((retention_chart + baseline_rule).configure_view(strokeWidth=0), use_container_width=True)
            else: 
                st.info("Retention rate plot unavailable.")
    else:
        st.error("🚨 **Critical Data Missing! Cannot render Validation Dashboard.**")

    st.markdown('<div class="panel-header" style="margin-top: 30px; margin-bottom: 24px;"><div class="panel-title">Model Diagnostics & Classification Curves</div></div>', unsafe_allow_html=True)
    
    c_diag1, c_diag2 = st.columns(2)
    with c_diag1:
        if os.path.exists(LEARNING_CURVE_PATH): 
            st.image(LEARNING_CURVE_PATH, caption="Final Learning Curve", use_container_width=True)
        else: 
            st.info("Learning Curve missing.")
    with c_diag2:
        if os.path.exists(CONF_MATRIX_PATH): 
            st.image(CONF_MATRIX_PATH, caption="Blind-Test Confusion Matrix", use_container_width=True)
        else: 
            st.info("Confusion Matrix missing.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_diag3, c_diag4 = st.columns(2)
    with c_diag3:
        if os.path.exists(ROC_CURVE_PATH): 
            st.image(ROC_CURVE_PATH, caption="Blind-Test ROC Curve", use_container_width=True)
        else: 
            st.info("ROC Curve missing.")
    with c_diag4:
        if os.path.exists(PR_CURVE_PATH): 
            st.image(PR_CURVE_PATH, caption="Blind-Test PR Curve", use_container_width=True)
        else: 
            st.info("PR Curve missing.")
    
    CALIB_PLOT_PATH = os.path.join(PROJECT_ROOT, "figures", "model_plot", "8_X_threshold_calibration_source.png")
    
    if os.path.exists(CALIB_PLOT_PATH):
        st.markdown('<div class="panel-header" style="margin-top: 40px; margin-bottom: 24px;"><div class="panel-title">Class Imbalance Mitigation</div><div class="panel-subtitle">A posteriori decision threshold calibration to maximize F1-Score without altering original data distributions (No SMOTE used).</div></div>', unsafe_allow_html=True)
        
        col_calib_img, col_calib_text = st.columns([1.5, 1])
        
        with col_calib_img:
            st.image(CALIB_PLOT_PATH, use_container_width=True)
            
        with col_calib_text:
            st.markdown(f"""
            <div style="background-color: #f8fbff; padding: 24px; border-left: 5px solid #10b981; border-radius: 0 8px 8px 0; height: 100%;">
                <div style="font-size: 1.3rem; font-weight: 900; color: #047857; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 0.5px;">Engineering Approach</div>
                <div style="color: #334155; font-size: 1.5 rem; line-height: 1.7;">
                    Instead of using controversial synthetic oversampling techniques (like SMOTE) which might distort physical petrophysical relationships, 
                    this framework handles the severe reservoir vs. non-reservoir class imbalance via <b>Post-Training Threshold Calibration</b>. <br><br>
                    By systematically scanning decision boundaries on the validation set, the system dynamically shifts the cutoff from the default 0.50 to the optimal point that maximizes the F1-Score. 
                    This ensures optimal sensitivity to potential CCS targets while preserving the integrity of the original geological dataset.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="panel-header" style="margin-top: 40px; margin-bottom: 24px;"><div class="panel-title">Global Feature Interpretability</div><div class="panel-subtitle">Basin-scale SHAP explanations.</div></div>', unsafe_allow_html=True)
    
    c_shap1, c_shap2 = st.columns(2)
    with c_shap1:
        if os.path.exists(SHAP_SUMMARY_PATH): 
            st.image(SHAP_SUMMARY_PATH, caption="Global SHAP Summary", use_container_width=True)
        else: 
            st.info("SHAP Summary missing.")
    with c_shap2:
        if os.path.exists(SHAP_DEP_PATH): 
            st.image(SHAP_DEP_PATH, caption="SHAP Feature Dependence", use_container_width=True)
        else: 
            st.info("SHAP Dependence missing.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_force, c_bar = st.columns(2)

    with c_force:
        if os.path.exists(SHAP_FORCE_PATH):
            st.image(SHAP_FORCE_PATH, caption="Local SHAP Force Plot", use_container_width=True)
        else:
            st.info("SHAP Force Plot missing.")

    with c_bar:
        if os.path.exists(SHAP_BAR_PATH):
            st.image(SHAP_BAR_PATH, caption="Global SHAP Bar Plot", use_container_width=True)
        else:
            st.info("SHAP Bar Plot missing.")

    st.markdown('<div class="panel-header" style="margin-top: 40px; margin-bottom: 24px;"><div class="panel-title">Continuous Well-log Depth Trace Verification</div></div>', unsafe_allow_html=True)
    
    if os.path.exists(DEPTH_PLOT_PATH):
        st.image(DEPTH_PLOT_PATH, use_container_width=True)
        st.markdown("<p style='font-size:1.1rem; color:#64748b; text-align:center;'><i>Green intervals indicate continuous reservoir candidate zones successfully captured by the model across deep geological formations.</i></p>", unsafe_allow_html=True)
    else: 
        st.error(f"Depth Plot not found at: {DEPTH_PLOT_PATH}")

if st.session_state.current_page != "Home":
    st.markdown("""
    <div class="formal-footer">
        <div class="footer-line-1">
            © 2026 MSc Dissertation Project. 
        </div>
        <div class="footer-line-2">
            <b>Author:</b> Hardy &nbsp;|&nbsp; ✉️ zeng.hongyun@student.zy.cdut.edu.cn &nbsp;|&nbsp; 📞 +86 15228036712
        </div>
    </div>
    """, unsafe_allow_html=True)