import os
import textwrap
import yaml
import re
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px

from openai import OpenAI
import google.generativeai as genai
import anthropic


# =========================
# ---- GLOBAL CONSTANTS ----
# =========================

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

MODEL_PROVIDER_MAP = {
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4.1-mini": ("openai", "gpt-4.1-mini"),
    "gemini-2.5-flash": ("google", "gemini-2.5-flash"),
    "gemini-2.5-flash-lite": ("google", "gemini-2.5-flash-lite"),
    # New: flash-preview for dataset prompting
    "gemini-3-flash-preview": ("google", "gemini-3-flash-preview"),
    # Kept for backward compatibility with agents.yaml
    "gemini-3-pro-preview": ("google", "gemini-3-pro-preview"),
    "claude-3-5-sonnet-latest": ("anthropic", "claude-3-5-sonnet-latest"),
    "claude-3-5-haiku-latest": ("anthropic", "claude-3-5-haiku-latest"),
    "grok-4-fast-reasoning": ("grok", "grok-4-fast-reasoning"),
    "grok-3-mini": ("grok", "grok-3-mini"),
}

DEFAULT_MAX_TOKENS = 12000

PAINTER_STYLES = {
    "Monet Impression": {"bg": "#f5f7fb", "primary": "#2f5fb3", "accent": "#ffaf40"},
    "Van Gogh Starry": {"bg": "#0b1736", "primary": "#ffd447", "accent": "#2aa9ff"},
    "Picasso Cubism": {"bg": "#faf4ef", "primary": "#ff6f61", "accent": "#2e294e"},
    "Dali Surreal": {"bg": "#fdf6e3", "primary": "#586e75", "accent": "#b58900"},
    "Kandinsky Abstract": {"bg": "#0f0f1a", "primary": "#ffcc00", "accent": "#ff4b5c"},
    "Matisse Cutouts": {"bg": "#fff7e6", "primary": "#004e64", "accent": "#ff6f61"},
    "Hokusai Wave": {"bg": "#e0f4ff", "primary": "#003f5c", "accent": "#ffa600"},
    "Frida Kahlo": {"bg": "#fff0f3", "primary": "#780000", "accent": "#c1121f"},
    "Rothko Fields": {"bg": "#1b1b2f", "primary": "#e43f5a", "accent": "#162447"},
    "Pollock Drips": {"bg": "#f8f9fa", "primary": "#1d3557", "accent": "#e63946"},
    "Da Vinci Classic": {"bg": "#f5f0e6", "primary": "#3e3a36", "accent": "#b8860b"},
    "Rembrandt Chiaroscuro": {"bg": "#121212", "primary": "#f6d365", "accent": "#fda085"},
    "Vermeer Light": {"bg": "#f2f7ff", "primary": "#274c77", "accent": "#f4a261"},
    "Turner Atmosphere": {"bg": "#fffaf0", "primary": "#2a4d69", "accent": "#f76c6c"},
    "Bauhaus Minimal": {"bg": "#ffffff", "primary": "#111827", "accent": "#f59e0b"},
    "Neo Tokyo": {"bg": "#050816", "primary": "#22d3ee", "accent": "#a855f7"},
    "Cyber Grid": {"bg": "#020617", "primary": "#22c55e", "accent": "#e11d48"},
    "Ink Wash": {"bg": "#f9fafb", "primary": "#111827", "accent": "#6b7280"},
    "Fauvism Bold": {"bg": "#fff7ed", "primary": "#b91c1c", "accent": "#2563eb"},
    "Nordic Calm": {"bg": "#f3f4f6", "primary": "#111827", "accent": "#10b981"},
}

UI_STRINGS = {
    "en": {
        "title": "Agentic Medical Device TPLC Dashboard",
        "sidebar_settings": "Global Settings",
        "language": "Language",
        "theme": "Theme",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "style": "Visual Style (Famous Painters)",
        "style_jackpot": "Jackpot Style",
        "api_keys": "API Keys & Status",
        "openai_status": "OpenAI",
        "gemini_status": "Gemini",
        "anthropic_status": "Anthropic",
        "grok_status": "Grok",
        "enter_api_key": "Enter API Key",
        "dashboard_tab": "WOW Dashboard",
        "agent_lab_tab": "Agent Lab",
        "note_keeper_tab": "AI Note Keeper",
        "config_tab": "Configuration",
        "data_manager": "Data Manager",
        "upload_gudid": "Upload GUDID CSV",
        "upload_510k": "Upload 510(k) CSV",
        "upload_classification": "Upload Classification CSV",
        "upload_safety": "Upload Safety Notice CSV",
        "upload_recall": "Upload Recall CSV",
        "wow_indicators": "WOW Status Indicators",
        "total_recalls": "Total Active Recalls",
        "avg_days_clearance": "Avg Days to Clearance",
        "high_risk_devices": "High-Risk Device Count",
        "agents": "Agents",
        "select_agent": "Select Agent",
        "agent_system_prompt": "Agent System Prompt (session editable)",
        "agent_model": "Model",
        "agent_temperature": "Temperature",
        "agent_max_tokens": "Max Tokens",
        "agent_user_prompt": "User Prompt / Query",
        "run_agent": "Run Agent",
        "agent_pipeline": "Agent Pipeline (Sequential Execution)",
        "use_as_next": "Use as Input to Next Agent",
        "view_mode": "View Mode",
        "view_text": "Text",
        "view_markdown": "Markdown",
        "note_input": "Paste / Type Your Note",
        "note_markdown": "Transformed Markdown",
        "transform_to_md": "Transform to Markdown",
        "edit_md_source": "Edit Markdown Source",
        "ai_formatting": "AI Formatting",
        "ai_keywords": "AI Keywords",
        "keywords_label": "Keywords (comma-separated)",
        "keyword_color": "Keyword Highlight Color",
        "apply_keywords": "Apply Keyword Highlight",
        "ai_entities": "AI Entities (20 with context)",
        "generate_entities": "Generate Entity Table",
        "ai_chat": "AI Chat (on Note)",
        "chat_prompt": "Chat Prompt",
        "ai_summary": "AI Summary",
        "summary_prompt": "Summary Prompt",
        "ai_magics": "AI Magics",
        "magic_1": "AI Risk Scenarist",
        "magic_2": "AI Regulatory Checklist",
        "config_agents": "agents.yaml Editor",
        "config_skills": "SKILL.md Editor",
        "save_config": "Apply to Session",
        "download": "Download",
        "upload": "Upload",
        "api_connected": "Connected",
        "api_missing": "Missing",
        # New: dataset preview and dataset chat
        "dataset_preview_section": "Data Explorer & Dataset Chat",
        "dataset_select": "Select Dataset",
        "dataset_510k": "510(k) List",
        "dataset_gudid": "GUDID Dataset",
        "dataset_class": "Classification Dataset",
        "dataset_safety": "Safety Notice List",
        "dataset_recall": "Recall List",
        "dataset_master": "Master Linked View",
        "dataset_rows": "Rows",
        "dataset_cols": "Columns",
        "dataset_prompt_header": "Ask AI About This Dataset",
        "dataset_prompt_input": "Question / Task (specific to this dataset)",
        "dataset_model": "Model (for dataset analysis)",
        "dataset_run": "Run Dataset Analysis",
        "dataset_empty": "This dataset is currently empty.",
    },
    "zh": {
        "title": "代理式醫療器材 TPLC 儀表板",
        "sidebar_settings": "全域設定",
        "language": "介面語言",
        "theme": "主題模式",
        "theme_light": "亮色",
        "theme_dark": "暗色",
        "style": "視覺風格（名畫家主題）",
        "style_jackpot": "風格 Jackpot",
        "api_keys": "API 金鑰與狀態",
        "openai_status": "OpenAI 狀態",
        "gemini_status": "Gemini 狀態",
        "anthropic_status": "Anthropic 狀態",
        "grok_status": "Grok 狀態",
        "enter_api_key": "輸入 API 金鑰",
        "dashboard_tab": "WOW 儀表板",
        "agent_lab_tab": "代理實驗室",
        "note_keeper_tab": "AI 筆記管家",
        "config_tab": "組態管理",
        "data_manager": "資料管理",
        "upload_gudid": "上傳 GUDID CSV",
        "upload_510k": "上傳 510(k) CSV",
        "upload_classification": "上傳分類代碼 CSV",
        "upload_safety": "上傳安全公告 CSV",
        "upload_recall": "上傳回收列表 CSV",
        "wow_indicators": "WOW 狀態指標",
        "total_recalls": "有效回收案件數",
        "avg_days_clearance": "平均核准天數",
        "high_risk_devices": "高風險醫材數量",
        "agents": "代理人",
        "select_agent": "選擇代理",
        "agent_system_prompt": "代理系統提示（本次工作階段可編輯）",
        "agent_model": "模型",
        "agent_temperature": "溫度",
        "agent_max_tokens": "最大 Token 數",
        "agent_user_prompt": "使用者提問／任務說明",
        "run_agent": "執行代理",
        "agent_pipeline": "代理流水線（逐一執行）",
        "use_as_next": "作為下一個代理輸入",
        "view_mode": "檢視模式",
        "view_text": "純文字",
        "view_markdown": "Markdown",
        "note_input": "貼上／輸入原始筆記",
        "note_markdown": "轉換後 Markdown",
        "transform_to_md": "轉換為 Markdown",
        "edit_md_source": "編輯 Markdown 原始碼",
        "ai_formatting": "AI 格式優化",
        "ai_keywords": "AI 關鍵字標色",
        "keywords_label": "關鍵字（以逗號分隔）",
        "keyword_color": "關鍵字顏色",
        "apply_keywords": "套用關鍵字標註",
        "ai_entities": "AI 實體萃取（20 個含脈絡）",
        "generate_entities": "產生實體表格",
        "ai_chat": "AI 對話（針對筆記內容）",
        "chat_prompt": "對話提示",
        "ai_summary": "AI 摘要",
        "summary_prompt": "摘要提示",
        "ai_magics": "AI Magics",
        "magic_1": "AI 風險情境設計師",
        "magic_2": "AI 法規檢查清單",
        "config_agents": "agents.yaml 編輯器",
        "config_skills": "SKILL.md 編輯器",
        "save_config": "套用至本次工作階段",
        "download": "下載",
        "upload": "上傳",
        "api_connected": "已連線",
        "api_missing": "尚未設定",
        # New: dataset preview and dataset chat
        "dataset_preview_section": "資料總覽與資料集 AI 對話",
        "dataset_select": "選擇資料集",
        "dataset_510k": "510(k) 清單",
        "dataset_gudid": "GUDID 資料集",
        "dataset_class": "分類代碼資料集",
        "dataset_safety": "安全公告清單",
        "dataset_recall": "回收清單",
        "dataset_master": "主連結檢視（Master View）",
        "dataset_rows": "筆數",
        "dataset_cols": "欄位數",
        "dataset_prompt_header": "針對此資料集提問 AI",
        "dataset_prompt_input": "與此資料集相關的問題／任務說明",
        "dataset_model": "使用模型（資料集分析）",
        "dataset_run": "執行資料集分析",
        "dataset_empty": "此資料集目前為空。",
    },
}


# =========================
# ---- HELPERS & STATE ----
# =========================

def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, key)


def init_session_state():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "en"
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"
    if "style_name" not in st.session_state:
        st.session_state["style_name"] = list(PAINTER_STYLES.keys())[0]
    if "agents_config" not in st.session_state:
        st.session_state["agents_config"] = load_agents_yaml()
    if "skills_text" not in st.session_state:
        st.session_state["skills_text"] = load_skills_markdown()
    if "skills_map" not in st.session_state:
        st.session_state["skills_map"] = parse_skills(st.session_state["skills_text"])
    if "llm_orchestrator" not in st.session_state:
        st.session_state["llm_orchestrator"] = LLMOrchestrator()
    if "data_manager" not in st.session_state:
        st.session_state["data_manager"] = DataManager()
    if "agent_pipeline" not in st.session_state:
        st.session_state["agent_pipeline"] = []
    if "note_raw" not in st.session_state:
        st.session_state["note_raw"] = ""
    if "note_md" not in st.session_state:
        st.session_state["note_md"] = ""
    if "note_md_edit_mode" not in st.session_state:
        st.session_state["note_md_edit_mode"] = False
    if "openai_key_ui" not in st.session_state:
        st.session_state["openai_key_ui"] = ""
    if "gemini_key_ui" not in st.session_state:
        st.session_state["gemini_key_ui"] = ""
    if "anthropic_key_ui" not in st.session_state:
        st.session_state["anthropic_key_ui"] = ""
    if "grok_key_ui" not in st.session_state:
        st.session_state["grok_key_ui"] = ""


def apply_theme():
    style_name = st.session_state.get("style_name", list(PAINTER_STYLES.keys())[0])
    palette = PAINTER_STYLES.get(style_name, list(PAINTER_STYLES.values())[0])
    bg = palette["bg"]
    primary = palette["primary"]
    accent = palette["accent"]

    base_text = "#111827" if st.session_state.get("theme_mode") == "light" else "#e5e7eb"
    card_bg = "rgba(255,255,255,0.85)" if st.session_state.get("theme_mode") == "light" else "rgba(17,24,39,0.9)"

    css = f"""
    <style>
    .stApp {{
        background: {bg};
        color: {base_text};
    }}
    .wow-card {{
        background: {card_bg};
        border-radius: 1rem;
        padding: 1rem 1.3rem;
        border: 1px solid rgba(148,163,184,0.4);
        box-shadow: 0 12px 30px rgba(15,23,42,0.15);
    }}
    .wow-metric-label {{
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    .wow-metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {primary};
    }}
    .wow-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.7rem;
        margin-left: 0.3rem;
        background: {accent}20;
        color: {accent};
        border: 1px solid {accent}40;
    }}
    .wow-dot-green {{
        height: 0.6rem;
        width: 0.6rem;
        border-radius: 999px;
        background: #22c55e;
        margin-right: 0.25rem;
    }}
    .wow-dot-red {{
        height: 0.6rem;
        width: 0.6rem;
        border-radius: 999px;
        background: #ef4444;
        margin-right: 0.25rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =======================
# ---- DATA MANAGER  ----
# =======================

class DataManager:
    def __init__(self):
        self.df_510k, self.df_gudid, self.df_class, self.df_safety, self.df_recall = self._create_mock_datasets()
        self.df_master = self._build_master_view()

    def _create_mock_datasets(self):
        # Mock 510(k)
        df_510k = pd.DataFrame([
            {"k_number": "K240001", "device_name": "CardioFlow Stent", "applicant": "Alpha Cardio Inc.",
             "product_code": "MNA", "decision_date": "2024-01-15", "device_class": "II", "specialty": "Cardiovascular"},
            {"k_number": "K240050", "device_name": "OrthoFlex Knee System", "applicant": "Beta Ortho Corp.",
             "product_code": "JWH", "decision_date": "2024-03-22", "device_class": "II", "specialty": "Orthopedic"},
            {"k_number": "K230900", "device_name": "NeuroWave Stimulator", "applicant": "NeuroX Ltd.",
             "product_code": "GZB", "decision_date": "2023-11-10", "device_class": "III", "specialty": "Neurology"},
        ])
        # Mock GUDID
        df_gudid = pd.DataFrame([
            {"primary_di": "00812345000011", "public_device_record_key": "GUDID001",
             "device_description": "CardioFlow Stent 3.0mm", "company_name": "Alpha Cardio Inc.",
             "gmdn_code": "47915", "k_number": "K240001"},
            {"primary_di": "00812345000022", "public_device_record_key": "GUDID002",
             "device_description": "OrthoFlex Knee Femoral Component", "company_name": "Beta Ortho Corp.",
             "gmdn_code": "37822", "k_number": "K240050"},
        ])
        # Mock Classification
        df_class = pd.DataFrame([
            {"product_code": "MNA", "device_class": "II", "specialty": "Cardiovascular",
             "device_definition": "Coronary stent system"},
            {"product_code": "JWH", "device_class": "II", "specialty": "Orthopedic",
             "device_definition": "Knee joint prosthesis"},
            {"product_code": "GZB", "device_class": "III", "specialty": "Neurology",
             "device_definition": "Neurostimulation system"},
        ])
        # Mock Safety Notices
        df_safety = pd.DataFrame([
            {"notice_id": "SN2024-001", "k_number": "K240001", "severity": "Urgent",
             "summary": "Potential fracture of stent struts in certain lots.",
             "date": "2024-06-10"},
            {"notice_id": "SN2023-015", "k_number": "K230900", "severity": "Routine",
             "summary": "Updated instructions for MRI safety labeling.",
             "date": "2023-12-05"},
        ])
        # Mock Recalls
        df_recall = pd.DataFrame([
            {"res_event_number": "RE2024-1001", "k_number": "K240001",
             "root_cause_text": "Material fatigue leading to stent fracture.",
             "classification": "Class II", "distribution_pattern": "US Nationwide",
             "year": 2024},
            {"res_event_number": "RE2023-2001", "k_number": "K230900",
             "root_cause_text": "Software error may lead to overstimulation.",
             "classification": "Class I", "distribution_pattern": "US & EU", "year": 2023},
        ])
        # parse dates
        df_510k["decision_date"] = pd.to_datetime(df_510k["decision_date"])
        df_safety["date"] = pd.to_datetime(df_safety["date"])
        return df_510k, df_gudid, df_class, df_safety, df_recall

    def _build_master_view(self):
        df = self.df_recall.merge(self.df_510k, on="k_number", how="left", suffixes=("_recall", "_510k"))
        df = df.merge(self.df_class[["product_code", "device_definition"]], on="product_code", how="left")
        return df

    def update_from_upload(self, kind: str, df: pd.DataFrame):
        if kind == "510k":
            self.df_510k = df
        elif kind == "gudid":
            self.df_gudid = df
        elif kind == "class":
            self.df_class = df
        elif kind == "safety":
            self.df_safety = df
        elif kind == "recall":
            self.df_recall = df
        self.df_master = self._build_master_view()

    def compute_indicators(self):
        total_recalls = len(self.df_recall)
        avg_days = None
        if "decision_date" in self.df_510k.columns and not self.df_510k["decision_date"].isna().all():
            # mock: days between decision and "today" proxy recall year
            avg_days = 120
        high_risk = len(self.df_510k[self.df_510k["device_class"] == "III"]) if "device_class" in self.df_510k else 0
        return total_recalls, avg_days, high_risk


# =========================
# ---- AGENTS & SKILLS ----
# =========================

def load_agents_yaml() -> Dict[str, Any]:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}  # HF Space owner should provide agents.yaml


def load_skills_markdown() -> str:
    try:
        with open("SKILL.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "# Skills\n\n"


def parse_skills(md_text: str) -> Dict[str, str]:
    """
    Parse SKILL.md into {skill_id: instruction_text}
    Assumes sections start with '## skill: <id>'
    """
    skills = {}
    current_id = None
    buffer: List[str] = []
    for line in md_text.splitlines():
        m = re.match(r"^##\s+skill:\s*([A-Za-z0-9_\-]+)", line.strip())
        if m:
            if current_id:
                skills[current_id] = "\n".join(buffer).strip()
                buffer = []
            current_id = m.group(1).strip()
        else:
            if current_id:
                buffer.append(line)
    if current_id:
        skills[current_id] = "\n".join(buffer).strip()
    return skills


def build_agent_prompt(agent_key: str,
                       agents_cfg: Dict[str, Any],
                       skills_map: Dict[str, str],
                       system_override: Optional[str] = None,
                       extra_instruction: str = "") -> str:
    agent_cfg = agents_cfg.get(agent_key, {})
    base_system = system_override or agent_cfg.get("system_prompt", "")
    skill_ids = agent_cfg.get("skills", []) or []
    skill_texts = []
    for sid in skill_ids:
        if sid in skills_map:
            skill_texts.append(f"[skill: {sid}]\n{skills_map[sid]}")
    full = base_system.strip() + "\n\n" + "\n\n".join(skill_texts).strip()
    if extra_instruction:
        full += "\n\n[session instruction]\n" + extra_instruction.strip()
    return full.strip()


# =========================
# ---- LLM ORCHESTRATOR ---
# =========================

class LLMOrchestrator:
    def __init__(self):
        self._openai_clients: Dict[str, OpenAI] = {}
        self._anthropic_clients: Dict[str, anthropic.Anthropic] = {}
        self._gemini_configured_keys: Dict[str, bool] = {}

    # ---- API key resolution ----
    def _get_openai_client(self):
        key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key_ui")
        if not key:
            raise RuntimeError("Missing OpenAI API key")
        if key not in self._openai_clients:
            self._openai_clients[key] = OpenAI(api_key=key)
        return self._openai_clients[key]

    def _ensure_gemini(self):
        key = os.getenv("GEMINI_API_KEY") or st.session_state.get("gemini_key_ui")
        if not key:
            raise RuntimeError("Missing Gemini API key")
        if key not in self._gemini_configured_keys:
            genai.configure(api_key=key)
            self._gemini_configured_keys[key] = True
        return key

    def _get_anthropic_client(self):
        key = os.getenv("ANTHROPIC_API_KEY") or st.session_state.get("anthropic_key_ui")
        if not key:
            raise RuntimeError("Missing Anthropic API key")
        if key not in self._anthropic_clients:
            self._anthropic_clients[key] = anthropic.Anthropic(api_key=key)
        return self._anthropic_clients[key]

    # ---- Public entry point ----
    def call_llm(self,
                 model: str,
                 system_prompt: str,
                 user_prompt: str,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.2) -> str:
        provider, provider_model = MODEL_PROVIDER_MAP[model]
        if provider == "openai":
            return self._call_openai(provider_model, system_prompt, user_prompt, max_tokens, temperature)
        elif provider == "google":
            return self._call_gemini(provider_model, system_prompt, user_prompt, max_tokens, temperature)
        elif provider == "anthropic":
            return self._call_anthropic(provider_model, system_prompt, user_prompt, max_tokens, temperature)
        elif provider == "grok":
            # Placeholder: provide your own implementation for Grok / xAI
            raise RuntimeError("Grok provider is not implemented in this demo. Please plug in your own client.")
        else:
            raise RuntimeError(f"Unknown provider: {provider}")

    # ---- Provider-specific calls ----
    def _call_openai(self, model: str, system_prompt: str, user_prompt: str,
                     max_tokens: int, temperature: float) -> str:
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    def _call_gemini(self, model: str, system_prompt: str, user_prompt: str,
                     max_tokens: int, temperature: float) -> str:
        self._ensure_gemini()
        full_prompt = f"{system_prompt}\n\n[User]\n{user_prompt}"
        gmodel = genai.GenerativeModel(model)
        resp = gmodel.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text

    def _call_anthropic(self, model: str, system_prompt: str, user_prompt: str,
                        max_tokens: int, temperature: float) -> str:
        client = self._get_anthropic_client()
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # Concatenate text blocks
        chunks = []
        for c in msg.content:
            if getattr(c, "type", None) == "text":
                chunks.append(c.text)
        return "\n".join(chunks)


# =========================
# ---- UI SECTIONS --------
# =========================

def render_sidebar():
    st.sidebar.header(t("sidebar_settings"))

    # Language
    lang = st.sidebar.selectbox(
        t("language"),
        options=[("en", "English"), ("zh", "繁體中文")],
        format_func=lambda x: x[1],
        index=1 if st.session_state["lang"] == "zh" else 0,
    )
    st.session_state["lang"] = lang[0]

    # Theme
    theme_mode = st.sidebar.radio(
        t("theme"),
        options=["light", "dark"],
        index=0 if st.session_state["theme_mode"] == "light" else 1,
        format_func=lambda x: t("theme_light") if x == "light" else t("theme_dark"),
    )
    st.session_state["theme_mode"] = theme_mode

    # Painter style & Jackpot
    st.sidebar.subheader(t("style"))
    style_options = list(PAINTER_STYLES.keys())
    current_index = style_options.index(st.session_state["style_name"]) if st.session_state["style_name"] in style_options else 0
    style_sel = st.sidebar.selectbox(
        " ",
        options=style_options,
        index=current_index,
    )
    st.session_state["style_name"] = style_sel
    if st.sidebar.button(t("style_jackpot")):
        import random
        st.session_state["style_name"] = random.choice(style_options)

    st.sidebar.markdown("---")
    st.sidebar.subheader(t("data_manager"))

    # File uploaders
    dm: DataManager = st.session_state["data_manager"]
    up_510k = st.sidebar.file_uploader(t("upload_510k"), type=["csv"], key="510k_upl")
    if up_510k is not None:
        df = pd.read_csv(up_510k)
        dm.update_from_upload("510k", df)

    up_gudid = st.sidebar.file_uploader(t("upload_gudid"), type=["csv"], key="gudid_upl")
    if up_gudid is not None:
        df = pd.read_csv(up_gudid)
        dm.update_from_upload("gudid", df)

    up_class = st.sidebar.file_uploader(t("upload_classification"), type=["csv"], key="class_upl")
    if up_class is not None:
        df = pd.read_csv(up_class)
        dm.update_from_upload("class", df)

    up_safety = st.sidebar.file_uploader(t("upload_safety"), type=["csv"], key="safety_upl")
    if up_safety is not None:
        df = pd.read_csv(up_safety)
        dm.update_from_upload("safety", df)

    up_recall = st.sidebar.file_uploader(t("upload_recall"), type=["csv"], key="recall_upl")
    if up_recall is not None:
        df = pd.read_csv(up_recall)
        dm.update_from_upload("recall", df)

    st.sidebar.markdown("---")
    st.sidebar.subheader(t("api_keys"))

    def api_status_row(label_key, env_name, ui_state_key):
        env_val = os.getenv(env_name)
        if env_val:
            # show connected badge only, do not show key
            st.sidebar.markdown(
                f"<div class='wow-badge'><div class='wow-dot-green'></div>{t(label_key)}: {t('api_connected')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                f"<div class='wow-badge'><div class='wow-dot-red'></div>{t(label_key)}: {t('api_missing')}</div>",
                unsafe_allow_html=True,
            )
            st.sidebar.text_input(
                f"{t(label_key)} - {t('enter_api_key')}",
                type="password",
                key=ui_state_key,
            )

    api_status_row("openai_status", "OPENAI_API_KEY", "openai_key_ui")
    api_status_row("gemini_status", "GEMINI_API_KEY", "gemini_key_ui")
    api_status_row("anthropic_status", "ANTHROPIC_API_KEY", "anthropic_key_ui")
    api_status_row("grok_status", "GROK_API_KEY", "grok_key_ui")


def render_dashboard():
    st.subheader(t("wow_indicators"))
    dm: DataManager = st.session_state["data_manager"]
    total_recalls, avg_days, high_risk = dm.compute_indicators()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='wow-metric-label'>{t('total_recalls')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='wow-metric-value'>{total_recalls}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='wow-metric-label'>{t('avg_days_clearance')}</div>", unsafe_allow_html=True)
        value = f"{avg_days} d" if avg_days is not None else "N/A"
        st.markdown(f"<div class='wow-metric-value'>{value}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='wow-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='wow-metric-label'>{t('high_risk_devices')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='wow-metric-value'>{high_risk}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### TPLC Overview")

    if not dm.df_master.empty:
        # Recalls by classification
        fig1 = px.bar(
            dm.df_master.groupby("classification")["res_event_number"].count().reset_index(),
            x="classification",
            y="res_event_number",
            title="Recalls by Classification",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Recalls by specialty
        if "specialty" in dm.df_master.columns:
            fig2 = px.bar(
                dm.df_master.groupby("specialty")["res_event_number"].count().reset_index(),
                x="specialty",
                y="res_event_number",
                title="Recalls by Specialty",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Timeline mock (decision date vs recall year)
        if "decision_date" in dm.df_master.columns:
            tmp = dm.df_master.dropna(subset=["decision_date"]).copy()
            if not tmp.empty:
                tmp["decision_year"] = tmp["decision_date"].dt.year
                fig3 = px.scatter(
                    tmp,
                    x="decision_year",
                    y="year",
                    color="device_name",
                    title="Decision Year vs Recall Year",
                )
                st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No master data available. Please upload datasets or use mock data.")

    # NEW: Dataset Explorer & Dataset Chat
    render_dataset_explorer()


def render_dataset_explorer():
    st.markdown("---")
    st.markdown(f"### {t('dataset_preview_section')}")

    dm: DataManager = st.session_state["data_manager"]

    dataset_options = [
        ("510k", t("dataset_510k")),
        ("gudid", t("dataset_gudid")),
        ("class", t("dataset_class")),
        ("safety", t("dataset_safety")),
        ("recall", t("dataset_recall")),
        ("master", t("dataset_master")),
    ]
    sel_key = st.selectbox(
        t("dataset_select"),
        options=[opt[0] for opt in dataset_options],
        format_func=lambda k: dict(dataset_options)[k],
        key="dataset_select_key",
    )

    # Map key to actual DataFrame
    if sel_key == "510k":
        df = dm.df_510k
    elif sel_key == "gudid":
        df = dm.df_gudid
    elif sel_key == "class":
        df = dm.df_class
    elif sel_key == "safety":
        df = dm.df_safety
    elif sel_key == "recall":
        df = dm.df_recall
    else:
        df = dm.df_master

    # Preview
    if df is None or df.empty:
        st.warning(t("dataset_empty"))
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{t('dataset_rows')}:** {len(df)}")
    with c2:
        st.markdown(f"**{t('dataset_cols')}:** {len(df.columns)}")

    st.dataframe(df.head(50), use_container_width=True)

    # Dataset-specific prompt
    st.markdown(f"#### {t('dataset_prompt_header')}")

    ds_models = ["gpt-4o-mini", "gemini-2.5-flash", "gemini-3-flash-preview"]
    ds_model = st.selectbox(
        t("dataset_model"),
        options=ds_models,
        index=0,
        key="dataset_model_select",
    )

    question = st.text_area(
        t("dataset_prompt_input"),
        height=160,
        key="dataset_prompt_input",
    )

    if st.button(t("dataset_run"), key="dataset_run_btn"):
        orchestrator: LLMOrchestrator = st.session_state["llm_orchestrator"]
        try:
            # Build schema and sample context
            schema_str = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
            # Use up to 20 rows as context
            sample_df = df.head(20)
            try:
                sample_md = sample_df.to_markdown(index=False)
            except Exception:
                sample_md = sample_df.to_string(index=False)

            system_prompt = textwrap.dedent(f"""
            你是一位專精 FDA 醫療器材 TPLC 資料的分析顧問。
            你現在看到的是單一資料集的結構與前幾列樣本，請根據這個資料集來回答問題或提出分析建議。
            - 優先解釋你如何利用欄位與樣本來推論
            - 若資料不足以支持某結論，請明確說明不確定性
            - 若需要，提供 pandas 分析步驟構想，但不要假設可以直接執行程式

            [DATASET SCHEMA]
            {schema_str}

            [DATASET SAMPLE]
            {sample_md}
            """).strip()

            user_prompt = question or "請根據這個資料集，說明可以進行哪些有意義的 TPLC 分析。"

            out = orchestrator.call_llm(
                model=ds_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=3000,
                temperature=0.3,
            )
            st.markdown("##### AI Output")
            st.markdown(out)
        except Exception as e:
            st.error(f"Dataset analysis error: {e}")


def render_agent_lab():
    st.subheader(t("agents"))
    agents_cfg = st.session_state["agents_config"] or {}
    skills_map = st.session_state["skills_map"]
    orchestrator: LLMOrchestrator = st.session_state["llm_orchestrator"]

    if not agents_cfg:
        st.warning("No agents loaded from agents.yaml.")
        return

    agent_keys = list(agents_cfg.keys())
    agent_labels = [f"{k} — {agents_cfg[k].get('name', '')}" for k in agent_keys]
    selected_idx = st.selectbox(
        t("select_agent"),
        options=list(range(len(agent_keys))),
        format_func=lambda i: agent_labels[i],
    )
    agent_key = agent_keys[selected_idx]
    agent = agents_cfg[agent_key]

    with st.expander(t("agent_system_prompt"), expanded=True):
        sys_prompt_session = st.text_area(
            "",
            value=agent.get("system_prompt", ""),
            height=160,
            key=f"sys_prompt_{agent_key}",
        )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        model_default = agent.get("model_name", "gpt-4o-mini")
        if model_default not in SUPPORTED_MODELS:
            model_default = "gpt-4o-mini"
        model_sel = st.selectbox(
            t("agent_model"),
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(model_default),
            key=f"agent_model_{agent_key}",
        )
    with c2:
        temperature = st.slider(
            t("agent_temperature"),
            min_value=0.0,
            max_value=1.0,
            value=float(agent.get("temperature", 0.2)),
            step=0.05,
            key=f"agent_temp_{agent_key}",
        )
    with c3:
        max_tokens = st.number_input(
            t("agent_max_tokens"),
            min_value=64,
            max_value=120000,
            value=DEFAULT_MAX_TOKENS,
            step=256,
            key=f"agent_maxtok_{agent_key}",
        )

    user_prompt = st.text_area(
        t("agent_user_prompt"),
        height=160,
        key=f"user_prompt_{agent_key}",
    )

    # Optionally chain from previous pipeline step
    pipeline = st.session_state["agent_pipeline"]
    if pipeline:
        st.markdown(f"#### {t('agent_pipeline')}")
        for idx, step in enumerate(pipeline):
            st.markdown(f"**[{idx}] {step['agent_label']}**")
        selected_pipeline_idx = st.selectbox(
            "Use previous output as context (optional)",
            options=["None"] + list(range(len(pipeline))),
            format_func=lambda x: "None" if x == "None" else f"Step {x}",
            key=f"pipeline_sel_{agent_key}",
        )
    else:
        selected_pipeline_idx = "None"

    if st.button(t("run_agent")):
        try:
            extra_instruction = ""
            context_text = ""
            if selected_pipeline_idx != "None":
                step = pipeline[int(selected_pipeline_idx)]
                context_text = step["output"]
                extra_instruction = "You are receiving prior agent output as context:\n" + context_text

            full_system_prompt = build_agent_prompt(
                agent_key,
                agents_cfg,
                skills_map,
                system_override=sys_prompt_session,
                extra_instruction=extra_instruction,
            )
            result = orchestrator.call_llm(
                model_sel,
                full_system_prompt,
                user_prompt,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            new_step = {
                "agent_key": agent_key,
                "agent_label": agents_cfg[agent_key].get("name", agent_key),
                "model": model_sel,
                "output": result,
            }
            st.session_state["agent_pipeline"].append(new_step)
        except Exception as e:
            st.error(f"Agent execution error: {e}")

    # Show pipeline with text/markdown toggle and "Use as next"
    if st.session_state["agent_pipeline"]:
        st.markdown(f"### {t('agent_pipeline')}")
        for i, step in enumerate(st.session_state["agent_pipeline"]):
            st.markdown(f"---\n**Step {i}: {step['agent_label']} ({step['model']})**")
            view_mode = st.radio(
                t("view_mode"),
                options=["text", "markdown"],
                index=1,
                key=f"view_mode_{i}",
                format_func=lambda x: t("view_text") if x == "text" else t("view_markdown"),
            )
            if view_mode == "markdown":
                st.markdown(step["output"])
            else:
                st.text_area(" ", step["output"], height=200, key=f"pipeline_output_{i}")
            if st.button(f"{t('use_as_next')} #{i}", key=f"use_next_{i}"):
                st.session_state[f"user_prompt_{agent_key}"] = step["output"]
                st.experimental_rerun()


def render_note_keeper():
    st.markdown("### " + t("note_keeper_tab"))
    orchestrator: LLMOrchestrator = st.session_state["llm_orchestrator"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### " + t("note_input"))
        st.session_state["note_raw"] = st.text_area(
            "",
            value=st.session_state["note_raw"],
            height=260,
            key="note_raw_input",
        )
        if st.button(t("transform_to_md")):
            try:
                system_prompt = textwrap.dedent("""
                你是一位專精醫療器材與法規寫作的筆記整理助手。
                將使用者提供的原始文字轉換為結構良好的 Markdown 筆記：
                - 使用有層次的標題（#、##、###）
                - 條列重點與行動項目
                - 保留重要術語與編號（如 K-number, product code）
                - 不要虛構事實，只重排與整理
                """)
                user_prompt = st.session_state["note_raw"]
                md = orchestrator.call_llm(
                    model="gpt-4o-mini",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=0.2,
                )
                st.session_state["note_md"] = md
                st.session_state["note_md_edit_mode"] = False
            except Exception as e:
                st.error(f"Markdown transform error: {e}")

    with c2:
        st.markdown("#### " + t("note_markdown"))
        st.checkbox(
            t("edit_md_source"),
            value=st.session_state["note_md_edit_mode"],
            key="note_md_edit_mode",
        )
        if st.session_state["note_md_edit_mode"]:
            st.session_state["note_md"] = st.text_area(
                "",
                value=st.session_state["note_md"],
                height=260,
                key="note_md_source",
            )
        else:
            if st.session_state["note_md"]:
                st.markdown(st.session_state["note_md"])
            else:
                st.info("No Markdown yet. Transform your note on the left side.")

    st.markdown("---")

    # ===== AI Formatting =====
    st.markdown("#### " + t("ai_formatting"))
    if st.button(t("ai_formatting")):
        try:
            system_prompt = textwrap.dedent("""
            你是 Markdown 排版與風格專家。
            目標：在保持語意不變的前提下，提升可讀性與一致性：
            - 合理使用標題層級與分節
            - 將列表、表格整理清楚
            - 對關鍵定義與警示增加粗體或引用區塊
            - 避免過度重寫內容
            """)
            user_prompt = st.session_state["note_md"] or st.session_state["note_raw"]
            out = orchestrator.call_llm(
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=0.2,
            )
            st.session_state["note_md"] = out
            st.session_state["note_md_edit_mode"] = False
        except Exception as e:
            st.error(f"AI formatting error: {e}")

    # ===== AI Keywords =====
    st.markdown("#### " + t("ai_keywords"))
    kw = st.text_input(t("keywords_label"), key="kw_input")
    color = st.color_picker(t("keyword_color"), value="#ff0000", key="kw_color")
    if st.button(t("apply_keywords")):
        note = st.session_state["note_md"] or st.session_state["note_raw"]
        if not note:
            st.warning("No note content available.")
        else:
            keywords = [k.strip() for k in kw.split(",") if k.strip()]
            colored = note
            for kword in keywords:
                if not kword:
                    continue
                colored = colored.replace(
                    kword,
                    f"<span style='background-color:{color}33; color:{color}; font-weight:bold;'>{kword}</span>",
                )
            st.markdown(colored, unsafe_allow_html=True)

    # ===== AI Entities =====
    st.markdown("#### " + t("ai_entities"))
    if st.button(t("generate_entities")):
        try:
            system_prompt = textwrap.dedent("""
            你是一位醫療器材 TPLC 實體萃取專家。
            根據以下筆記內容，擷取最多 20 個關鍵實體，輸出為 Markdown 表格，欄位包含：
            - Entity（實體名稱）
            - Type（類型，如 Manufacturer, Device, K-number, Product Code, Risk, Event, Regulation）
            - Context（摘錄的關鍵語句或簡短說明）
            - Importance（High / Medium / Low）
            僅使用文本中可推得的資訊，不要虛構。
            """)
            user_prompt = st.session_state["note_md"] or st.session_state["note_raw"]
            out = orchestrator.call_llm(
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2000,
                temperature=0.2,
            )
            st.markdown(out)
        except Exception as e:
            st.error(f"AI entities error: {e}")

    # ===== AI Chat & Summary =====
    st.markdown("---")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### " + t("ai_chat"))
        chat_model = st.selectbox(
            t("agent_model"),
            options=SUPPORTED_MODELS,
            index=0,
            key="note_chat_model",
        )
        chat_maxtok = st.number_input(
            t("agent_max_tokens"),
            min_value=64,
            max_value=120000,
            value=DEFAULT_MAX_TOKENS,
            step=256,
            key="note_chat_maxtok",
        )
        chat_prompt = st.text_area(
            t("chat_prompt"),
            value="請根據目前的筆記內容，回答我的問題。",
            height=120,
            key="note_chat_prompt",
        )
        if st.button("Run Chat", key="note_chat_run"):
            try:
                system_prompt = "你是一位專精 FDA 醫療器材 TPLC 的顧問，回答時可直接引用筆記內容。"
                user_prompt = f"=== NOTE ===\n{st.session_state['note_md'] or st.session_state['note_raw']}\n\n=== QUESTION ===\n{chat_prompt}"
                out = orchestrator.call_llm(
                    model=chat_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(chat_maxtok),
                    temperature=0.3,
                )
                st.markdown(out)
            except Exception as e:
                st.error(f"AI chat error: {e}")

    with c4:
        st.markdown("#### " + t("ai_summary"))
        sum_model = st.selectbox(
            t("agent_model"),
            options=SUPPORTED_MODELS,
            index=0,
            key="note_sum_model",
        )
        sum_maxtok = st.number_input(
            t("agent_max_tokens"),
            min_value=64,
            max_value=120000,
            value=DEFAULT_MAX_TOKENS,
            step=256,
            key="note_sum_maxtok",
        )
        sum_prompt = st.text_area(
            t("summary_prompt"),
            value="請用系統化方式摘要此筆記，拆分為：背景、關鍵風險、已知事件、建議行動。",
            height=120,
            key="note_sum_prompt",
        )
        if st.button("Run Summary", key="note_sum_run"):
            try:
                system_prompt = "你是一位醫療器材風險與法規摘要專家。"
                user_prompt = f"=== NOTE ===\n{st.session_state['note_md'] or st.session_state['note_raw']}\n\n=== SUMMARY INSTRUCTION ===\n{sum_prompt}"
                out = orchestrator.call_llm(
                    model=sum_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(sum_maxtok),
                    temperature=0.2,
                )
                st.markdown(out)
            except Exception as e:
                st.error(f"AI summary error: {e}")

    # ===== AI Magics (two additional features) =====
    st.markdown("---")
    st.markdown("### " + t("ai_magics"))

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("#### " + t("magic_1"))
        prompt_risk = st.text_area(
            "自訂情境說明（選填）",
            value="請基於筆記內容，設計 3-5 個合理的風險情境，並為每個情境給出：觸發條件、潛在後果、可能失效機制、建議控制措施。",
            height=120,
            key="magic1_prompt",
        )
        if st.button("Generate Risk Scenarios", key="magic1_run"):
            try:
                system_prompt = "你是一位醫療器材風險管理專家，擅長以 ISO 14971 思維設計風隻情境與控制措施。"
                user_prompt = f"=== NOTE ===\n{st.session_state['note_md'] or st.session_state['note_raw']}\n\n=== TASK ===\n{prompt_risk}"
                out = orchestrator.call_llm(
                    model="gpt-4o-mini",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=3000,
                    temperature=0.35,
                )
                st.markdown(out)
            except Exception as e:
                st.error(f"AI Risk Scenarist error: {e}")

    with m2:
        st.markdown("#### " + t("magic_2"))
        prompt_check = st.text_area(
            "自訂檢查重點（選填）",
            value="請將此筆記轉換為可操作的法規／合規檢查清單，每一項包含：檢查點、法規依據（若可推得）、優先層級、負責角色。",
            height=120,
            key="magic2_prompt",
        )
        if st.button("Generate Regulatory Checklist", key="magic2_run"):
            try:
                system_prompt = "你是一位 FDA 醫療器材法規與品質系統專家，熟悉 QSR、ISO 13485 與 TPLC 思維。"
                user_prompt = f"=== NOTE ===\n{st.session_state['note_md'] or st.session_state['note_raw']}\n\n=== TASK ===\n{prompt_check}"
                out = orchestrator.call_llm(
                    model="gpt-4o-mini",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=4000,
                    temperature=0.25,
                )
                st.markdown(out)
            except Exception as e:
                st.error(f"AI Regulatory Checklist error: {e}")


def render_config_tab():
    st.markdown("### " + t("config_tab"))

    tabs = st.tabs([t("config_agents"), t("config_skills")])

    # agents.yaml editor
    with tabs[0]:
        st.markdown("#### agents.yaml")
        agents_text = yaml.safe_dump(st.session_state["agents_config"], allow_unicode=True, sort_keys=False)
        new_text = st.text_area(
            "",
            value=agents_text,
            height=400,
            key="agents_yaml_editor",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(t("save_config"), key="agents_apply"):
                try:
                    new_cfg = yaml.safe_load(new_text) or {}
                    st.session_state["agents_config"] = new_cfg
                    st.success("agents.yaml applied to session.")
                except Exception as e:
                    st.error(f"YAML parse error: {e}")
        with c2:
            st.download_button(
                t("download"),
                data=new_text,
                file_name="agents.yaml",
                mime="text/yaml",
                key="agents_download_btn",
            )
        with c3:
            upload = st.file_uploader(t("upload"), type=["yaml", "yml"], key="agents_upload")
            if upload is not None:
                try:
                    up_cfg = yaml.safe_load(upload.read().decode("utf-8")) or {}
                    st.session_state["agents_config"] = up_cfg
                    st.success("Uploaded agents.yaml applied.")
                except Exception as e:
                    st.error(f"Upload parse error: {e}")

    # SKILL.md editor
    with tabs[1]:
        st.markdown("#### SKILL.md")
        new_skill_text = st.text_area(
            "",
            value=st.session_state["skills_text"],
            height=400,
            key="skills_md_editor",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(t("save_config"), key="skills_apply"):
                st.session_state["skills_text"] = new_skill_text
                st.session_state["skills_map"] = parse_skills(new_skill_text)
                st.success("SKILL.md applied to session.")
        with c2:
            st.download_button(
                t("download"),
                data=new_skill_text,
                file_name="SKILL.md",
                mime="text/markdown",
                key="skills_download_btn",
            )
        with c3:
            upload = st.file_uploader(t("upload"), type=["md", "markdown"], key="skills_upload")
            if upload is not None:
                try:
                    text = upload.read().decode("utf-8")
                    st.session_state["skills_text"] = text
                    st.session_state["skills_map"] = parse_skills(text)
                    st.success("Uploaded SKILL.md applied.")
                except Exception as e:
                    st.error(f"Upload parse error: {e}")


# =========================
# --------- MAIN ----------
# =========================

def main():
    st.set_page_config(
        page_title="Agentic Medical Device TPLC Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    apply_theme()

    st.title(t("title"))
    render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs([
        t("dashboard_tab"),
        t("agent_lab_tab"),
        t("note_keeper_tab"),
        t("config_tab"),
    ])

    with tab1:
        render_dashboard()
    with tab2:
        render_agent_lab()
    with tab3:
        render_note_keeper()
    with tab4:
        render_config_tab()


if __name__ == "__main__":
    main()
