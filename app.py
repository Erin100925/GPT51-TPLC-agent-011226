import os
import json
import re
import random
from typing import Dict, Any, List, Optional

import streamlit as st
import yaml

# Data / viz
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# External LLM SDKs
import openai
import google.generativeai as genai
import anthropic


# ---------------------------
# Constants & UI Dictionaries
# ---------------------------

DEFAULT_MAX_TOKENS = 12000

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

PAINTER_STYLES = [
    "Vincent van Gogh",
    "Claude Monet",
    "Pablo Picasso",
    "Leonardo da Vinci",
    "Salvador Dalí",
    "Frida Kahlo",
    "Edvard Munch",
    "Gustav Klimt",
    "Georgia O’Keeffe",
    "Jackson Pollock",
    "Henri Matisse",
    "Wassily Kandinsky",
    "Paul Cézanne",
    "Joan Miró",
    "Rembrandt",
    "Caravaggio",
    "Diego Velázquez",
    "Marc Chagall",
    "Roy Lichtenstein",
    "Andy Warhol",
]

PAINTER_STYLE_PALETTES = {
    "Vincent van Gogh": ("linear-gradient(135deg,#0f172a,#1e3a8a)", "#fbbf24"),
    "Claude Monet": ("linear-gradient(135deg,#e0f2fe,#bae6fd)", "#0369a1"),
    "Pablo Picasso": ("linear-gradient(135deg,#111827,#4b5563)", "#f97316"),
    "Leonardo da Vinci": ("linear-gradient(135deg,#fef3c7,#fde68a)", "#92400e"),
    "Salvador Dalí": ("linear-gradient(135deg,#fef2f2,#fee2e2)", "#b91c1c"),
    "Frida Kahlo": ("linear-gradient(135deg,#fdf2f8,#fce7f3)", "#be123c"),
    "Edvard Munch": ("linear-gradient(135deg,#111827,#7f1d1d)", "#f97316"),
    "Gustav Klimt": ("linear-gradient(135deg,#fef3c7,#facc15)", "#b45309"),
    "Georgia O’Keeffe": ("linear-gradient(135deg,#dcfce7,#bbf7d0)", "#166534"),
    "Jackson Pollock": ("linear-gradient(135deg,#020617,#1f2937)", "#67e8f9"),
    "Henri Matisse": ("linear-gradient(135deg,#eff6ff,#dbeafe)", "#1d4ed8"),
    "Wassily Kandinsky": ("linear-gradient(135deg,#f9fafb,#e5e7eb)", "#7c3aed"),
    "Paul Cézanne": ("linear-gradient(135deg,#fef9c3,#fef08a)", "#ca8a04"),
    "Joan Miró": ("linear-gradient(135deg,#faf5ff,#ede9fe)", "#7e22ce"),
    "Rembrandt": ("linear-gradient(135deg,#0f172a,#1f2937)", "#facc15"),
    "Caravaggio": ("linear-gradient(135deg,#020617,#1f2937)", "#f97316"),
    "Diego Velázquez": ("linear-gradient(135deg,#111827,#374151)", "#facc15"),
    "Marc Chagall": ("linear-gradient(135deg,#eef2ff,#e0e7ff)", "#4c1d95"),
    "Roy Lichtenstein": ("linear-gradient(135deg,#faf5ff,#fee2e2)", "#1d4ed8"),
    "Andy Warhol": ("linear-gradient(135deg,#ecfeff,#e0f2fe)", "#e11d48"),
}

LANG_EN = "en"
LANG_ZH = "zh-TW"

UI_TEXT = {
    LANG_EN: {
        "title": "Agentic AI Project Orchestrator",
        "project_input": "Project Description / Tender Text",
        "run_orchestrator": "Generate Project Plan",
        "orchestrator_settings": "Orchestrator Settings",
        "model": "Model",
        "max_tokens": "Max tokens",
        "system_prompt": "Orchestrator System Prompt (optional, advanced)",
        "dashboard": "Project Dashboard",
        "work_breakdown": "Work Breakdown",
        "timeline": "Timeline",
        "agent_matrix": "Agent Allocation",
        "risk_heatmap": "Risk Heatmap",
        "dependencies": "Dependency Graph",
        "config": "Configuration",
        "agents_tab": "Agents & Execution",
        "skills_tab": "Skills",
        "chat_tab": "Refinement / Prompt on Results",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language": "Language",
        "painter_style": "Painter Style",
        "jackpot": "Jackpot!",
        "api_section": "API Keys",
        "api_hint": "If environment variables exist, they will be used. You only need to fill missing keys.",
        "openai_key": "OpenAI API Key",
        "gemini_key": "Gemini API Key",
        "anthropic_key": "Anthropic API Key",
        "grok_key": "Grok API Key",
        "save_keys": "Save keys to session",
        "plan_missing": "No project plan yet. Please generate it first.",
        "run_agent": "Run this agent",
        "agent_input": "Agent input / task context",
        "agent_result_view": "Result view",
        "text_view": "Plain text",
        "markdown_view": "Markdown",
        "shared_handoff": "Shared Agent Handoff Buffer",
        "use_last_output": "Use last agent output in handoff buffer",
        "refresh_config": "Reload agents.yaml & SKILL.md",
        "wow_status": "WOW Status",
        "wow_agents": "Agents loaded",
        "wow_workitems": "Work items",
        "wow_risks": "Identified risks",
        "wow_ready": "Ready to orchestrate",
        "chat_prompt": "Refinement prompt (Prompt on Results)",
        "run_refinement": "Run refinement",
        "apply_refinement": "Apply refined fragment to plan",
        "nodes_label": "Nodes are work items; arrows show dependencies.",

        # New – supply chain tab
        "supply_tab": "Medical Supply Chain",
        "supply_intro": "Upload, preview, modify and analyze medical device tracking records (supplier packing list & hospital incoming list).",
        "supplier_dataset": "Supplier Packing List",
        "hospital_dataset": "Hospital Incoming List",
        "upload_supplier": "Upload supplier packing list (CSV or JSON)",
        "upload_hospital": "Upload hospital incoming list (CSV or JSON)",
        "download_supplier_csv": "Download supplier data (CSV)",
        "download_supplier_json": "Download supplier data (JSON)",
        "download_hospital_csv": "Download hospital data (CSV)",
        "download_hospital_json": "Download hospital data (JSON)",
        "reset_to_mock": "Reset to mock sample dataset",
        "supply_datasets_section": "Datasets – Medical Device Tracking",
        "supply_summary_section": "AI Summary & Markdown Report (1000–2000 words)",
        "summary_language": "Summary language",
        "summary_run": "Generate supply chain summary",
        "summary_words_hint": "The summary will be 1000–2000 words and include 5 graphs described in markdown.",
        "summary_latest": "Latest supply chain summary (editable markdown):",
        "graph_section": "Interactive Supply Chain Network",
        "graph_hint": "Suppliers and hospitals are visualized as a network. Hover nodes/edges to explore flows.",
        "min_shipments": "Minimum total quantity per supplier-hospital link",
        "filter_device": "Filter by device ID (optional)",
        "no_graph_data": "Not enough columns to build a supply chain graph. Expect at least 'supplier_name' and 'hospital_name'.",
        "supply_metrics": "Supply Chain WOW",
        "metric_suppliers": "Suppliers",
        "metric_hospitals": "Hospitals",
        "metric_shipments": "Supplier records",
        "metric_receipts": "Hospital receipts",

        "data_chat_section": "Prompt on Datasets (Ad‑hoc Questions)",
        "data_chat_prompt": "Ask anything about the current supplier & hospital datasets.",
        "data_chat_run": "Ask AI about datasets",
        "data_agent_section": "Run Agents from agents.yaml on Datasets",
        "data_agent_select": "Select agent",
        "data_agent_run": "Run selected agent on datasets",
        "data_prompt_with_context": "Additional instructions / focus for this run",
    },
    LANG_ZH: {
        "title": "智慧代理專案協調器",
        "project_input": "專案說明 / 標案文字",
        "run_orchestrator": "產生專案計畫",
        "orchestrator_settings": "協調器設定",
        "model": "模型",
        "max_tokens": "最大 token 數",
        "system_prompt": "協調器系統提示（選填，高階設定）",
        "dashboard": "專案儀表板",
        "work_breakdown": "工作分解結構",
        "timeline": "時程規劃",
        "agent_matrix": "代理與資源配置",
        "risk_heatmap": "風險熱度圖",
        "dependencies": "相依關係圖",
        "config": "設定",
        "agents_tab": "代理與執行",
        "skills_tab": "技能",
        "chat_tab": "優化 / 結果再提示",
        "theme": "主題",
        "light": "亮色",
        "dark": "暗色",
        "language": "語言",
        "painter_style": "畫家風格",
        "jackpot": "隨機大補帖",
        "api_section": "API 金鑰",
        "api_hint": "若已設定環境變數，將自動使用。僅需填寫缺少的金鑰即可。",
        "openai_key": "OpenAI API 金鑰",
        "gemini_key": "Gemini API 金鑰",
        "anthropic_key": "Anthropic API 金鑰",
        "grok_key": "Grok API 金鑰",
        "save_keys": "儲存金鑰到本次工作階段",
        "plan_missing": "目前尚未有專案計畫，請先執行協調器。",
        "run_agent": "執行此代理",
        "agent_input": "代理輸入 / 任務內容",
        "agent_result_view": "結果檢視模式",
        "text_view": "純文字",
        "markdown_view": "Markdown",
        "shared_handoff": "代理交辦共用緩衝區",
        "use_last_output": "以上一個代理輸出更新交辦內容",
        "refresh_config": "重新載入 agents.yaml 與 SKILL.md",
        "wow_status": "WOW 狀態指標",
        "wow_agents": "已載入代理數",
        "wow_workitems": "工作項目數",
        "wow_risks": "風險項目數",
        "wow_ready": "可開始協調",
        "chat_prompt": "優化提示（針對目前結果進一步要求）",
        "run_refinement": "執行優化",
        "apply_refinement": "套用優化片段至計畫",
        "nodes_label": "節點為工作項目，箭頭為相依關係。",

        # New – supply chain tab
        "supply_tab": "醫療器材供應鏈",
        "supply_intro": "上傳、預覽、修改與分析醫療器材追蹤紀錄（供應商裝箱單與醫院入庫清單）。",
        "supplier_dataset": "供應商裝箱單",
        "hospital_dataset": "醫院入庫清單",
        "upload_supplier": "上傳供應商裝箱單（CSV 或 JSON）",
        "upload_hospital": "上傳醫院入庫清單（CSV 或 JSON）",
        "download_supplier_csv": "下載供應商資料（CSV）",
        "download_supplier_json": "下載供應商資料（JSON）",
        "download_hospital_csv": "下載醫院資料（CSV）",
        "download_hospital_json": "下載醫院資料（JSON）",
        "reset_to_mock": "重置為預設範例資料",
        "supply_datasets_section": "資料集 – 醫療器材追蹤",
        "supply_summary_section": "AI 摘要與 Markdown 報告（1000–2000 字）",
        "summary_language": "摘要語言",
        "summary_run": "產生供應鏈摘要",
        "summary_words_hint": "摘要將以 1000–2000 字撰寫，並包含 5 個以 Markdown 描述的圖表。",
        "summary_latest": "最新供應鏈摘要（可編輯的 Markdown）：",
        "graph_section": "互動式供應鏈關係圖",
        "graph_hint": "以網路圖呈現供應商與醫院關係，游標移動可檢視細節。",
        "min_shipments": "每一供應商–醫院連線的最小總數量",
        "filter_device": "依裝置編號過濾（選填）",
        "no_graph_data": "目前欄位不足以建立供應鏈關係圖，預期至少包含「supplier_name」與「hospital_name」。",
        "supply_metrics": "供應鏈 WOW",
        "metric_suppliers": "供應商數",
        "metric_hospitals": "醫院數",
        "metric_shipments": "供應商紀錄數",
        "metric_receipts": "醫院入庫紀錄數",

        "data_chat_section": "針對資料集提問（即時 QA）",
        "data_chat_prompt": "請就目前的供應商與醫院資料提問。",
        "data_chat_run": "詢問 AI 關於資料集的問題",
        "data_agent_section": "在資料集上執行 agents.yaml 中的代理",
        "data_agent_select": "選擇代理",
        "data_agent_run": "在資料集上執行所選代理",
        "data_prompt_with_context": "本次執行的額外說明 / 聚焦重點",
    },
}


# -------------------------------------------
# Mock datasets for medical device tracking
# -------------------------------------------

def create_mock_supplier_df() -> pd.DataFrame:
    data = [
        {
            "shipment_id": "SHP-1001",
            "supplier_name": "Global MedTech Ltd.",
            "hospital_code": "H001",
            "hospital_name": "City General Hospital",
            "device_id": "DEV-VENT-01",
            "device_name": "ICU Ventilator X100",
            "lot_number": "LOT-A1",
            "expiry_date": "2027-03-31",
            "ship_date": "2026-01-03",
            "quantity": 10,
            "uom": "units",
            "po_number": "PO-9001",
        },
        {
            "shipment_id": "SHP-1002",
            "supplier_name": "Global MedTech Ltd.",
            "hospital_code": "H001",
            "hospital_name": "City General Hospital",
            "device_id": "DEV-PUMP-02",
            "device_name": "Infusion Pump Pro",
            "lot_number": "LOT-B2",
            "expiry_date": "2026-11-30",
            "ship_date": "2026-01-05",
            "quantity": 25,
            "uom": "units",
            "po_number": "PO-9002",
        },
        {
            "shipment_id": "SHP-1003",
            "supplier_name": "Precision Devices Corp.",
            "hospital_code": "H002",
            "hospital_name": "St. Mary Cardiac Center",
            "device_id": "DEV-STENT-03",
            "device_name": "Coronary Stent Alpha",
            "lot_number": "LOT-C3",
            "expiry_date": "2028-05-15",
            "ship_date": "2026-01-07",
            "quantity": 40,
            "uom": "units",
            "po_number": "PO-9100",
        },
        {
            "shipment_id": "SHP-1004",
            "supplier_name": "Precision Devices Corp.",
            "hospital_code": "H003",
            "hospital_name": "Metro Children’s Hospital",
            "device_id": "DEV-VENT-01",
            "device_name": "ICU Ventilator X100",
            "lot_number": "LOT-A1",
            "expiry_date": "2027-03-31",
            "ship_date": "2026-01-08",
            "quantity": 6,
            "uom": "units",
            "po_number": "PO-9101",
        },
        {
            "shipment_id": "SHP-1005",
            "supplier_name": "SterileCare Supplies",
            "hospital_code": "H001",
            "hospital_name": "City General Hospital",
            "device_id": "DEV-CATH-04",
            "device_name": "Central Venous Catheter",
            "lot_number": "LOT-D4",
            "expiry_date": "2026-09-30",
            "ship_date": "2026-01-09",
            "quantity": 120,
            "uom": "units",
            "po_number": "PO-9200",
        },
    ]
    return pd.DataFrame(data)


def create_mock_hospital_df() -> pd.DataFrame:
    data = [
        {
            "receipt_id": "RCV-5001",
            "linked_shipment_id": "SHP-1001",
            "hospital_code": "H001",
            "hospital_name": "City General Hospital",
            "device_id": "DEV-VENT-01",
            "device_name": "ICU Ventilator X100",
            "lot_number": "LOT-A1",
            "expiry_date": "2027-03-31",
            "received_date": "2026-01-04",
            "quantity": 10,
            "status": "accepted",
        },
        {
            "receipt_id": "RCV-5002",
            "linked_shipment_id": "SHP-1002",
            "hospital_code": "H001",
            "hospital_name": "City General Hospital",
            "device_id": "DEV-PUMP-02",
            "device_name": "Infusion Pump Pro",
            "lot_number": "LOT-B2",
            "expiry_date": "2026-11-30",
            "received_date": "2026-01-06",
            "quantity": 25,
            "status": "accepted",
        },
        {
            "receipt_id": "RCV-5003",
            "linked_shipment_id": "SHP-1003",
            "hospital_code": "H002",
            "hospital_name": "St. Mary Cardiac Center",
            "device_id": "DEV-STENT-03",
            "device_name": "Coronary Stent Alpha",
            "lot_number": "LOT-C3",
            "expiry_date": "2028-05-15",
            "received_date": "2026-01-09",
            "quantity": 38,
            "status": "accepted_with_variance",
        },
        {
            "receipt_id": "RCV-5004",
            "linked_shipment_id": "SHP-1004",
            "hospital_code": "H003",
            "hospital_name": "Metro Children’s Hospital",
            "device_id": "DEV-VENT-01",
            "device_name": "ICU Ventilator X100",
            "lot_number": "LOT-A1",
            "expiry_date": "2027-03-31",
            "received_date": "2026-01-10",
            "quantity": 6,
            "status": "accepted",
        },
        {
            "receipt_id": "RCV-5005",
            "linked_shipment_id": None,
            "hospital_code": "H001",
            "hospital_name": "City General Hospital",
            "device_id": "DEV-CATH-04",
            "device_name": "Central Venous Catheter",
            "lot_number": "LOT-D4",
            "expiry_date": "2026-09-30",
            "received_date": "2026-01-11",
            "quantity": 115,
            "status": "pending_investigation",
        },
    ]
    return pd.DataFrame(data)


# -------------------------------------------
# Session State Initialization & Theme / Lang
# -------------------------------------------

def init_session_state():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    if "lang" not in st.session_state:
        st.session_state["lang"] = LANG_EN
    if "painter_style" not in st.session_state:
        st.session_state["painter_style"] = PAINTER_STYLES[0]
    if "api_keys" not in st.session_state:
        st.session_state["api_keys"] = {
            "openai": None,
            "gemini": None,
            "anthropic": None,
            "grok": None,
        }
    if "agents_config" not in st.session_state:
        st.session_state["agents_config"] = {"agents": []}
    if "skills" not in st.session_state:
        st.session_state["skills"] = {}
    if "project_plan" not in st.session_state:
        st.session_state["project_plan"] = None
    if "last_agent_output" not in st.session_state:
        st.session_state["last_agent_output"] = ""
    if "handoff_buffer" not in st.session_state:
        st.session_state["handoff_buffer"] = ""
    if "refined_fragment" not in st.session_state:
        st.session_state["refined_fragment"] = ""
    if "orchestrator_settings" not in st.session_state:
        st.session_state["orchestrator_settings"] = {
            "model": SUPPORTED_MODELS[0],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system_prompt": "",
        }

    # New: datasets & analysis state
    if "supplier_df" not in st.session_state:
        st.session_state["supplier_df"] = create_mock_supplier_df()
    if "hospital_df" not in st.session_state:
        st.session_state["hospital_df"] = create_mock_hospital_df()
    if "supply_summary_md" not in st.session_state:
        st.session_state["supply_summary_md"] = ""
    if "data_chat_output" not in st.session_state:
        st.session_state["data_chat_output"] = ""


def get_ui_text() -> Dict[str, str]:
    return UI_TEXT.get(st.session_state["lang"], UI_TEXT[LANG_EN])


def apply_custom_theme():
    style_name = st.session_state.get("painter_style", PAINTER_STYLES[0])
    bg, accent = PAINTER_STYLE_PALETTES.get(
        style_name,
        ("linear-gradient(135deg,#020617,#1f2937)", "#3b82f6"),
    )
    is_dark = st.session_state.get("theme", "dark") == "dark"

    text_color = "#e5e7eb" if is_dark else "#111827"
    card_bg = "rgba(15,23,42,0.85)" if is_dark else "rgba(255,255,255,0.9)"
    border_color = "rgba(148,163,184,0.4)" if is_dark else "rgba(148,163,184,0.6)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg} !important;
            color: {text_color} !important;
        }}
        .wow-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1.1rem 1.25rem;
            border: 1px solid {border_color};
            box-shadow: 0 18px 45px rgba(15,23,42,0.6);
            backdrop-filter: blur(16px);
        }}
        .wow-title {{
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: {accent};
        }}
        .wow-pill {{
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            border: 1px solid {border_color};
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: {text_color};
        }}
        .wow-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.7;
        }}
        .wow-value {{
            font-size: 1.3rem;
            font-weight: 600;
        }}
        .wow-accent {{
            color: {accent};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Config Loading & Parsing
# -------------------------

def load_agents_config() -> Dict[str, Any]:
    path = "agents.yaml"
    if not os.path.exists(path):
        return {"agents": []}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "agents" not in cfg:
        cfg["agents"] = []
    return cfg


def parse_skills_md() -> Dict[str, Dict[str, Any]]:
    path = "SKILL.md"
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    skills: Dict[str, Dict[str, Any]] = {}
    blocks = re.split(r"^#\s*Skill:\s*", content, flags=re.MULTILINE)
    for block in blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        skill_id = first_line
        rest = "\n".join(lines[1:])
        desc_match = re.search(r"\*\*Description:\*\*\s*(.*)", rest)
        params_match = re.search(r"\*\*Parameters:\*\*\s*(.*)", rest)
        skills[skill_id] = {
            "id": skill_id,
            "description": desc_match.group(1).strip() if desc_match else "",
            "parameters": params_match.group(1).strip() if params_match else "",
            "raw": rest.strip(),
        }
    return skills


def refresh_config():
    st.session_state["agents_config"] = load_agents_config()
    st.session_state["skills"] = parse_skills_md()


# ----------------------------
# API Keys & LLM Client Helper
# ----------------------------

def init_api_keys_from_env():
    keys = st.session_state["api_keys"]
    if keys["openai"] is None:
        env_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
        if env_key:
            keys["openai"] = env_key
    if keys["gemini"] is None:
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            keys["gemini"] = env_key
    if keys["anthropic"] is None:
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            keys["anthropic"] = env_key
    if keys["grok"] is None:
        env_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        if env_key:
            keys["grok"] = env_key
    st.session_state["api_keys"] = keys


def detect_provider(model_name: str) -> str:
    mn = model_name.lower()
    if mn.startswith("gpt-"):
        return "openai"
    if mn.startswith("gemini-"):
        return "gemini"
    if mn.startswith("claude-") or "sonnet" in mn or "haiku" in mn or "anthropic" in mn:
        return "anthropic"
    if mn.startswith("grok-"):
        return "grok"
    return "openai"


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    provider = detect_provider(model)
    keys = st.session_state["api_keys"]

    if provider == "openai":
        api_key = keys.get("openai")
        if not api_key:
            raise RuntimeError("Missing OpenAI API key.")
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        api_key = keys.get("gemini")
        if not api_key:
            raise RuntimeError("Missing Gemini API key.")
        genai.configure(api_key=api_key)
        prompt = system_prompt + "\n\n" + user_prompt if system_prompt else user_prompt
        model_obj = genai.GenerativeModel(model)
        resp = model_obj.generate_content(prompt)
        return resp.text

    elif provider == "anthropic":
        api_key = keys.get("anthropic")
        if not api_key:
            raise RuntimeError("Missing Anthropic API key.")
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = ""
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text += block.text
        return text

    elif provider == "grok":
        api_key = keys.get("grok")
        if not api_key:
            raise RuntimeError("Missing Grok API key.")
        grok_client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        resp = grok_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

    else:
        raise RuntimeError(f"Unknown provider for model {model}")


# ---------------------------------
# Orchestrator Prompt & JSON Helper
# ---------------------------------

def build_orchestrator_system_prompt() -> str:
    agents = st.session_state["agents_config"]["agents"]
    skills = st.session_state["skills"]

    skill_summaries = []
    for sid, s in skills.items():
        skill_summaries.append(
            f"- {sid}: {s.get('description','')} (params: {s.get('parameters','')})"
        )
    skills_block = "\n".join(skill_summaries) if skill_summaries else "None."

    agent_summaries = []
    for a in agents:
        agent_summaries.append(
            f"- id: {a.get('id')} | name: {a.get('name')} | role: {a.get('role')} | "
            f"capabilities: {', '.join(a.get('capabilities', []))}"
        )
    agents_block = "\n".join(agent_summaries) if agent_summaries else "None."

    return f"""
You are the Orchestrator for an Agentic AI Project Planning system.

You must read an unstructured project or tender description, then output a JSON object
strictly matching the following TypeScript interface (no extra keys):

interface ProjectPlan {{
  meta: {{
    title: string;
    summary: string;
    domain: string;
  }};
  workItems: Array<{{ 
    id: string;
    title: string;
    description: string;
    assignedAgentId: string;
    complexity: "Low" | "Medium" | "High";
    phase: string;
  }}>;
  timeline: Array<{{
    phaseName: string;
    startDate: string;
    duration: string;
    milestones: string[];
  }}>;
  risks: Array<{{
    description: string;
    impact: number;
    probability: number;
    mitigationStrategy: string;
  }}>;
  dependencies: Array<{{
    source: string;
    target: string;
    type: "Blocking" | "Informational";
  }}>;
}}

Agents available (from agents.yaml):
{agents_block}

Skills available (from SKILL.md):
{skills_block}

Rules:
- Use assignedAgentId values that match existing agent ids when possible.
- Ensure IDs like workItems[i].id are unique strings (e.g., "1.1", "2.3").
- Return ONLY valid JSON. Do NOT wrap JSON in markdown fences. Do not add comments.
"""


def parse_json_from_llm(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception as e:
            raise ValueError(f"Failed to parse model output as JSON. Error: {e}")
    raise ValueError("No JSON object found in model output.")


# ---------------------------
# Visualization – WOW & Views
# ---------------------------

def render_wow_status():
    ui = get_ui_text()
    plan = st.session_state["project_plan"]
    agents = st.session_state["agents_config"]["agents"]
    skills = st.session_state["skills"]

    work_items_count = len(plan.get("workItems", [])) if plan else 0
    risks_count = len(plan.get("risks", [])) if plan else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["wow_agents"]}</div>
              <div class="wow-value wow-accent">{len(agents)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">Skills</div>
              <div class="wow-value wow-accent">{len(skills)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["wow_workitems"]}</div>
              <div class="wow-value wow-accent">{work_items_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["wow_risks"]}</div>
              <div class="wow-value wow-accent">{risks_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_supply_wow_status():
    """Additional WOW metrics for the medical supply chain."""
    ui = get_ui_text()
    supplier_df: pd.DataFrame = st.session_state["supplier_df"]
    hospital_df: pd.DataFrame = st.session_state["hospital_df"]

    suppliers = supplier_df["supplier_name"].nunique() if "supplier_name" in supplier_df.columns else len(supplier_df)
    hospitals = 0
    if "hospital_name" in supplier_df.columns:
        hospitals = supplier_df["hospital_name"].nunique()
    if "hospital_name" in hospital_df.columns:
        hospitals = max(hospitals, hospital_df["hospital_name"].nunique())

    shipments = len(supplier_df)
    receipts = len(hospital_df)

    st.markdown(f"#### {ui['supply_metrics']}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["metric_suppliers"]}</div>
              <div class="wow-value wow-accent">{suppliers}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["metric_hospitals"]}</div>
              <div class="wow-value wow-accent">{hospitals}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["metric_shipments"]}</div>
              <div class="wow-value wow-accent">{shipments}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="wow-card">
              <div class="wow-label">{ui["metric_receipts"]}</div>
              <div class="wow-value wow-accent">{receipts}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_work_breakdown(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["work_breakdown"])
    work_items = plan.get("workItems", [])
    if not work_items:
        st.info("No work items in the plan yet.")
        return

    rows = []
    for wi in work_items:
        rows.append(
            {
                "ID": wi.get("id"),
                "Title": wi.get("title"),
                "Description": wi.get("description"),
                "Agent": wi.get("assignedAgentId"),
                "Complexity": wi.get("complexity"),
                "Phase": wi.get("phase"),
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_timeline(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["timeline"])
    timeline = plan.get("timeline", [])
    if not timeline:
        st.info("No timeline information available.")
        return
    for phase in timeline:
        with st.expander(f"{phase.get('phaseName','(Phase)')} – {phase.get('duration','')}"):
            st.write(f"Start: {phase.get('startDate','')}")
            mstones = phase.get("milestones", [])
            if mstones:
                st.markdown("**Milestones:**")
                for m in mstones:
                    st.markdown(f"- {m}")


def render_agent_matrix(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["agent_matrix"])
    work_items = plan.get("workItems", [])
    agents_by_id = {
        a.get("id"): a for a in st.session_state["agents_config"]["agents"]
    }

    if not work_items:
        st.info("No work items in the plan.")
        return

    rows = []
    for wi in work_items:
        agent_id = wi.get("assignedAgentId")
        agent = agents_by_id.get(agent_id)
        rows.append(
            {
                "Work Item ID": wi.get("id"),
                "Title": wi.get("title"),
                "Agent ID": agent_id,
                "Agent Name": agent.get("name") if agent else "",
                "Agent Role": agent.get("role") if agent else "",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_risk_heatmap(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["risk_heatmap"])
    risks = plan.get("risks", [])
    if not risks:
        st.info("No risks defined in the plan.")
        return

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**High Impact / High Probability**")
        for r in risks:
            if r.get("impact", 0) >= 7 and r.get("probability", 0) >= 7:
                st.markdown(f"- {r.get('description')}")
    with cols[1]:
        st.markdown("**High Impact / Medium Probability**")
        for r in risks:
            if r.get("impact", 0) >= 7 and 4 <= r.get("probability", 0) < 7:
                st.markdown(f"- {r.get('description')}")
    with cols[2]:
        st.markdown("**Medium Impact / Low Probability**")
        for r in risks:
            if 4 <= r.get("impact", 0) < 7 and r.get("probability", 0) < 4:
                st.markdown(f"- {r.get('description')}")


def render_dependency_graph(plan: Dict[str, Any]):
    ui = get_ui_text()
    st.subheader(ui["dependencies"])
    deps = plan.get("dependencies", [])
    if not deps:
        st.info("No dependencies defined.")
        return

    work_items = {w.get("id"): w for w in plan.get("workItems", [])}
    nodes = []
    edges = []
    for wid, w in work_items.items():
        title = w.get("title", "").replace('"', "'")
        nodes.append(f'"{wid}" [label="{wid}: {title}"];')
    for d in deps:
        s = d.get("source")
        t = d.get("target")
        dep_type = d.get("type", "Informational")
        if not s or not t:
            continue
        color = "red" if dep_type == "Blocking" else "gray"
        edges.append(f'"{s}" -> "{t}" [color="{color}"];')

    dot = "digraph G {\n" + "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}"
    st.graphviz_chart(dot)
    st.caption(ui["nodes_label"])


# --------------------------
# Agent Execution / Chaining
# --------------------------

def render_agents_tab():
    ui = get_ui_text()
    plan = st.session_state["project_plan"]
    agents = st.session_state["agents_config"]["agents"]
    if not agents:
        st.info("No agents loaded from agents.yaml.")
        return

    st.subheader(ui["agents_tab"])

    st.markdown(f"**{ui['shared_handoff']}**")
    st.text_area(
        label="",
        value=st.session_state["handoff_buffer"],
        key="handoff_buffer_widget",
        height=120,
    )
    st.session_state["handoff_buffer"] = st.session_state.get(
        "handoff_buffer_widget", ""
    )

    st.markdown("---")

    if not plan:
        st.info(ui["plan_missing"])
        return

    work_items = plan.get("workItems", [])
    if not work_items:
        st.info("No work items to assign agents to.")
        return

    agent_to_items: Dict[str, List[Dict[str, Any]]] = {a["id"]: [] for a in agents}
    for wi in work_items:
        aid = wi.get("assignedAgentId")
        if aid in agent_to_items:
            agent_to_items[aid].append(wi)

    for agent in agents:
        aid = agent.get("id")
        with st.expander(f"Agent: {agent.get('name')} ({aid}) – {agent.get('role')}"):
            items = agent_to_items.get(aid, [])
            if not items:
                st.caption("No work items currently assigned.")
            else:
                for wi in items:
                    st.markdown(f"**[{wi.get('id')}] {wi.get('title')}**")
                    st.caption(wi.get("description", ""))

                    default_input = (
                        f"You are {agent.get('name')} with role {agent.get('role')}.\n"
                        f"Task: {wi.get('title')} (ID: {wi.get('id')})\n"
                        f"Description: {wi.get('description')}\n\n"
                        f"Handoff context (if any):\n{st.session_state.get('handoff_buffer','')}\n"
                    )
                    user_input_key = f"agent_input_{aid}_{wi.get('id')}"
                    agent_input = st.text_area(
                        ui["agent_input"],
                        value=default_input,
                        key=user_input_key,
                        height=160,
                    )

                    col_run, col_view = st.columns([1, 1])
                    with col_run:
                        btn_key = f"run_{aid}_{wi.get('id')}"
                        if st.button(ui["run_agent"], key=btn_key):
                            try:
                                # Allow override by orchestrator settings if agent.model not set
                                model = agent.get("model") or st.session_state[
                                    "orchestrator_settings"
                                ]["model"]
                                max_tokens = st.session_state["orchestrator_settings"][
                                    "max_tokens"
                                ]
                                system_prompt = agent.get("system_prompt", "")
                                result = call_llm(
                                    model=model,
                                    system_prompt=system_prompt,
                                    user_prompt=agent_input,
                                    max_tokens=max_tokens,
                                )
                                st.session_state["last_agent_output"] = result
                                st.session_state["handoff_buffer"] = result
                                st.session_state["handoff_buffer_widget"] = result
                                st.success("Agent execution completed.")
                            except Exception as e:
                                st.error(f"Agent call failed: {e}")

                    with col_view:
                        view_mode = st.radio(
                            ui["agent_result_view"],
                            options=[ui["text_view"], ui["markdown_view"]],
                            key=f"view_{aid}_{wi.get('id')}",
                            horizontal=True,
                        )

                    if st.session_state["last_agent_output"]:
                        if view_mode == ui["markdown_view"]:
                            st.markdown(st.session_state["last_agent_output"])
                        else:
                            st.text(st.session_state["last_agent_output"])


# --------------------------
# Refinement / Prompt on Plan
# --------------------------

def render_refinement_tab():
    ui = get_ui_text()
    plan = st.session_state["project_plan"]
    st.subheader(ui["chat_tab"])
    if not plan:
        st.info(ui["plan_missing"])
        return

    refinement_prompt = st.text_area(
        ui["chat_prompt"],
        value="For Item 10, strictly focus on Generative AI for label recognition.",
        height=140,
    )
    if st.button(ui["run_refinement"]):
        try:
            model = st.session_state["orchestrator_settings"]["model"]
            max_tokens = 2000
            system_prompt = (
                "You are a JSON patch generator for the current project plan. "
                "You MUST return only a valid JSON fragment that can be merged into "
                "the existing plan (e.g., an updated workItems array or a single updated object)."
            )
            user_prompt = (
                "Current plan JSON:\n"
                + json.dumps(plan, ensure_ascii=False, indent=2)
                + "\n\nUser refinement request:\n"
                + refinement_prompt
                + "\n\nReturn ONLY the updated JSON fragment, no comments, no markdown."
            )
            result = call_llm(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            st.session_state["refined_fragment"] = result
            st.success("Refinement result generated (raw JSON fragment below).")
        except Exception as e:
            st.error(f"Refinement LLM call failed: {e}")

    if st.session_state["refined_fragment"]:
        st.markdown("**Refined JSON fragment (editable before apply):**")
        frag = st.text_area(
            "",
            value=st.session_state["refined_fragment"],
            key="refined_fragment",
            height=200,
        )
        if st.button(ui["apply_refinement"]):
            try:
                fragment_obj = parse_json_from_llm(frag)
                plan = st.session_state["project_plan"] or {}
                for key in ["meta", "workItems", "timeline", "risks", "dependencies"]:
                    if key in fragment_obj:
                        plan[key] = fragment_obj[key]
                st.session_state["project_plan"] = plan
                st.success("Refined fragment applied to current plan.")
            except Exception as e:
                st.error(f"Failed to apply fragment: {e}")


# -------------------------
# Orchestrator Runner (UI)
# -------------------------

def run_orchestrator_ui():
    ui = get_ui_text()
    st.subheader(ui["orchestrator_settings"])

    settings = st.session_state["orchestrator_settings"]
    col1, col2 = st.columns([2, 1])
    with col1:
        model = st.selectbox(
            ui["model"],
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(settings["model"])
            if settings["model"] in SUPPORTED_MODELS
            else 0,
        )
    with col2:
        max_tokens = st.number_input(
            ui["max_tokens"],
            min_value=256,
            max_value=64000,
            value=settings.get("max_tokens", DEFAULT_MAX_TOKENS),
            step=512,
        )

    system_prompt_override = st.text_area(
        ui["system_prompt"],
        value=settings.get("system_prompt", ""),
        height=150,
    )

    settings["model"] = model
    settings["max_tokens"] = max_tokens
    settings["system_prompt"] = system_prompt_override
    st.session_state["orchestrator_settings"] = settings

    project_text = st.text_area(
        ui["project_input"],
        height=280,
        key="project_input",
        value="",
    )

    if st.button(ui["run_orchestrator"]):
        if not project_text.strip():
            st.warning("Please paste a project / tender description first.")
        else:
            try:
                base_system_prompt = (
                    system_prompt_override
                    if system_prompt_override.strip()
                    else build_orchestrator_system_prompt()
                )
                user_prompt = (
                    "Project / tender description:\n\n"
                    + project_text
                    + "\n\nNow generate the ProjectPlan JSON as specified."
                )
                with st.spinner("Orchestrating project plan with selected model..."):
                    raw = call_llm(
                        model=model,
                        system_prompt=base_system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )
                plan = parse_json_from_llm(raw)
                st.session_state["project_plan"] = plan
                st.success("Project plan generated successfully.")
            except Exception as e:
                st.error(f"Orchestrator call failed: {e}")


# -------------------------
# Supply chain – helpers
# -------------------------

def load_uploaded_df(file) -> pd.DataFrame:
    if file is None:
        return None
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith(".json"):
            data = json.load(file)
            return pd.DataFrame(data)
        else:
            # try CSV first, then JSON
            try:
                file.seek(0)
                return pd.read_csv(file)
            except Exception:
                file.seek(0)
                data = json.load(file)
                return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")
        return None


def build_supply_chain_graph(
    supplier_df: pd.DataFrame,
    hospital_df: pd.DataFrame,
    min_quantity: float = 0.0,
    device_filter: Optional[List[str]] = None,
) -> Optional[go.Figure]:
    # require at least supplier_name and hospital_name from supplier_df
    if "supplier_name" not in supplier_df.columns:
        return None
    if "hospital_name" not in supplier_df.columns and "hospital_name" not in hospital_df.columns:
        return None

    df_sup = supplier_df.copy()
    if device_filter:
        if "device_id" in df_sup.columns:
            df_sup = df_sup[df_sup["device_id"].isin(device_filter)]

    # quantity aggregation
    qty_col = "quantity" if "quantity" in df_sup.columns else None
    if qty_col:
        grp = (
            df_sup.groupby(
                ["supplier_name", "hospital_name"],
                dropna=False,
            )[qty_col]
            .sum()
            .reset_index()
        )
        grp = grp[grp[qty_col] >= min_quantity]
    else:
        grp = (
            df_sup.groupby(
                ["supplier_name", "hospital_name"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )
        qty_col = "count"
        grp = grp[grp[qty_col] >= min_quantity]

    if grp.empty:
        return None

    G = nx.Graph()
    for _, row in grp.iterrows():
        s = row["supplier_name"]
        h = row["hospital_name"]
        qty = row[qty_col]
        if pd.isna(h):
            continue
        G.add_node(s, type="supplier")
        G.add_node(h, type="hospital")
        G.add_edge(s, h, weight=float(qty))

    if not G.nodes:
        return None

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x = []
    edge_y = []
    edge_weights = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(edge[2].get("weight", 1.0))

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    max_weight = max(edge_weights) if edge_weights else 1.0
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        typ = data.get("type", "other")
        if typ == "supplier":
            node_color.append("#0ea5e9")  # blue
        elif typ == "hospital":
            node_color.append("#22c55e")  # green
        else:
            node_color.append("#e5e7eb")
        total_weight = sum(
            e[2].get("weight", 1.0)
            for e in G.edges(node, data=True)
        )
        size = 10 + 30 * (total_weight / max_weight)
        node_size.append(size)
        node_text.append(f"{typ}: {node}\nTotal quantity: {total_weight}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#9ca3af"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line_width=1,
            line_color="#020617",
        ),
        text=node_text,
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


# -------------------------
# Supply chain – main tab UI
# -------------------------

def render_supply_chain_tab():
    ui = get_ui_text()
    st.subheader(ui["supply_tab"])
    st.caption(ui["supply_intro"])

    supplier_df: pd.DataFrame = st.session_state["supplier_df"]
    hospital_df: pd.DataFrame = st.session_state["hospital_df"]

    # WOW metrics for supply chain
    render_supply_wow_status()

    st.markdown("---")
    st.markdown(f"### {ui['supply_datasets_section']}")

    col_sup, col_hosp = st.columns(2)

    # Supplier dataset
    with col_sup:
        st.markdown(f"**{ui['supplier_dataset']}**")
        uploaded_sup = st.file_uploader(
            ui["upload_supplier"],
            type=["csv", "json"],
            key="upload_supplier",
        )
        if uploaded_sup is not None:
            df = load_uploaded_df(uploaded_sup)
            if df is not None:
                st.session_state["supplier_df"] = df
                supplier_df = df
                st.success("Supplier dataset loaded.")

        supplier_df = st.data_editor(
            supplier_df,
            key="supplier_df_editor",
            use_container_width=True,
            num_rows="dynamic",
        )
        st.session_state["supplier_df"] = supplier_df

        sup_csv = supplier_df.to_csv(index=False).encode("utf-8")
        sup_json = json.dumps(
            supplier_df.to_dict(orient="records"), ensure_ascii=False, indent=2
        ).encode("utf-8")

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.download_button(
                ui["download_supplier_csv"],
                data=sup_csv,
                file_name="supplier_packing_list.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                ui["download_supplier_json"],
                data=sup_json,
                file_name="supplier_packing_list.json",
                mime="application/json",
            )
        with c3:
            if st.button(ui["reset_to_mock"], key="reset_supplier"):
                st.session_state["supplier_df"] = create_mock_supplier_df()
                st.experimental_rerun()

    # Hospital dataset
    with col_hosp:
        st.markdown(f"**{ui['hospital_dataset']}**")
        uploaded_hosp = st.file_uploader(
            ui["upload_hospital"],
            type=["csv", "json"],
            key="upload_hospital",
        )
        if uploaded_hosp is not None:
            df = load_uploaded_df(uploaded_hosp)
            if df is not None:
                st.session_state["hospital_df"] = df
                hospital_df = df
                st.success("Hospital dataset loaded.")

        hospital_df = st.data_editor(
            hospital_df,
            key="hospital_df_editor",
            use_container_width=True,
            num_rows="dynamic",
        )
        st.session_state["hospital_df"] = hospital_df

        hosp_csv = hospital_df.to_csv(index=False).encode("utf-8")
        hosp_json = json.dumps(
            hospital_df.to_dict(orient="records"), ensure_ascii=False, indent=2
        ).encode("utf-8")

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.download_button(
                ui["download_hospital_csv"],
                data=hosp_csv,
                file_name="hospital_incoming_list.csv",
                mime="text/csv",
            )
        with c2:
            st.download_button(
                ui["download_hospital_json"],
                data=hosp_json,
                file_name="hospital_incoming_list.json",
                mime="application/json",
            )
        with c3:
            if st.button(ui["reset_to_mock"], key="reset_hospital"):
                st.session_state["hospital_df"] = create_mock_hospital_df()
                st.experimental_rerun()

    st.markdown("---")

    # Summary section
    st.markdown(f"### {ui['supply_summary_section']}")
    st.caption(ui["summary_words_hint"])

    lang_choice = st.radio(
        ui["summary_language"],
        options=[LANG_EN, LANG_ZH],
        horizontal=True,
        key="supply_summary_lang",
        index=0 if st.session_state["lang"] == LANG_EN else 1,
    )

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        model = st.selectbox(
            ui["model"],
            options=SUPPORTED_MODELS,
            key="supply_summary_model",
        )
    with col_m2:
        max_tokens = st.number_input(
            ui["max_tokens"],
            min_value=2000,
            max_value=32000,
            value=8000,
            step=500,
            key="supply_summary_max_tokens",
        )

    default_prompt_en = (
        "You are a senior medical device supply chain analyst.\n"
        "Using the supplier packing list and hospital incoming list provided below, "
        "write a comprehensive analytical report between 1000 and 2000 words.\n\n"
        "Requirements:\n"
        "- Focus on traceability between suppliers and hospitals, lot-level tracking, and potential discrepancies.\n"
        "- Highlight risks (e.g., expiry risk, quantity mismatch, shipments without receipts, receipts without shipments).\n"
        "- Provide operational insights and recommendations for quality, safety, and regulatory compliance.\n"
        "- Use Markdown formatting with clear headings and subheadings.\n"
        "- Include exactly 5 graph-style sections in Markdown. For each graph, use a heading like:\n"
        "  '### Graph N: <short title>' and then describe what the chart shows and the key insights.\n"
        "- You do NOT need to embed actual code, only describe the graphs and reference the metrics that could be plotted.\n"
    )
    default_prompt_zh = (
        "你是一位資深醫療器材供應鏈分析顧問。\n"
        "請根據下方的供應商裝箱單與醫院入庫清單，撰寫一份長度約 1000–2000 字的完整分析報告。\n\n"
        "需求：\n"
        "- 著重於供應商與醫院之間的追溯性（包含批號、效期、數量等資訊）。\n"
        "- 說明可能的風險（例如效期風險、數量差異、已出貨但未入庫、已入庫但無出貨紀錄等）。\n"
        "- 提出在品質、安全與法規遵循方面的實務建議。\n"
        "- 使用 Markdown 格式，清楚分段與標題。\n"
        "- 請加入剛好 5 個圖表說明段落。每個圖表以標題開頭，例如：\n"
        "  「### 圖 N：<簡短標題>」，接著描述圖表呈現的指標與主要洞見。\n"
        "- 不需要提供實際程式碼，只要以文字說明圖表的內容與可視化重點。\n"
    )

    default_prompt = default_prompt_en if lang_choice == LANG_EN else default_prompt_zh

    custom_prompt = st.text_area(
        "System-level guidance for the summary (you can modify):",
        value=default_prompt,
        height=220,
        key="supply_summary_custom_prompt",
    )

    if st.button(ui["summary_run"]):
        try:
            # Serialize datasets; keep reasonable size by truncating if huge
            supplier_csv = st.session_state["supplier_df"].to_csv(index=False)
            hospital_csv = st.session_state["hospital_df"].to_csv(index=False)

            prefix_en = (
                "Below are two CSV datasets.\n\n"
                "=== SUPPLIER PACKING LIST (CSV) ===\n"
            )
            prefix_zh = "以下為兩份 CSV 資料集。\n\n=== 供應商裝箱單（CSV） ===\n"

            mid_en = "\n\n=== HOSPITAL INCOMING LIST (CSV) ===\n"
            mid_zh = "\n\n=== 醫院入庫清單（CSV） ===\n"

            user_prompt = (
                (prefix_en if lang_choice == LANG_EN else prefix_zh)
                + supplier_csv
                + (mid_en if lang_choice == LANG_EN else mid_zh)
                + hospital_csv
            )

            system_prompt = custom_prompt
            with st.spinner("Generating supply chain summary with selected model..."):
                result = call_llm(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(max_tokens),
                )
            st.session_state["supply_summary_md"] = result
            st.success("Supply chain summary generated.")
        except Exception as e:
            st.error(f"Failed to generate summary: {e}")

    if st.session_state["supply_summary_md"]:
        st.markdown(f"**{ui['summary_latest']}**")
        edited = st.text_area(
            "",
            value=st.session_state["supply_summary_md"],
            key="supply_summary_md_editor",
            height=420,
        )
        st.session_state["supply_summary_md"] = edited
        # Render as markdown preview
        st.markdown("---")
        view_mode = st.radio(
            ui["agent_result_view"],
            options=[ui["text_view"], ui["markdown_view"]],
            horizontal=True,
            key="supply_summary_view_mode",
        )
        if view_mode == ui["markdown_view"]:
            st.markdown(edited)
        else:
            st.text(edited)

    st.markdown("---")

    # Interactive supply chain graph
    st.markdown(f"### {ui['graph_section']}")
    st.caption(ui["graph_hint"])

    # Filters
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        min_ship = st.number_input(
            ui["min_shipments"],
            min_value=0.0,
            max_value=1e9,
            value=0.0,
            step=1.0,
            key="graph_min_shipments",
        )
    with col_f2:
        dev_ids = []
        if "device_id" in supplier_df.columns:
            dev_ids = sorted(supplier_df["device_id"].dropna().unique())
        if dev_ids:
            device_filter = st.multiselect(
                ui["filter_device"],
                options=dev_ids,
                key="graph_device_filter",
            )
        else:
            device_filter = None

    fig = build_supply_chain_graph(
        supplier_df=supplier_df,
        hospital_df=hospital_df,
        min_quantity=min_ship,
        device_filter=device_filter,
    )
    if fig is None:
        st.info(ui["no_graph_data"])
    else:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Prompt-on-datasets
    st.markdown(f"### {ui['data_chat_section']}")
    data_chat_prompt = st.text_area(
        ui["data_chat_prompt"],
        height=140,
        key="data_chat_prompt",
        value="Identify any discrepancies between shipped and received quantities by device and hospital, "
              "and summarize key risks and operational recommendations.",
    )

    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        data_chat_model = st.selectbox(
            ui["model"],
            options=SUPPORTED_MODELS,
            key="data_chat_model",
        )
    with col_c2:
        data_chat_max_tokens = st.number_input(
            ui["max_tokens"],
            min_value=512,
            max_value=24000,
            value=4000,
            step=512,
            key="data_chat_max_tokens",
        )

    if st.button(ui["data_chat_run"]):
        try:
            supplier_csv = st.session_state["supplier_df"].to_csv(index=False)
            hospital_csv = st.session_state["hospital_df"].to_csv(index=False)

            if st.session_state["lang"] == LANG_EN:
                system_prompt = (
                    "You are a medical device supply chain data analyst. "
                    "You are given two CSV datasets: a supplier packing list and a hospital incoming list. "
                    "Use them to answer the user's question accurately. "
                    "If you compute metrics, explain them clearly in text or markdown."
                )
                user_prompt = (
                    "SUPPLIER PACKING LIST (CSV):\n"
                    + supplier_csv
                    + "\n\nHOSPITAL INCOMING LIST (CSV):\n"
                    + hospital_csv
                    + "\n\nUser question / instruction:\n"
                    + data_chat_prompt
                )
            else:
                system_prompt = (
                    "你是一位醫療器材供應鏈資料分析師。"
                    "你會看到兩份 CSV 資料：供應商裝箱單與醫院入庫清單。"
                    "請善用這些資料回答使用者的問題，並以清楚的文字或 Markdown 說明你的計算與結論。"
                )
                user_prompt = (
                    "【供應商裝箱單（CSV）】\n"
                    + supplier_csv
                    + "\n\n【醫院入庫清單（CSV）】\n"
                    + hospital_csv
                    + "\n\n使用者問題 / 指示：\n"
                    + data_chat_prompt
                )

            with st.spinner("Running ad‑hoc analysis on datasets..."):
                out = call_llm(
                    model=data_chat_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(data_chat_max_tokens),
                )
            st.session_state["data_chat_output"] = out
            st.success("Dataset question answered.")
        except Exception as e:
            st.error(f"Failed to run dataset QA: {e}")

    if st.session_state["data_chat_output"]:
        view_mode = st.radio(
            ui["agent_result_view"],
            options=[ui["text_view"], ui["markdown_view"]],
            horizontal=True,
            key="data_chat_view_mode",
        )
        if view_mode == ui["markdown_view"]:
            st.markdown(st.session_state["data_chat_output"])
        else:
            st.text(st.session_state["data_chat_output"])

    st.markdown("---")

    # Run agents.yaml agents on datasets
    st.markdown(f"### {ui['data_agent_section']}")
    agents = st.session_state["agents_config"]["agents"]
    if not agents:
        st.info("No agents loaded from agents.yaml.")
        return

    agent_options = {f"{a.get('id')} – {a.get('name')}": a for a in agents}
    selected_label = st.selectbox(
        ui["data_agent_select"],
        options=list(agent_options.keys()),
        key="data_agent_select",
    )
    selected_agent = agent_options[selected_label]

    st.markdown(
        f"- **ID:** {selected_agent.get('id')}\n"
        f"- **Name:** {selected_agent.get('name')}\n"
        f"- **Role:** {selected_agent.get('role')}\n"
        f"- **Capabilities:** {', '.join(selected_agent.get('capabilities', []))}"
    )

    col_am1, col_am2 = st.columns([2, 1])
    with col_am1:
        agent_model = st.selectbox(
            ui["model"],
            options=SUPPORTED_MODELS,
            key="data_agent_model",
            index=SUPPORTED_MODELS.index(selected_agent.get("model"))
            if selected_agent.get("model") in SUPPORTED_MODELS
            else 0,
        )
    with col_am2:
        agent_max_tokens = st.number_input(
            ui["max_tokens"],
            min_value=512,
            max_value=24000,
            value=6000,
            step=512,
            key="data_agent_max_tokens",
        )

    default_agent_user_prompt = (
        "You are executing this agent directly on two tabular datasets: "
        "a supplier packing list and a hospital incoming list. "
        "Use your built-in role and capabilities, but ground your reasoning strictly on the data.\n\n"
        "Typical tasks include:\n"
        "- cross‑checking shipment vs. receipt quantities and dates\n"
        "- flagging anomalies and risks\n"
        "- generating structured findings or recommendations\n"
        "- preparing data for downstream agents\n\n"
        "You may respond in markdown or plain text as appropriate.\n"
    )
    data_agent_user_prompt = st.text_area(
        ui["data_prompt_with_context"],
        value=default_agent_user_prompt,
        height=200,
        key="data_agent_user_prompt",
    )

    if st.button(ui["data_agent_run"]):
        try:
            supplier_csv = st.session_state["supplier_df"].to_csv(index=False)
            hospital_csv = st.session_state["hospital_df"].to_csv(index=False)

            system_prompt = selected_agent.get("system_prompt", "") or (
                "You are an AI agent defined in agents.yaml. "
                "You are now applied to medical device supply chain datasets "
                "to perform analysis or transformation according to the user instructions."
            )
            user_prompt = (
                data_agent_user_prompt
                + "\n\n=== SUPPLIER PACKING LIST (CSV) ===\n"
                + supplier_csv
                + "\n\n=== HOSPITAL INCOMING LIST (CSV) ===\n"
                + hospital_csv
            )

            with st.spinner("Running selected agent on datasets..."):
                out = call_llm(
                    model=agent_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=int(agent_max_tokens),
                )
            st.session_state["last_agent_output"] = out
            st.session_state["handoff_buffer"] = out
            st.session_state["handoff_buffer_widget"] = out
            st.success("Agent execution on datasets completed.")

            view_mode = st.radio(
                ui["agent_result_view"],
                options=[ui["text_view"], ui["markdown_view"]],
                key="data_agent_view_mode",
                horizontal=True,
            )
            if view_mode == ui["markdown_view"]:
                st.markdown(out)
            else:
                st.text(out)
        except Exception as e:
            st.error(f"Failed to run agent on datasets: {e}")


# ---------------------
# API Key Config (UI)
# ---------------------

def render_api_key_section():
    ui = get_ui_text()
    st.subheader(ui["api_section"])
    st.caption(ui["api_hint"])

    keys = st.session_state["api_keys"]

    def masked_placeholder(value: Optional[str]) -> str:
        if value:
            return "******** (from env / stored)"
        return ""

    openai_input = st.text_input(
        ui["openai_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("openai")),
    )
    gemini_input = st.text_input(
        ui["gemini_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("gemini")),
    )
    anthropic_input = st.text_input(
        ui["anthropic_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("anthropic")),
    )
    grok_input = st.text_input(
        ui["grok_key"],
        type="password",
        placeholder=masked_placeholder(keys.get("grok")),
    )

    if st.button(get_ui_text()["save_keys"]):
        if openai_input.strip():
            keys["openai"] = openai_input.strip()
        if gemini_input.strip():
            keys["gemini"] = gemini_input.strip()
        if anthropic_input.strip():
            keys["anthropic"] = anthropic_input.strip()
        if grok_input.strip():
            keys["grok"] = grok_input.strip()
        st.session_state["api_keys"] = keys
        st.success("API keys updated for this session.")


# ----------------------
# Layout & Main Entrypoint
# ----------------------

def sidebar_controls():
    ui = get_ui_text()
    st.sidebar.markdown("### UI")

    theme = st.sidebar.radio(
        ui["theme"],
        options=["light", "dark"],
        key="theme_widget",
        horizontal=True,
        index=0 if st.session_state["theme"] == "light" else 1,
    )
    st.session_state["theme"] = theme

    lang = st.sidebar.radio(
        ui["language"],
        options=[LANG_EN, LANG_ZH],
        key="lang_widget",
        horizontal=True,
        index=0 if st.session_state["lang"] == LANG_EN else 1,
    )
    st.session_state["lang"] = lang

    apply_custom_theme()
    ui = get_ui_text()

    st.sidebar.markdown("### 🎨 Style")
    st.sidebar.selectbox(
        ui["painter_style"],
        options=PAINTER_STYLES,
        index=PAINTER_STYLES.index(st.session_state["painter_style"])
        if st.session_state["painter_style"] in PAINTER_STYLES
        else 0,
        key="painter_style",
    )

    if st.sidebar.button(ui["jackpot"]):
        st.session_state["painter_style"] = random.choice(PAINTER_STYLES)

    st.sidebar.markdown("---")
    if st.sidebar.button(get_ui_text()["refresh_config"]):
        refresh_config()
        st.sidebar.success("Config reloaded.")


def main():
    st.set_page_config(
        page_title="Agentic AI Project Orchestrator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    init_api_keys_from_env()
    apply_custom_theme()
    refresh_config()

    sidebar_controls()
    ui = get_ui_text()

    st.markdown(
        f"""
        <div class="wow-card" style="margin-bottom:1.2rem;">
          <div class="wow-pill">Agentic AI • Multi-LLM • Visualization</div>
          <div class="wow-title" style="margin-top:0.4rem;">{ui["title"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_wow_status()

    tabs = st.tabs(
        [
            ui["dashboard"],
            ui["supply_tab"],
            ui["agents_tab"],
            ui["chat_tab"],
            ui["skills_tab"],
            ui["config"],
        ]
    )

    # Project dashboard
    with tabs[0]:
        run_orchestrator_ui()
        plan = st.session_state["project_plan"]
        if plan:
            st.markdown("---")
            render_work_breakdown(plan)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                render_timeline(plan)
            with col2:
                render_agent_matrix(plan)
            st.markdown("---")
            col3, col4 = st.columns(2)
            with col3:
                render_risk_heatmap(plan)
            with col4:
                render_dependency_graph(plan)

    # Medical supply chain tab
    with tabs[1]:
        render_supply_chain_tab()

    # Agents & execution tab
    with tabs[2]:
        render_agents_tab()

    # Refinement / prompt on plan
    with tabs[3]:
        render_refinement_tab()

    # Skills
    with tabs[4]:
        st.subheader(ui["skills_tab"])
        skills = st.session_state["skills"]
        if not skills:
            st.info("No skills parsed from SKILL.md.")
        else:
            for sid, s in skills.items():
                with st.expander(f"{sid}"):
                    st.markdown(f"**Description:** {s.get('description','')}")
                    if s.get("parameters"):
                        st.markdown(f"**Parameters:** {s.get('parameters','')}")
                    if s.get("raw"):
                        st.code(s["raw"], language="markdown")

    # Config / API keys
    with tabs[5]:
        render_api_key_section()
        st.markdown("---")
        st.markdown("**Raw agents.yaml preview:**")
        st.json(st.session_state["agents_config"], expanded=False)


if __name__ == "__main__":
    main()
