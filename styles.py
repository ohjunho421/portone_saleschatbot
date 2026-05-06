"""PortOne brand-aligned CSS for the Streamlit app."""
from __future__ import annotations

CUSTOM_CSS: str = """
<style>
  :root {
    --po-primary: #2E5BFF;
    --po-primary-strong: #1E3FCC;
    --po-accent: #FF6B3D;
    --po-bg: #F7F9FC;
    --po-surface: #FFFFFF;
    --po-text: #0F172A;
    --po-text-muted: #475569;
    --po-border: #E2E8F0;
    --po-success: #16A34A;
    --po-radius: 14px;
    --po-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
  }

  /* App background */
  .stApp {
    background:
      radial-gradient(1200px 600px at 90% -10%, rgba(46,91,255,0.10), transparent 60%),
      radial-gradient(900px 500px at -10% 10%, rgba(255,107,61,0.08), transparent 55%),
      var(--po-bg);
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 6rem; max-width: 920px; }

  /* Hero */
  .po-hero {
    display: flex; align-items: center; gap: 16px;
    padding: 22px 26px; border-radius: var(--po-radius);
    background: linear-gradient(135deg, var(--po-primary) 0%, #5C7CFF 100%);
    color: #fff; box-shadow: var(--po-shadow); margin-bottom: 18px;
  }
  .po-hero .po-logo {
    width: 44px; height: 44px; border-radius: 12px;
    background: rgba(255,255,255,0.18);
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 18px; letter-spacing: -0.5px;
  }
  .po-hero h1 { margin: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.3px; }
  .po-hero p { margin: 2px 0 0; font-size: 0.92rem; opacity: 0.92; }

  /* Cards / panels */
  .po-card {
    background: var(--po-surface); border: 1px solid var(--po-border);
    border-radius: var(--po-radius); padding: 14px 16px;
    box-shadow: var(--po-shadow);
  }

  /* Suggested-question chips */
  .po-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 10px; }
  div[data-testid="stHorizontalBlock"] .stButton > button {
    background: var(--po-surface);
    color: var(--po-text);
    border: 1px solid var(--po-border);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all .15s ease;
    box-shadow: none;
  }
  div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: var(--po-primary);
    color: var(--po-primary);
    transform: translateY(-1px);
  }

  /* Chat messages */
  [data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
  }
  [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: var(--po-surface);
    border: 1px solid var(--po-border);
    border-radius: var(--po-radius);
    padding: 14px 16px;
    box-shadow: var(--po-shadow);
    line-height: 1.65;
  }
  /* User bubble accent */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    background: linear-gradient(180deg, #EEF2FF 0%, #E8EEFF 100%);
    border-color: #C7D3FF;
  }

  /* Source pills */
  .po-sources { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
  .po-source-title {
    font-size: 0.78rem; font-weight: 600; color: var(--po-text-muted);
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .po-source-item {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F1F5F9; border: 1px solid var(--po-border);
    color: var(--po-text); padding: 4px 10px; border-radius: 999px;
    font-size: 0.82rem; max-width: 100%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .po-source-item a { color: var(--po-primary); text-decoration: none; }
  .po-source-item a:hover { text-decoration: underline; }
  .po-source-icon { font-size: 12px; }

  /* Chat input */
  [data-testid="stChatInput"] {
    border-radius: var(--po-radius) !important;
    box-shadow: var(--po-shadow);
    border: 1px solid var(--po-border) !important;
  }
  [data-testid="stChatInput"] textarea { font-size: 0.95rem; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: var(--po-surface);
    border-right: 1px solid var(--po-border);
  }
  section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid var(--po-border);
    background: var(--po-surface);
    font-weight: 500;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--po-primary);
    color: var(--po-primary);
  }

  /* Status badge */
  .po-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  }
  .po-badge--ok { background: #DCFCE7; color: #166534; }
  .po-badge--warn { background: #FEF3C7; color: #92400E; }
  .po-badge--err { background: #FEE2E2; color: #991B1B; }
  .po-badge .dot {
    width: 6px; height: 6px; border-radius: 50%; background: currentColor;
  }

  /* Empty state */
  .po-empty {
    text-align: center; padding: 28px 12px; color: var(--po-text-muted);
    border: 1px dashed var(--po-border); border-radius: var(--po-radius);
    background: var(--po-surface);
  }
  .po-empty h3 { margin: 0 0 6px; color: var(--po-text); font-size: 1.05rem; }
  .po-empty p { margin: 0; font-size: 0.9rem; }
</style>
"""
