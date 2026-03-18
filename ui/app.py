"""Streamlit UI for Customer Support Agents — chat and view traces."""
import pathlib
import yaml
import streamlit as st
from dotenv import load_dotenv

from pages import agent_list, chat, traces, compare

# Load .env for AZURE_OPENAI_* vars used by the evaluator
load_dotenv(pathlib.Path(__file__).parent / ".env")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agent Playground", page_icon="🤖", layout="wide")

# ── Load config ──────────────────────────────────────────────────────────────
CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


@st.cache_data
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()
agents = config["agents"]
workspace_id = config["app_insights_workspace_id"]

# ── Session defaults ─────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "agents"
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Agent Playground")
    st.divider()

    if st.button("← Agent List", use_container_width=True):
        st.session_state.page = "agents"
        st.session_state.selected_agent = None
        st.session_state.messages = []
        st.rerun()

    if st.button("🔍 Compare Agents", use_container_width=True):
        st.session_state.page = "compare"
        st.rerun()

    selected = st.session_state.selected_agent
    if selected:
        st.markdown(f"**Active:** {selected['icon']} {selected['name']}")
        st.divider()

        page = st.radio(
            "Navigate",
            ["💬 Chat", "📊 Traces"],
            label_visibility="collapsed",
        )
        if page == "💬 Chat":
            st.session_state.page = "chat"
        elif page == "📊 Traces":
            st.session_state.page = "traces"

# ── Page routing ─────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "compare":
    compare.render(workspace_id)
elif page == "agents" or st.session_state.selected_agent is None:
    agent_list.render(agents)
elif page == "chat":
    chat.render(st.session_state.selected_agent)
elif page == "traces":
    traces.render(st.session_state.selected_agent, workspace_id)
