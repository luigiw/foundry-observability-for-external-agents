"""Agent list page — select an agent to chat with."""
import streamlit as st


def render(agents: list[dict]):
    st.title("🤖 Customer Support Agents")
    st.caption("Select an agent to start chatting.")

    cols = st.columns(len(agents))
    for col, agent in zip(cols, agents):
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #444; border-radius:12px; padding:24px; text-align:center;">
                    <div style="font-size:48px;">{agent['icon']}</div>
                    <h3>{agent['name']}</h3>
                    <p style="color:#888; font-size:14px;">{agent['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Chat →", key=f"select_{agent['id']}", use_container_width=True):
                st.session_state.selected_agent = agent
                st.session_state.page = "chat"
                st.session_state.messages = []
                st.rerun()
