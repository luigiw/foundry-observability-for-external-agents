"""Chat page — conversational UI with the selected agent."""
import streamlit as st
from lib.agent_client import invoke_agent


def render(agent: dict):
    st.title(f"{agent['icon']} {agent['name']}")

    # Display chat history
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                meta = msg["metadata"]
                st.caption(
                    f"Handled by **{meta.get('handled_by', '?')}** · "
                    f"Type: `{meta.get('query_type', '?')}` · "
                    f"Escalation: {'⚠️ Yes' if meta.get('needs_escalation') else 'No'}"
                )

    # Chat input
    if prompt := st.chat_input("Type your message…"):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call agent
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking…"):
                try:
                    result = invoke_agent(agent["url"], prompt)
                    response_text = result.get("response", str(result))
                    metadata = result.get("metadata", {})

                    st.markdown(response_text)
                    st.caption(
                        f"Handled by **{metadata.get('handled_by', '?')}** · "
                        f"Type: `{metadata.get('query_type', '?')}` · "
                        f"Escalation: {'⚠️ Yes' if metadata.get('needs_escalation') else 'No'}"
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "metadata": metadata,
                    })
                except Exception as e:
                    st.error(f"Error calling agent: {e}")
