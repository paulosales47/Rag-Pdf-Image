import streamlit as st


def confirm_icon_action(icon: str, key: str, help: str | None = None) -> bool:
    """Botão-ícone de ação destrutiva, com confirmação compacta em duas etapas."""
    confirm_key = f"confirm::{key}"
    if not st.session_state.get(confirm_key, False):
        if st.button(icon, key=f"{key}_ask", help=help):
            st.session_state[confirm_key] = True
            st.rerun()
        return False

    st.caption("Confirma?")
    c1, c2 = st.columns(2)
    confirmed = c1.button("✅", key=f"{key}_yes")
    if c2.button("❌", key=f"{key}_no"):
        st.session_state[confirm_key] = False
    if confirmed:
        st.session_state[confirm_key] = False
    return confirmed
