from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Halchemy Album Dashboard",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


from ui.dashboard import main


if __name__ == "__main__":
    main()
