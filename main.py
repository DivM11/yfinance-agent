"""
Main entry point for the YFinance Agent application.
"""

import streamlit as st

from src.config import load_config
from src.dashboard import run_dashboard

if __name__ == "__main__":
    config = load_config()
    st.set_page_config(layout=config["app"]["layout"], page_icon="img/finance_icon.png")
    run_dashboard(config)
