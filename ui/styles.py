from __future__ import annotations

import streamlit as st

GLOBAL_STYLES = """
<style>
.filter-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: -0.35rem;
}
.filter-chip {
    display: inline-flex;
    align-items: center;
    border: 1px solid rgba(49, 51, 63, 0.18);
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
    background: rgba(49, 51, 63, 0.04);
    color: rgb(49, 51, 63);
    font-size: 0.85rem;
    line-height: 1.35;
}
.rating-key {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin: -0.25rem 0 0.85rem;
    color: rgba(49, 51, 63, 0.76);
    font-size: 0.84rem;
}
.rating-key-item {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    white-space: nowrap;
}
.rating-key-dot {
    width: 0.62rem;
    height: 0.62rem;
    border-radius: 999px;
    box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.12);
}
.detail-row {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    border-bottom: 1px solid rgba(49, 51, 63, 0.12);
    padding: 0.45rem 0;
    font-size: 0.92rem;
}
.detail-row span {
    color: rgba(49, 51, 63, 0.66);
}
.detail-row strong {
    max-width: 68%;
    text-align: right;
    overflow-wrap: anywhere;
}
.gap-metric {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    min-height: 4.5rem;
    justify-content: center;
}
.gap-metric span {
    color: rgba(49, 51, 63, 0.66);
    font-size: 0.88rem;
}
.gap-metric strong {
    font-size: 1.85rem;
    line-height: 1.15;
}
.gap-metric.positive strong {
    color: #1a9850;
}
.gap-metric.negative strong {
    color: #d73027;
}
.gap-metric.neutral strong {
    color: #8c8c8c;
}
@media (prefers-color-scheme: dark) {
    .filter-chip {
        border-color: rgba(250, 250, 250, 0.2);
        background: rgba(250, 250, 250, 0.08);
        color: rgb(250, 250, 250);
    }
    .rating-key {
        color: rgba(250, 250, 250, 0.78);
    }
    .detail-row {
        border-bottom-color: rgba(250, 250, 250, 0.16);
    }
    .detail-row span,
    .gap-metric span {
        color: rgba(250, 250, 250, 0.68);
    }
}
</style>
"""


def render_global_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)
