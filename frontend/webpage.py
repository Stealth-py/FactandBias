import plot_utils as tp
import streamlit as st
import numpy as np

VALID_SRC = False

st.set_page_config(layout="wide")

def valid_source(news_src: str = "BBC"):
    global VALID_SRC

    if news_src == "BBC":
        VALID_SRC = True
    else:
        VALID_SRC = False
    
    return

def plot1(fig1, fig2):
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

with st.form("news_inp", clear_on_submit = True):
    news_src = st.text_input(
        "Enter the news source here ",
        label_visibility=st.session_state.visibility,
        placeholder="Natural News",
    )
    
    st.form_submit_button(label = "Analyse", on_click = valid_source(news_src))

if VALID_SRC:
    fig1 = tp.get_plotscat()
    fig2 = tp.get_plotbar()
    fig3 = tp.get_plotpie()
    
    with st.container():
        st.markdown("<h3 style='text-align: center; color: black;'>BBC News</h3>", unsafe_allow_html=True)

    plot1(fig1, fig3)
    # with st.container():
    #     st.plotly_chart(fig2, use_container_width=True)