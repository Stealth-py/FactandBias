import frontend.utils as tp
import streamlit as st
import numpy as np
import pandas as pd
import torch, requests, time
from frontend.cfg import ROOT
VALID_SRC = False

factscores = np.random.randn(1, 2)
biasscores = np.random.randn(1, 3)

factscores = torch.nn.Softmax(dim = -1)(torch.from_numpy(factscores))
factresult = pd.DataFrame({
    'Factuality':['Factual', 'Not Factual'],
    'Scores': factscores[0].tolist()
})

biasscores = torch.nn.Softmax(dim = -1)(torch.from_numpy(biasscores))
biasresult = pd.DataFrame({
    'Bias':['Left', 'Center', 'Right'],
    'Scores': biasscores[0].tolist()
})

def valid_url(news_src: str = "https://www.bbc.com/news"):
    """
    Check the validity of the News URL entered.
    
    Args:
        news_src (str): The URL of the webpage to analyse.
    """
    global VALID_SRC

    try:
        request = requests.get(news_src)
        if request.status_code != 200:
            VALID_SRC = False
        else:
            VALID_SRC = True
    except:
        VALID_SRC = False

    return

def plot(barfig, piefig):
    """
    Plots the bar and the pie charts on the webpage.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(barfig, use_container_width=True)
    
    with col2:
        st.plotly_chart(piefig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    with st.form("news_inp", clear_on_submit = True):
        news_src = st.text_input(
            "Enter the news source here ",
            label_visibility=st.session_state.visibility,
            placeholder="https://www.bbc.com/news",
        )
        
        st.form_submit_button(label = "Analyse", on_click = valid_url(news_src))

    if VALID_SRC:
        with st.empty():
            with st.spinner('Scraping...'):
                tp.make_request(news_src)
            barfig = tp.plotbar(biasresult)
            piefig = tp.plotpie(factresult)

            st.markdown(f"<h3 style='text-align: center; color: black;'>{news_src}</h3>", unsafe_allow_html=True)

            plot(barfig, piefig)