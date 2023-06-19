import datetime

import frontend.utils as tp
import streamlit as st
import numpy as np
import pandas as pd
import torch, requests, time
import numpy as np
from frontend.cfg import ROOT
from streamlit_searchbox import st_searchbox
from thefuzz import process, fuzz
VALID_SRC = False

def search(searchterm: str, ):
    base_urls = tp.get_base_urls()
    return [i[0] for i in process.extract(searchterm, base_urls + [searchterm], scorer=fuzz.ratio)[:5]]

def valid_url(news_src: str = "https://www.bbc.com/news"):
    """
    Check the validity of the News URL entered.
    
    Args:
        news_src (str): The URL of the webpage to analyse.
    """
    global VALID_SRC
    print(news_src)
    try:
        request = requests.get(news_src)
        if request.status_code != 200:
            VALID_SRC = False
        else:
            VALID_SRC = True
    except:
        VALID_SRC = False

    return

def plot_fact_bias(biasfig, factfig, source, date):
    """
    Plots the bar and the pie charts on the webpage.
    """
    st.markdown(f"<h3 style='text-align: center; color: black;'>{source} (updated at {date})</h3>", unsafe_allow_html=True)
    st.write("Bias Scores")
    st.plotly_chart(biasfig, use_container_width=True)
    st.write("\nFactuality Results")
    st.plotly_chart(factfig, use_container_width=True)

def plot_ident_pers(identfig, persfig):
    st.write("Identity Framing Results")
    st.plotly_chart(identfig, use_container_width=True)
    st.write("\nPersuasion Results")
    st.plotly_chart(persfig, use_container_width=True)


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    news_src = st_searchbox(
        search,
        label="Enter and select the news source from here.",
        key = "news_searchbox",
        placeholder="https://www.bbc.com/news",
    )

    valid_url(news_src)

    if VALID_SRC:
        main_empty = st.empty()
        with main_empty.container():
            with st.spinner('Scraping...'):
                results = tp.make_request(news_src).json()

            results = tp.aggr_scores(results)

            print('\n\n\n', results)

            biasfig = tp.plotbias(results['bias_results'])
            factfig = tp.plotfact(results['factuality_results'])

            identity_results, persuasion_results = tp.get_parq(news_src = news_src)
            is_identity_persuasion = True if identity_results else False

            if not is_identity_persuasion:
                st.write("Identity Framing and Persuasion Results were not found in the database. Displaying Factuality and Bias Results only.")
                plot_fact_bias(biasfig, factfig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))
            else:
                fig_col1, fig_col2 = st.columns(2)

                identfig = tp.plotiden(identity_results)
                persfig = tp.plotpers(persuasion_results)

                is_reanalyse = st.button("Reanalyse", key = "reanalysebutton", on_click = valid_url(news_src))
                plot_fact_bias(biasfig, factfig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))
                
                plot_ident_pers(identfig, persfig)

                if is_reanalyse:
                    pass
                    # with st.spinner("Reanalysing..."):
                    #     tp.reanalyse(news_src)
