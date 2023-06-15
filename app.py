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

def plot(barfig, piefig, source, date):
    """
    Plots the bar and the pie charts on the webpage.
    """
    with st.container():
        st.markdown(f"<h3 style='text-align: center; color: black;'>{source} (updated at {date})</h3>", unsafe_allow_html=True)
        st.plotly_chart(barfig, use_container_width=True)
        st.plotly_chart(piefig, use_container_width=True)

def aggr_scores(results):
    biasscores = []
    factscores = []

    aggregatedBiasScores = []
    aggregatedFactScores = []
    times = []
    for i in range(len(results)):
        biasresults = results[i]['bias_results']
        factresults = results[i]['factuality_results']
        times.append(datetime.datetime.strptime(results[i]['date_added'],
                                                "%Y-%m-%dT%H:%M:%S.%f"))
        biasscores.append(list(biasresults['Scores'].values()))
        factscores.append(list(factresults['Scores'].values()))

    biaslabs = np.array([np.argmax(i) for i in biasscores])
    factlabs = np.array([np.argmax(i) for i in factscores])

    aggregatedBiasScores.append(np.count_nonzero(biaslabs == 0)/len(biaslabs))
    aggregatedBiasScores.append(np.count_nonzero(biaslabs == 1)/len(biaslabs))
    aggregatedBiasScores.append(np.count_nonzero(biaslabs == 2)/len(biaslabs))

    aggregatedFactScores.append(np.count_nonzero(factlabs == 0)/len(factlabs))
    aggregatedFactScores.append(np.count_nonzero(factlabs == 1)/len(factlabs))
    aggregatedFactScores.append(np.count_nonzero(factlabs == 2)/len(factlabs))
    #finalResult = results[0]
    finalResult = {
        'bias_results': {
            "Bias": {"0": "Left", "1": "Center", "2": "Right"},
            "Scores": {"0": aggregatedBiasScores[0], "1": aggregatedBiasScores[1], "2": aggregatedBiasScores[2]}
        },
        'factuality_results': {
            "Factuality": {"0": "Less Factual", "1": "Mixed Factuality", "2": "Highly Factual"},
            "Scores": {"0": aggregatedFactScores[0], "1": aggregatedFactScores[1], "2": aggregatedFactScores[2]}
        },
        "date": max(times),
    }

    return finalResult


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False

    with st.container():
        news_src = st_searchbox(
            search,
            label="Enter and select the news source from here.",
            placeholder="https://www.bbc.com/news",
        )
        st.button(label="Show Scores", on_click=valid_url(news_src))

    if VALID_SRC:
        with st.empty():
            with st.spinner('Scraping...'):
                results = tp.make_request(news_src).json()

            results = aggr_scores(results)

            print('\n\n\n', results)

            barfig = tp.plotbar(results['bias_results'])
            piefig = tp.plotpie(results['factuality_results'])

            st.markdown(f"<h3 style='text-align: center; color: black;'>{news_src}</h3>", unsafe_allow_html=True)

            plot(barfig, piefig, news_src, datetime.datetime.strftime(results['date'],'%Y-%m-%d'))