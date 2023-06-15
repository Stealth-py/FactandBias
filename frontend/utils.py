import plotly.express as px
import requests as r
from .cfg import ROOT
from functools import lru_cache
def plotpie(result):
    fig = px.pie(result, values='Scores', names='Factuality',
                color = "Factuality",
                color_discrete_map={
                    "Less Factual": "red",
                    "Mixed Factuality": "cyan",
                    "Highly Factual": "green"
                },
                category_orders={"Factuality": ["Less Factual", "Mixed Factuality", "Highly Factual"]},)
    return fig

def plotbar(result):
    fig = px.bar(result, x='Bias', y='Scores',
                color = "Bias",
                color_discrete_map={
                    "Left": "green",
                    "Center": "gray",
                    "Right": "blue"
                },
                category_orders={"Bias": ["Left", "Center", "Right"]})
    fig.update_layout(
        xaxis = dict(
            tickvals = list(range(0, 1))
        )
    )
    return fig


def make_request(url):
    return r.get(ROOT+'parse', params={'url':url})
@lru_cache(32)
def get_base_urls():
    return r.get(ROOT+'urls').json()
