import plotly.express as px
import requests as r
from .cfg import ROOT

def plotpie(result):
    fig = px.pie(result, x='Scores', y='Factuality',
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
