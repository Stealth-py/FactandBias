import plotly.express as px
import requests as r
from .cfg import ROOT

def plotpie(result):
    fig = px.pie(result, values='Scores', names='Factuality', color = "Factuality")
    return fig

def plotbar(result):
    fig = px.bar(result, x='Bias', y='Scores', color = 'Bias')
    return fig

def make_request(url):
    return r.get(ROOT+'parse', params={'url':url})
