import plotly.express as px
import requests as r
import os
from .cfg import ROOT
from functools import lru_cache
import datetime, json
import numpy as np
import pandas as pd

agg_pq = pd.read_parquet("data/media_agg.parquet")

pq = pd.read_parquet("data/media_agg_subtask3.parquet")
pq.columns = [i.replace('labels.', '') for i in pq.columns.tolist()]
with open("src/text_mapping.json", 'r') as f:
    tags_mapping = json.loads(f.read())
guns_jsons = [i for i in os.listdir("src") if 'guns' in i]
gpt_guns = {}
for js in guns_jsons:
    with open(f"src/{js}", "r") as f:
        j = json.loads(f.read())
        gpt_guns.update(j)
env_jsons = [i for i in os.listdir("src") if 'env' in i]
gpt_env = {}
for js in env_jsons:
    with open(f"src/{js}", "r") as f:
        j = json.loads(f.read())
        gpt_env.update(j)

business_jsons = [i for i in os.listdir("src") if 'business' in i]
gpt_business = {}
for js in business_jsons:
    with open(f"src/{js}", "r") as f:
        j = json.loads(f.read())
        gpt_business.update(j)

healthcare_jsons = [i for i in os.listdir("src") if 'healthcare' in i]
gpt_healthcare = {}
for js in healthcare_jsons:
    with open(f"src/{js}", "r") as f:
        j = json.loads(f.read())
        gpt_healthcare.update(j)

immigration_jsons = [i for i in os.listdir("src") if 'immigration' in i]
gpt_immigration = {}
for js in immigration_jsons:
    with open(f"src/{js}", "r") as f:
        j = json.loads(f.read())
        gpt_immigration.update(j)

with open("data/linkmapping.json", "r") as file:
  basemapped_to_link = json.loads(file.read())

def get_parq(news_src):
    news_src = basemapped_to_link[news_src]
    try:
        return dict(agg_pq.loc[news_src]), dict(pq.loc[news_src])
    except KeyError as e:
        return {}, {}

def plotfact(result):
    fig = px.pie(result, values='Scores', names='Factuality',
                color = "Factuality",
                color_discrete_map={
                    "Less Factual": "red",
                    "Mixed Factuality": "cyan",
                    "Highly Factual": "green"
                },
                category_orders={"Factuality": ["Less Factual", "Mixed Factuality", "Highly Factual"]},)
    return fig

def plotbias(result):
    fig = px.bar(result, x='Bias', y='Scores',
                color = "Bias",
                color_discrete_map={
                    "Left": "blue",
                    "Center": "purple",
                    "Right": "red"
                },
                category_orders={"Bias": ["Left", "Center", "Right"]})
    fig.update_layout(
        xaxis = dict(
            tickvals = list(range(0, 1))
        )
    )
    return fig

def plotiden(result):
    fig = px.bar(x=list(result.values()), y=list(result.keys()), color = list(result.keys()), color_discrete_sequence = px.colors.sequential.Viridis, orientation = 'h')
    fig.update_layout(
        autosize=False,
        height=600,
        xaxis = dict(
            tickvals = list(range(0, 1))
        )
    )
    return fig
    # fig = px.pie(values=list(result.values()), names=list(result.keys()))
    # return fig

def plotpers(result):
    fig = px.bar(x=list(result.values()), y=list(result.keys()), orientation = 'h')
    fig.update_layout(
        autosize=False,
        height=800,
        xaxis = dict(
            tickvals = list(range(0, 1))
        )
    )
    return fig

def aggr_scores(results):
    biasscores = []
    factscores = []

    aggregatedBiasScores = []
    aggregatedFactScores = []
    nelas = []
    times = []
    for i in range(len(results)):
        biasresults = results[i]['bias_results']
        factresults = results[i]['factuality_results']
        nelas.append(results[i]['nela'])
        if results[i].get('date_added') is not None:
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
        "date": max(times) if len(times) > 0 else datetime.datetime.now(),
        'nela': calculate_mean_per_key(nelas),
    }

    return finalResult


def make_request(url, is_forced=False):
    return r.get(ROOT+'parse', params={'url':url, 'is_forced':is_forced})


@lru_cache(32)
def get_base_urls():
    return r.get(ROOT+'urls').json() + list(basemapped_to_link.keys())


def get_tags_by_source(source):
    return tags_mapping.get(source)


def get_gpt(source):
    if not source in gpt_guns:
        return None
    return {
        "Gun Rights": gpt_guns.get(source) if not len(gpt_guns.get(source).split())>=5 else "Unclear",
        "Environmental Policy": gpt_env.get(source) if not len(gpt_env.get(source).split())>=5 else "Unclear",
        "Worker’s/Business Rights": gpt_business.get(source) if not len(gpt_business.get(source).split())>=5 else "Unclear",
        "Healthcare": gpt_healthcare.get(source) if not len(gpt_healthcare.get(source).split())>=5 else "Unclear",
        "Immigration": gpt_immigration.get(source) if not len(gpt_immigration.get(source).split())>=5 else "Unclear"
    }

def calculate_mean_per_key(json_list):
    key_sum = {}
    key_count = {}
    print(json_list)
    # Iterate through each JSON in the list
    for data in json_list[0]:

        # Iterate through each key-value pair in the JSON
        for key, value in data.items():
            # Check if the value is a number
            if isinstance(value, (int, float)):
                # Update the sum and count for the key
                key_sum[key] = key_sum.get(key, 0) + value
                key_count[key] = key_count.get(key, 0) + 1

    # Calculate the mean for each key
    mean_per_key = {}
    for key, total_sum in key_sum.items():
        mean_per_key[key] = total_sum / key_count[key]

    return mean_per_key
