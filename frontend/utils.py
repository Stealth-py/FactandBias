import plotly.express as px

def plotpie(result):
    fig = px.pie(result, values='Scores', names='Factuality', color = "Factuality")
    return fig

def plotbar(result):
    fig = px.bar(result, x='Bias', y='Scores', color = 'Bias')
    return fig