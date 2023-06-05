import plotly.express as px

def get_plotscat():
    df = px.data.iris()
    fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
    return fig

def get_plotbar():
    data_canada = px.data.gapminder().query("country == 'Canada'")
    fig = px.bar(data_canada, x='year', y='pop')
    return fig

def get_plotpie():
    df = px.data.tips()
    fig = px.pie(df, values='tip', names='day')
    return fig