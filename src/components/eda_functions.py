# Import Libraries
import random
import numpy as np
import pandas as pd

from plotly import graph_objs as go
from plotly import express as px
from plotly import figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


def get_random_color():
    colors = ['black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'firebrick', 'forestgreen', 'fuchsia', 'goldenrod', 'gray', 'grey', 'green', 'honeydew', 'indianred', 'indigo', 'khaki', 'lawngreen', 'maroon', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumslateblue', 'midnightblue', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'peru', 'purple', 'rebeccapurple', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'slateblue', 'slategray', 'slategrey', 'steelblue', 'tan', 'teal', 'tomato']
    color = random.choice(colors)
    return color


def pieplot(labels, values, title_):
    layout = go.Layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' })
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Pie(labels=labels, values=values, pull=[0.2, 0, 0, 0]))
    return fig


def barplot(labels, values, title_):
    colors = ['orange', 'red']
    layout = go.Layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' })
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(x=labels, y=values, text=values, textposition='auto', marker={'color': colors}))
    return fig


def tsne_plot(df):
    x1, y1 = df.drop('Class', axis=1), df['Class']
    _, x1, _, y1 = train_test_split(x1, y1, test_size=0.01, stratify=y1, random_state=1)
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)
    fig = px.scatter(x=X_t[:, 0], y=X_t[:, 1], color=y1, color_continuous_scale=["yellow", "blue"])
    return fig

def histplot(values, title_):
    layout = go.Layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' })
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Histogram(x=values, bingroup=1, marker=dict(color=get_random_color())))
    return fig

def correlation_heatmap(train_df, title_):
    colorscale = random.choice(['Jet', 'Rainbow', 'Portland', 'Electric', 'Blues', 'Greens', 'YlOrRd'])
    layout = go.Layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' })
    fig = go.Figure(layout=layout)
    corr1 = train_df.corr().values
    fig.add_trace(go.Heatmap(z=corr1, colorscale=colorscale) )
    return fig