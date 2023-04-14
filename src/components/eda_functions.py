# Import Libraries
import random
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass

from plotly import graph_objs as go
from plotly import express as px
from plotly import figure_factory as ff
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from src.utils import load_object
# /Users/keval_78/Keval/Data Science/Loyalist/Term 4/AIP/Credit-Card-Fraud-Detection/artifacts/charts/Pieplot

@dataclass
class EDAConfig:
    charts = os.path.join('artifacts', "charts")
eda_config = EDAConfig()

def get_random_color():
    colors = ['black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'firebrick', 'forestgreen', 'fuchsia', 'goldenrod', 'gray', 'grey', 'green', 'honeydew', 'indianred', 'indigo', 'khaki', 'lawngreen', 'maroon', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumslateblue', 'midnightblue', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'peru', 'purple', 'rebeccapurple', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'slateblue', 'slategray', 'slategrey', 'steelblue', 'tan', 'teal', 'tomato']
    color = random.choice(colors)
    return color

def pieplot(title_):
    fig = load_object(os.path.join(eda_config.charts, "Pieplot"))
    fig.update_layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' }) 
    return fig

def barplot(title_):
    fig = load_object(os.path.join(eda_config.charts, "Barplot"))
    fig.update_layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' }) 
    return fig

def histplots(title1, title2):
    fig1 = load_object(os.path.join(eda_config.charts, "Barplot-Time"))
    fig1.update_layout(title={'text': f'<b>{title1}</b>'})
    fig2 = load_object(os.path.join(eda_config.charts, "Barplot-Amount"))
    fig2.update_layout(title={'text': f'<b>{title2}</b>'})
    return fig1, fig2

def histplots_fraud(title1, title2):
    fig1 = load_object(os.path.join(eda_config.charts, "Barplot-Time-Fraud"))
    fig1.update_layout(title={'text': f'<b>{title1}</b>'})
    fig2 = load_object(os.path.join(eda_config.charts, "Barplot-Amount-Fraud"))
    fig2.update_layout(title={'text': f'<b>{title2}</b>'})
    return fig1, fig2

def tsne_plot_original(title_):
    fig = load_object(os.path.join(eda_config.charts, "TSNEplot-Original"))
    fig.update_layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' }) 
    return fig

def tsne_plot_smoteenn(title_):
    fig = load_object(os.path.join(eda_config.charts, "TSNEplot-SMOTE+ENN"))
    fig.update_layout(title={'text': f'<b>{title_}</b>', 'x': 0.5, 'y': 0.9 , 'xanchor': "center", 'yanchor': 'top' })
    return fig

def correlation_heatmap_original(title_):
    fig = load_object(os.path.join(eda_config.charts, "Heatmap-Original"))
    fig.update_layout(title={'text': f'<b>{title_}</b>'})
    return fig

def correlation_heatmap_smoteenn(title_):
    fig = load_object(os.path.join(eda_config.charts, "Heatmap-SMOTE+ENN"))
    fig.update_layout(title={'text': f'<b>{title_}</b>'})
    return fig

def confusion_matrix():
    fig = load_object(os.path.join(eda_config.charts, "ConfusionMatrix"))
    return fig

def roc_curve():
    fig = load_object(os.path.join(eda_config.charts, "ROCcurve"))
    return fig