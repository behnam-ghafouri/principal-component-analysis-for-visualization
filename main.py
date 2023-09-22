import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pca_utils import plot_widget
from bokeh.io import show, output_notebook
from bokeh.plotting import figure
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go

X = np.array([[99, -1],
              [98, -1],
              [97, -2],
              [101, 1],
              [102, 1],
              [103, 2]])
def display_chart(x_):
    if len(x_[0]) > 1:
        fig = px.scatter(x=x_[:, 0], y=x_[:, 1])
    else:
        fig = px.scatter(x=x_[:, 0])
    fig.show()

#display_chart(X)

pca_2 = PCA(n_components=2)
pca_2.fit(X)
X_trans_2 = pca_2.transform(X)
#display_chart(X_trans_2)
X_trans_2 = pca_2.inverse_transform(X_trans_2)
#display_chart(X_trans_2)

pca_1 = PCA(n_components=1)
pca_1.fit(X)
X_trans_1 = pca_1.transform(X)
display_chart(pca_1.inverse_transform(X_trans_1))
display_chart(X)