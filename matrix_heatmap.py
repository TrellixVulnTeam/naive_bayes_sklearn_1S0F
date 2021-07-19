import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np

@st.cache(hash_funcs={mpl.figure.Figure: lambda _: None})
def matrix_heatmap(matrix, options={'x_labels': [], 'y_labels': [], 'annotation_format': 'd', 'color_map': 'Blues', 'custom_range': True, 'vmin_vmax': (0,1), 'center': 0, 'title_axis_labels': ('Default Title', 'Default x-axis label', 'Default y-axis label'), 'rotate x_tick_labels': False}):

    # Create matrix figure
    fig, ax = plt.subplots()

    # Resize the figure if dimension is larger than a cutoff so that heatmap annotations
    # do not overflow their cells (chosen via testing to be 7)
    max_dimension = max(len(options['x_labels']), len(options['y_labels']))
    if max_dimension >= 7:
        fig.set_size_inches(max_dimension, max_dimension)
    
    # ----------------------------------
    # ----- Create seaborn heatmap -----
    # ----------------------------------

    # For custom colorbar, note cbar=False keyword is used to prevent duplicate colorbars

    # Set custom vmin, vmax if 'custom_range' option is True
    if options['custom_range']:
        ax = sns.heatmap(matrix, annot=True, fmt=options['annotation_format'], ax = ax, cmap=options['color_map'], vmin=options['vmin_vmax'][0], vmax=options['vmin_vmax'][1], center=options['center'], square=True, cbar=False)
    else:
        ax = sns.heatmap(matrix, annot=True, fmt=options['annotation_format'], ax = ax, cmap=options['color_map'], center=options['center'], square=True, cbar=False)
    
    # Format colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = plt.colorbar(ax.collections[0], cax=cax)
    cbar.outline.set_edgecolor('black')

    # make heatmap frame visible
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    
    # title, axis labels, and ticks
    ax.set_title(options['title_axis_labels'][0])
    ax.set_xlabel(options['title_axis_labels'][1])
    ax.set_ylabel(options['title_axis_labels'][2])
    
    # Make y-axis labels horizontal
    ax.yaxis.set_tick_params(rotation=0)

    ax.xaxis.set_ticklabels(options['x_labels'])
    ax.yaxis.set_ticklabels(options['y_labels'])

    # Rotate x-axis ticks and align
    if options['rotate x_tick_labels']:
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right", rotation_mode="anchor")

    # Return the figure
    return fig