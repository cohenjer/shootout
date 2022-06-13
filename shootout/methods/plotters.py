# Some custom plotting functions
from matplotlib import pyplot as plt # used for concurrent plots, todo go to plotly
import plotly.express as px


def plot_speed_comparison(thresh_time, scores_time, thresh_it, scores_it, legend=None, title=None, xlabel="Rec error threshold", ylabel="Number of faster runs"):
    fig0 = plt.figure()
    plt.subplot(121)
    plt.semilogx(thresh_time, scores_time.T)
    if legend:
        plt.legend(legend)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.subplot(122)
    plt.semilogx(thresh_it, scores_it.T)
    if legend:
        plt.legend(legend)
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig0


def change_px_fig():
    # some custom function to modify plotly express plots, because I always forget the correct commands
    return

