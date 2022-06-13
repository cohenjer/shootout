from shootout.methods.plotters import plot_speed_comparison
from shootout.methods.post_processors import find_best_at_all_thresh
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Testing plot_speed_comparison
df = pd.read_pickle("run-example")
# note: df has a column "full_error" and "full_time" which we will use for finding the performance and various thresholds
thresh = np.logspace(-3,-8,50) 
scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, 5,
                                                err_name="full_error", time_name="full_time")

fig = plot_speed_comparison(thresh,scores_time,thresh,scores_it)
plt.show()