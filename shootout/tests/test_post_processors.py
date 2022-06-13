from shootout.methods.post_processors import error_at_time_or_it, df_to_convergence_df, find_best_at_all_thresh
import pandas as pd
import numpy as np


# Testing find_best_at_all_thresh
# TODO: Use smaller example df
# TODO: remove batch parameter/provide clever default

df = pd.read_pickle("run-example")
# note: df has a column "full_error" and "full_time" which we will use for finding the performance and various thresholds
thresh = np.logspace(-3,-8,50) 
scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, 5,
                                                err_name="full_error", time_name="full_time")
# stupid unit test
assert scores_time[0,0] == 50.

# Testing df_to_convergence_df
df2 = df_to_convergence_df(df, err_name="full_error", time_name="full_time", algorithm_name="method",
                            other_names=["NbIter_inner", "m", "n", "r", "noise_variance"], max_time=0.5,
                            nb_seed_used=5, groups=True, groups_names=["r", "n", "m", "noise_variance"])
# another stupid test
assert len(df2) == 452752

# Testing err_at_time_or_it
df = error_at_time_or_it(df, time_stamps=[0.1,0.5,1], it_stamps=[0,10,100], err_name="full_error", time_name="full_time")