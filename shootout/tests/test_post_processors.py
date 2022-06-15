from shootout.methods.post_processors import error_at_time_or_it, df_to_convergence_df, find_best_at_all_thresh, regroup_columns
import pandas as pd
import numpy as np


# Testing find_best_at_all_thresh
# TODO: Use smaller example df
# TODO: remove batch parameter/provide clever default

df = pd.read_pickle("run-example")
# note: df has a column "full_error" and "full_time" which we will use for finding the performance and various thresholds
thresh = np.logspace(5,2,50) 
scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, 2)
# stupid unit test
assert scores_time[0,0] == 2.

# Testing df_to_convergence_df
df2 = df_to_convergence_df(df, other_names=["U_lines","ranks_0"], max_time=1.5,
                            filters={"seed_idx":[0,1], "U_lines":20}, groups=True, groups_names=["U_lines"])
# another stupid test
assert len(df2) == 2004

# Testing err_at_time_or_it
df = error_at_time_or_it(df, time_stamps=[0.1,0.5,1], it_stamps=[0,10,100])

# Testing regrouping
df = regroup_columns(df,keys=["ranks"], how_many=3)
assert df["ranks"][0]==[4,5,6]