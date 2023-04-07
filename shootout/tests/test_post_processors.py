from shootout.methods.post_processors import nearest_neighbors_err_at_time_or_it, df_to_convergence_df, find_best_at_all_thresh, regroup_columns, interpolate_time_and_error, median_convergence_plot, my_argmin
import pandas as pd
import numpy as np
import os

path = os.path.dirname(__file__)
df = pd.read_pickle(path+"/run-example")

# TODO: myargmin
def test_my_argmin():
    a = np.array([[[1,1],[1,4]],[[5,6],[6,6]]])
    out = my_argmin(a)
    assert out[0,0] == [0,1]
    assert out[0,1] == 0
    assert out[1,0] == 0
    assert out[1,1] == [0,1]

# TODO: check that all args are tested

def test_find_best_at_all_thresh(df=df):
    thresh = np.logspace(5,2,50) 
    scores_time, scores_it, timings, iterations = find_best_at_all_thresh(df,thresh, 2)
    # stupid unit test
    assert scores_time[0,0] == 35

# Testing df_to_convergence_df
def test_df_to_convergence_df(df=df):
    other_names = ["delta"]  #useless, just to invoke the option
    df2 = df_to_convergence_df(df, other_names=other_names, max_time=1.5,
                                filters={"seed":[0,1], "loss":"l2"}, groups=True, groups_names=other_names)
    # another stupid test
    assert len(df2) == 410

# Testing err_at_time_or_it
def test_nearest_neighbors_err_at_time_or_it(df=df):
    df = nearest_neighbors_err_at_time_or_it(df, time_stamps=[0.1,0.5,1], it_stamps=[0,10,40])
    assert df["err_at_time_1"][0]<1.132584e-7 and df["err_at_time_1"][0]>1.132582e-7

# Testing regrouping
# manually adding columns
def test_regroup_columns(df=df):
    df["ranks_0"]=0
    df["ranks_1"]=0
    df = regroup_columns(df,keys=["ranks"], how_many=2, textify=True)
    # output is a string
    assert df["ranks"][0]=='[0, 0]'

def test_interpolate_time_and_error(df=df):
    df = interpolate_time_and_error(df, logtime=True, npoints=39, adaptive_grid=True)
    assert  32 > df["errors_interp"][69][0] > 31.8

def test_median_convergence_plot(df=df):
    df = interpolate_time_and_error(df, logtime=True, npoints=39, adaptive_grid=True)
    df_l2_conv = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                                filters={"loss":"l2"}, err_name="errors_interp", time_name="timings_interp")
    df_l2_conv = df_l2_conv.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
    df_l2_conv_it = df_to_convergence_df(df, groups=True, groups_names=[], other_names=[],
                                filters={"loss":"l2"})
    df_l2_conv_median_time = median_convergence_plot(df_l2_conv, type_x="timings")
    df_l2_conv_median_it = median_convergence_plot(df_l2_conv_it, type_x="iterations")
    assert 1.15e-8<df_l2_conv_median_it["q_errors_p"][0]<1.17e-8 and 1.4e-10<df_l2_conv_median_time["q_errors_p"][0]<1.41e-10