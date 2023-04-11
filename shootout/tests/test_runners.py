from shootout.methods.runners import run_and_track
import pandas as pd
import os
import numpy as np

# Testing run_and_track
# check that a DataFrame is created and that this function runs smoothly
def test_run_and_track_multiple_algs():  
    vars = dict({
        "n": [10,15],
        "m": [25,30],
        "listpar":[[1,2,3],[4,5,6]],
        "noise": 0.25,
        "r": 3,
        "seed": [1,2]
    })
    @run_and_track(add_track={"mytrack":[7,12,20]}, name_store="test-df", algorithm_names=["method1","method2"], **vars)
    def myalgorithm(**v):
        w = v["m"]+v["n"]+v["r"]+v["noise"]+v["seed"]
        h = [1,2,3,4]
        err = [v["n"]-v["m"], v["n"]+v["m"]+v["seed"]]
        time = [0,0.2]
        return {"errors":[err,err],"timings": [time,time],"myoutput":[h,h],"scalaroutput":7}
    # We check if the dataframe has been created by loading it
    df = pd.read_pickle("test-df")
    # picking a random element to check
    assert df["errors"][6][1] == 37
    # deleting temporary df
    os.remove("test-df")
    #return df


# Another test for single algorithm
def test_run_and_track_single_algorithm():
    vars = dict(
        {
            "n": [10,15],
            "m": [25,30],
            "noise": 0.25,
            "r": 3,
            "random": True
        }
    )
    @run_and_track(add_track={"mytrack":[7,12,20]}, name_store="test-df", algorithm_names=["method1"], **vars)
    def myalgorithm(**v):
        w = v['m']+v['n']+v['r']+v['noise']
        h = np.array([[1,2,3,4],[5,6,7,8]])
        err = [v['n']-v['m'], v['n']+v['m'], v['n']**2]
        time = [0,0.2 ]
        return {"errors":err,"timings": time, "myoutput":h, "scalaroutput": 7}
    # We check if the dataframe has been created by loading it
    df = pd.read_pickle("test-df")
    # picking a random element to check
    print(df)
    assert df["errors"][0][0] == -15
    # deleting temporary df
    os.remove("test-df")