from shootout.methods.runners import run_and_track
import pandas as pd
import os
import numpy as np

# Testing run_and_track
# Not much to test here; just check that a DataFrame is created and that this function runs smoothly
def test_run_and_track_multiple_algs():  
    @run_and_track(n=[10,15], m=[25,30],noise=0.25, add_track={"mytrack":[7,12,20]}, name_store="test-df", nb_seeds=2, algorithm_names=["method1","method2"], seeded_fun=True)
    def myalgorithm(n=10,m=20,r=3,random=True,noise=0.5, seed=0):
        w = m+n+r+noise+seed
        h = [1,2,3,4]
        err = [n-m, n+m+seed]
        time = [0,0.2]
        return {"errors":[err,err],"timings": [time,time],"myoutput":[h,h],"scalaroutput":7}
    # We check if the dataframe has been created by loading it
    df = pd.read_pickle("test-df")
    print(df)
    # picking a random element to check
    assert df["errors"][6][1] == 41
    # deleting temporary df
    os.remove("test-df")


# Another test for single algorithm
def test_run_and_track_single_algorithm():
    @run_and_track(n=[10,15], m=[25,30],noise=0.25, add_track={"mytrack":[7,12,20]}, name_store="test-df", nb_seeds=2, algorithm_names=["method1"])
    def myalgorithm(n=10,m=20,r=3,random=True,noise=0.5):
        w = m+n+r+noise
        h = np.array([[1,2,3,4],[5,6,7,8]])
        err = [n-m, n+m, n**2]
        time = [0,0.2 ]
        return {"errors":err,"timings": time, "myoutput":h, "scalaroutput": 7}
    # We check if the dataframe has been created by loading it
    df = pd.read_pickle("test-df")
    # picking a random element to check
    print(df)
    assert df["errors"][0][0] == -15
    # deleting temporary df
    os.remove("test-df")