# todolist:
# - transform df to add error at timestamps/itstamps
# - build new df for convergence plots
# - build scores for comparison plots

# I use tensorly to be backend agnostic
import tensorly as tl
import pandas as pd
# for internal stuff
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

def find_best_at_all_thresh(df, thresh, batch_size, err_name="errors", time_name="timings"):
    """
    This utility function find the method was the fastest to reach a given threshold, at each threshold in the list thres.

    Parameters:
    ----------
    df : Pandas DataFrame
         The dataframe containing the errors and timings for each algorithm at each iterations, for several runs.
         For details on the expected names, check synthetic_comparisons_Frobenius.py
         Because I am a lazy coder:
            - Batch size must be constant
            - The algorithms must always be stored in df in the same order

    thresh: list
            A list of thresholds to be used for computing which method was faster.

    batch_size: int
            Number of algorithm runs to compare for the max pooling. Should be a multiple (typically 1x) of the number of algorithms.

    Returns:
    -------
    scores_time: nd array
            A table "method x thresh" with how many times each method was the fastest to reach a given threshold. Here faster is understood in runtime.

    scores_it: nd array
            A table "method x thresh" with how many times each method was the fastest to reach a given threshold. Here faster is understood in number of iterations.
    """

    timings = []
    iterations = []
    # Strategy: we sweep each error and find at which time each threshold was attained
    for row_errs,errs in enumerate(df[err_name]):
        pos = 0
        time_reached = []
        it_reached = []
        for pos_err,err in enumerate(errs):
            while pos<len(thresh) and err<thresh[pos]:
                # just in case several thresholds are beaten in one iteration
                time_reached.append(df[time_name][row_errs][pos_err])
                it_reached.append(pos_err)
                pos+=1
        if len(time_reached)<len(thresh):
            time_reached = time_reached +( (len(thresh)-len(time_reached))*[np.Inf] )
            it_reached = it_reached +( (len(thresh)-len(it_reached))*[np.Inf] )
        timings.append(time_reached)
        iterations.append(it_reached)
    # casting as a tl tensor (matrix) for slicing vertically
    timings = tl.tensor(timings)
    iterations = tl.tensor(iterations)

    # Then we find who is the winner for each batch and for each threshold
    Nb_batch = int(len(timings)/batch_size)  # should be integer without recast
    # reshaping timings into a 3-way tensor for broadcasting numpy argmax
    timings = tl.reshape(timings, [Nb_batch,batch_size,len(thresh)])
    iterations = tl.reshape(iterations, [Nb_batch,batch_size,len(thresh)])
    # we can now find which count how many times each algorithm was faster by finding the index of the fastest method for each batch
    winners_time = my_argmin(timings)
    winners_it = my_argmin(iterations)
    # Assuming results are stored always in the same order, a batch row index corresponds to an algorithm name
    scores_time = tl.zeros((batch_size,len(thresh)))
    scores_it = tl.zeros((batch_size,len(thresh)))
    for k in range(batch_size):
        for i in range(Nb_batch): 
            for j in range(len(thresh)):
                if type(winners_time[i,j])==list:
                    if k in winners_time[i,j]:
                        scores_time[k,j]+=1
                else:
                    if winners_time[i,j]==k:
                        scores_time[k,j]+=1
                    
                if type(winners_it[i,j])==list:
                    if k in winners_it[i,j]:
                        scores_it[k,j]+=1
                else:
                    if winners_it[i,j]==k:
                        scores_it[k,j]+=1

    return scores_time, scores_it, timings, iterations

def my_argmin(a):
    """
    argmin but returns list of equal indices. Axis must be 1, a is a third order tensor.
    """
    tutu = a.min(axis=1)[:,None]
    tutu[tutu==np.Inf]=0 #removing tl.inf
    minpos = (a == tutu)
    # TODO: remove np.Inf counting
    tlargmin = tl.argmin(a,axis=1)
    myargmin= tl.zeros(tlargmin.shape, dtype=object)-1
    for i in range(minpos.shape[0]):
        for j in range(minpos.shape[1]):
            for k in range(minpos.shape[2]):
                if minpos[i,j,k]:
                    if type(myargmin[i,k])==list:
                        myargmin[i,k] = myargmin[i,k] + [j]
                    elif myargmin[i,k]==-1:
                        myargmin[i,k] = j
                    else:
                        myargmin[i,k] = [myargmin[i,k]] + [j]

    return myargmin


def df_to_convergence_df(df, err_name="errors", time_name="timings", algorithm_name="algorithm", other_names=None, max_time=np.Inf, groups=False, groups_names=None, filters=None):
    # Plotting a few curves for all methods

    # We start by creating a new dataframe in long format (one error per row instead of storing list)
    df2 = pd.DataFrame()
    # exploring the errors one by one
    for idx_pd,i in enumerate(df[err_name]):
        # filters including seeds
        if filters:
            flag=1
            for cond in filters:
                # filters entries can be lists or floats/ints. two different behaviors.
                if type(filters[cond])==list:
                    if df.iloc[idx_pd][cond] not in filters[cond]:
                        flag = 0
                else:
                    if df.iloc[idx_pd][cond]!=filters[cond]:
                        flag = 0
        else:
            flag=1
        if flag:
            its = tl.arange(0,len(i),1)
            dic = {
                "it":its,
                time_name: df.iloc[idx_pd][time_name],
                err_name:i,
                algorithm_name:df.iloc[idx_pd][algorithm_name],
                "seed_idx":df.iloc[idx_pd]["seed_idx"]}
                # Other custom names to store
            for name in other_names:
                dic.update({name: df.iloc[idx_pd][name]})
            df2=pd.concat([df2, pd.DataFrame(dic)], ignore_index=True)

    # cutting time for more regular plots
    if max_time:
        df2 = df2[df2[time_name]<max_time]

    # small preprocessing for grouping plots (avoids lines from end to start in plotly; also more precise, each group represents lines of the same color albeit possibly different runs.)
    if groups:
        zip_arg = []
        if not groups_names:
            # TODO warning
            print("You asked to group convergence plots together, but no parameter has been provided for grouping.")
        groups_names += ["seed_idx"]
        for name in groups_names:
            zip_arg.append(df2[name])
        df2["groups"] = list(zip(*zip_arg))

    return df2

def error_at_time_or_it(df, time_stamps=None, it_stamps=None, err_name="errors", time_name="timings"):
    # add to df error for each run at given time stamps in dedicated columns, same for iterations
    # Since error and time are discrete, this returns a nearest neighbor estimation of the error in the time-space wrt to timings asked by the user.
    # In particular, for large enough time, error is the final recorded error.
    for time in time_stamps:
        store_list = []
        for err,timings in list(zip(df[err_name],df[time_name])):
            # find the error for time closest to the desired time
            # better way since sorted?
            idx_time = tl.argmin(tl.abs(np.add(timings,-time)))
            err_at_time = err[idx_time]
            store_list.append(err_at_time)
        df["err_at_time_"+str(time)] = store_list
    for it in it_stamps:
        store_list = []
        for err in df[err_name]:
            # find the error for time closest to the desired time
            # better way since sorted?
            err_at_it = err[it]
            store_list.append(err_at_it)
        df["err_at_it_"+str(it)] = store_list
    return df
            
def regroup_columns(df,keys=None, how_many=None):
    """
    Because we split input lists in DataFrames for storage, it may be convenient to re-introduce the original lists as columns on user demand.
    Keys is the list of strings of names of the form foo_n where foo is in keys and n is an integer.
    How many tells the upper bound on n.
    """
    for name in keys:
        df[name] = pd.Series([[] for i in range(len(df))])
        for j in range(len(df)):
            df[name][j] = [df[name+"_"+str(i)][j] for i in range(how_many)]
    return df
