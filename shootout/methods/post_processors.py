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
    
    TODO: remove batch parameter/provide clever default
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
            if time_name:
                dic = {
                    "it":its,
                    time_name: df.iloc[idx_pd][time_name],
                    err_name:i,
                    algorithm_name:df.iloc[idx_pd][algorithm_name],
                    "seed":df.iloc[idx_pd]["seed"]}
            else:
                dic = {
                    "it":its,
                    err_name:i,
                    algorithm_name:df.iloc[idx_pd][algorithm_name],
                    "seed":df.iloc[idx_pd]["seed"]}
            # Other custom names to store
            for name in other_names:
                ## filter if object or not TODO
                #if type(df.iloc[idx_pd][name])==list:
                #    dic.update({name: [df.iloc[idx_pd][name]]}) # objectified
                #else:
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
            groups_names = []
        groups_names += ["seed"]
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
            
def regroup_columns(df,keys=None, how_many=None, textify=True):
    """
    Because we split input lists in DataFrames for storage, it may be convenient to re-introduce the original lists as columns on user demand.
    Keys is the list of strings of names of the form foo_n where foo is in keys and n is an integer.
    How many tells the upper bound on n.
    Textify makes strings inputs instead of list of integers because this makes life 8 times easier with Pandas and Plotly.
    TODO support single input for keys instead of list
    """
    for name in keys:
        df[name] = pd.Series([[] for i in range(len(df))])
        for j in range(len(df)):
            if textify:
                df[name][j] = str([df[name+"_"+str(i)][j] for i in range(how_many)])
            else: 
                df[name][j] = [df[name+"_"+str(i)][j] for i in range(how_many)]
        # then we remove the columns that were merged
        df = df.drop([name+"_"+str(i) for i in range(how_many)], axis=1)    
    return df

def interpolate_time_and_error(df, err_name="errors", time_name="timings", k=0, logtime=False, npoints=500, adaptive_grid=False, alg_name="algorithm"):
    """
    some doc

    Parameters
    ----------
    df : pandas dataframe
        raw shootout results dataframe with errors and time in lists.
    k  : int, default 0
        the max time for interpolation will be the (longest - k) runtime for the algorithms. Set to k>0 if k runs are abnormally long.
    logtime : boolean, default False
        choose if interpolation grid is linear (False) or logarithmic (True). Set to True when timings are very different between several runs.
    npoints : int, default 500
        number of iterpolation points.
    adaptive_grid : bool, default False
        determines if each test condition has its own time grid. If True, a dirty BAD hack is used: shootout runs samples in most intern loop, therefore we can compute grids for each block of rows in df cut according to periodicity of seed.
    alg_name : string, default "algorithm"
        the string value of the key in df that contains the algorithm name.
    """
    # Check if grid is individual or global
    if adaptive_grid:
        # empty columns for init
        df["errors_interp"] = df[err_name]
        df["timings_interp"] = df[time_name]
        # We work on blocks of df cut according to seed periodicity
        if alg_name in df.keys():
            nb_algs = np.maximum(len(np.unique(df["algorithm"])),1) #0 not allowed, in case algorithms are not provided
        else:
            nb_algs = 1
        seed_periodicity = (df["seed"].max() + 1)*nb_algs
        nb_rows = len(df)
        nb_blocks = int(nb_rows/seed_periodicity)
        for block_idx in range(nb_blocks):
            for alg_idx in range(nb_algs):
                maxtime_list = []
                for row in df[time_name][seed_periodicity*block_idx+alg_idx:seed_periodicity*(block_idx+1):nb_algs]: # todo check indices
                    maxtime_list.append(row[-1])
                maxtime = np.sort(maxtime_list)[-1]
                # then we create a grid
                if logtime:
                    time_grid = np.logspace(0, np.log10(maxtime), npoints)
                else:
                    time_grid = np.linspace(0, maxtime, npoints)
                # now we populate the time column for the block and interpolate
                for idx in range(seed_periodicity*block_idx+alg_idx, seed_periodicity*(block_idx+1), nb_algs):
                    df["timings_interp"][idx] = time_grid
                    new_errors = np.interp(time_grid, df[time_name][idx], df[err_name][idx])
                    df["errors_interp"][idx]=new_errors

    else:
        # First we look for the k-th max timing over all runs
        maxtime_list = []
        for timings in df[time_name]:
            maxtime_list.append(timings[-1])
        kmaxtime = np.sort(maxtime_list)[-1-k]
        # then we create a grid based on that time
        if logtime:
            time_grid = np.logspace(0, np.log10(kmaxtime), npoints)
        else:
            time_grid = np.linspace(0, kmaxtime, npoints)
        # creating dummy columns
        df["errors_interp"] = df[err_name]
        df["timings_interp"] = df[time_name]
        # now we interpolate each error on that grid
        for idx_errors, errors in enumerate(df[err_name]):
            new_errors = np.interp(time_grid, df[time_name][idx_errors], errors)
            # we can update the dataframe on the fly
            df["errors_interp"][idx_errors]=new_errors
            df["timings_interp"][idx_errors]=time_grid

    return df


def median_convergence_plot(df_conv, type="iterations", err_name="errors", time_name="timings", mean=False):
    """some doc

    Parameters
    ----------
    df : pandas dataframe
        input dataframe with error at each iteration split in rows, see xxx
        requires timings to be aligned by linear interpolation first.
    type : string, default "timings"
        choose if the median is over time or over iterations

    TODO: use another syntax more similar to above?
    """
    # we use the groupby function; we will groupy by everything except:
    # - errors (we want to median them)
    # - seeds (they don't matter since we use conditional mean over everything else)
    # - timings
    df = df_conv.copy()
    df.pop("seed")
    #if type == "timings":
        ## we store time to put it back at the end
        #timings_saved = df[time_name] 
    if type=="iterations":
        df.pop(time_name) # we always pop time, since we made sure it is aligned with iterations
    elif type=="timings":
        df.pop("it")
    # else do nothing

    # iterations behave like an index for computing the median
    df.pop("groups") # good idea?

    namelist = list(df.keys())
    namelist.remove(err_name)

    if mean:
        df_med = df.groupby(namelist, as_index=False).mean() 
    else:
        df_med = df.groupby(namelist, as_index=False).median() 
    df_02 = df.groupby(namelist, as_index=False).quantile(q=0.2) 
    df_08 = df.groupby(namelist, as_index=False).quantile(q=0.8) 
    df_med["q_errors_p"] = df_08[err_name]-df_med[err_name]
    df_med["q_errors_m"] = df_med[err_name]-df_02[err_name]

    #if type=="timings":
        #df_med[time_name] = timings_saved

    return df_med#, df_time.groupby(namelist_time).median()