import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# todolist:
# - transform df to add error at timestamps/itstamps
# - build new df for convergence plots
# - build scores for comparison plots

def find_best_at_all_thresh(df, thresh, batch_size, err_name="errors", time_name="timings"):
    """
    This utility function finds the method was the fastest to reach a given threshold, at each threshold in the list thres.

    Parameters
    -----------
    df : Pandas DataFrame
        The dataframe containing the errors and timings for each algorithm at each iterations, for several runs.
        Because I am a lazy coder:

            - Batch size must be constant
            - The algorithms must always be stored in df in the same order

    thresh: list
        A list of thresholds to be used for computing which method was faster.

    batch_size: int
        Number of algorithm runs to compare for the max pooling. Should be a multiple (typically 1x) of the number of algorithms.

    Returns
    --------
    scores_time: nd array
        A table (method x thresh) with how many times each method was the fastest to reach a given threshold. Here faster is understood in runtime.

    scores_it: nd array
        A table (method x thresh) with how many times each method was the fastest to reach a given threshold. Here faster is understood in number of iterations.
    """
    #TODO: remove batch parameter/provide clever default

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
    # casting as a numpy array (matrix) for slicing vertically
    timings = np.array(timings)
    iterations = np.array(iterations)

    # Then we find who is the winner for each batch and for each threshold
    Nb_batch = int(len(timings)/batch_size)  # should be integer without recast
    # reshaping timings into a 3-way tensor for broadcasting numpy argmax
    timings = np.reshape(timings, [Nb_batch,batch_size,len(thresh)])
    iterations = np.reshape(iterations, [Nb_batch,batch_size,len(thresh)])
    # we can now find which count how many times each algorithm was faster by finding the index of the fastest method for each batch
    winners_time = my_argmin(timings)
    winners_it = my_argmin(iterations)
    # Assuming results are stored always in the same order, a batch row index corresponds to an algorithm name
    scores_time = np.zeros((batch_size,len(thresh)))
    scores_it = np.zeros((batch_size,len(thresh)))
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
    tutu[tutu==np.Inf]=0 #improve? 
    minpos = (a == tutu)
    npargmin = np.argmin(a,axis=1)
    myargmin= np.zeros(npargmin.shape, dtype=object)-1
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


def df_to_convergence_df(df_in, err_name="errors", time_name="timings", algorithm_name="algorithm", other_names=None, max_time=np.Inf, groups=True, groups_names=None, filters=None):
    """
    Convert a compact Pandas Dataframe with a column ``errors`` containing lists into a long format where these lists are unfolded. This is useful as a post-processor of @run_and_track to be able to easily plot convergence plots with plotly. For instance using @run_and_track to compute and store results,

    >>> @run_and_track(n=[1,2],algorithm_names=["goodalg"]):
    >>> myfun(n=1):
    >>>     costs = [11]
    >>>     time = [0]
    >>>     for i in range(10):
    >>>         # here a dummy cost function over 10 iterations
    >>>         costs.append(10-i)
    >>>         time.append(i)
    >>>     return {"errors": costs, "timings": time}
    >>> # the above code stores the results in a dataframe df
    >>> df = np.load(...)
    >>> # Now we want to plot errors vs time: we use df_to_convergence_df to convert df into a long format for this
    >>> vars = ["n"]
    >>> df_conv = df_to_convergence_df(df, other_names=vars, groups_names=vars)
    >>> # Convergence plots can be easily done with plotly at this stage
    >>> import plotly.express as px
    >>> px.line(df,y="errors",x="timings", facet_col="n") 

    This function will strip all the columns of df which are not error, timings, seed or algorithm_name, and will add an iteration counter. If other columns should be kept, specify their name using the other_names option.

    Parameters
    ----------
    df_in : pandas dataframe
        A dataframe with lists in columns to be unfolded, typically generated by @run_and_track
    err_name : str, optional
        specify a different name of the column in df containing the lists to unfold. Can be useful if you custumized the error names, or if several metrics have been stored in df. By default ``errors``
    time_name : str, optional
        specify a different name of the column in df containing the time lists to unfold. Can be useful if you custumized the timings names. By default ``timings``
    algorithm_name : str, optional
        name of the column containing the algorithm names, by default ``algorithm``
    other_names : list of str, optional
        a list containing the names of columns to keep in the unfolded dataframe, by default None
    max_time : float, optional
        specify a maximum time value after which the dataframe will be truncated, by default np.Inf
    groups : bool, optional
        When plotting line plots with plotly, if other_names have been provided, the lines may loop from end to beginning of the plot. This can be solved by providing a group entry in plotly plots. Groups contains seed by default if seed where provided in df. By default True
    groups_names : list of str, optional
        the name of the columns of df to group, by default None
    filters : dict, optional
        a dictionary with pairs of {column of df: value in that column}, to compute the convergence plot only for the rows in df where the values match the filters values. By default None

    Returns
    -------
    df : pandas dataframe
        A new dataframe in a long format with each row containing a single time, iteration, error value, seed and algorithm_name. Easy to use for plotting convergence plots with plotly.
    """
    # First, we filter undesired rows
    df = df_in.copy() # potentially bad to copy. TODO Better solution
    # sort by algorithm name to avoid issues with lines ordering in plotly
    df = df.sort_values("algorithm")
    df2 = pd.DataFrame()
    # Listify filters
    if filters:
        for key in filters:
            if not type(filters[key])==list:
                filters[key]=[filters[key]]
    # exploring the errors one by one
    for idx_pd,i in enumerate(df[err_name]):
        # filtering
        if filters:
            flag = 0
            for key in filters:
                if df.iloc[idx_pd][key] not in filters[key]:
                    flag=1
                    break
            if flag:
                continue
        
        its = np.arange(0,len(i),1)
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
    
    del df
    return df2

def nearest_neighbors_err_at_time_or_it(df, time_stamps=None, it_stamps=None, err_name="errors", time_name="timings"):
    """Adds to dataframe df columns ``err_at_time_xx`` or ``err_at_it_xx`` containing an estimation of errors at given time points or iteration values ``xx``. The estimation is performed for the ``err_at_time_xx`` column using the nearest neighbor error value.

    For a more precise interpolation, check out the ``interpolate_time_and_error`` function.

    Parameters
    ----------
    df : pandas dataframe
        A dataframe containing an ``err_name`` column and a ``time_name`` column 
    time_stamps : list, optional
        a list of values at which the column ``err_name`` should be evaluated, by default None
    it_stamps : list, optional
        a list of indices at which the column ``err_name`` should be evaluated, by default None
    err_name : str, optional
        the name of the error column in df, by default ``errors``
    time_name : str, optional
        the name of the timings columns in df, by default ``timings``

    Returns
    -------
    df: pandas dataframe
        the input dataframe with appended columns ``err_at_time_xx`` or ``err_at_it_xx``. 
    """
    for time in time_stamps:
        store_list = []
        for err,timings in list(zip(df[err_name],df[time_name])):
            # find the error for time closest to the desired time
            # better way since sorted?
            idx_time = np.argmin(np.abs(np.add(timings,-time)))
            err_at_time = err[idx_time]
            store_list.append(err_at_time)
        df["err_at_time_"+str(time)] = store_list
    for it in it_stamps:
        store_list = []
        for err in df[err_name]:
            err_at_it = err[it]
            store_list.append(err_at_it)
        df["err_at_it_"+str(it)] = store_list
    return df
            
def regroup_columns(df,keys=None, how_many=None, textify=True):
    """
    Because we split input lists in DataFrames for storage, it may be convenient to re-introduce the original lists as columns on user demand.

    For instance, to grid over a parameter r=[r1,r2,r3] which is naturally represented as a list, in the current version of the shootout toolbox, we are forced to store the results in three columns labeled "r_1", "r_2", "r_3".

    Parameters
    ----------
    df: pandas Dataframe
        a dataframe containing columns to regroup
    keys: list of strings, or string 
        a list of column names of the form foo_n where foo is in keys and n is an integer.
    how_many: int
        the number of elements that were splitted as dataframe columns
    textify: bool, optional
        makes strings inputs instead of list of integers because this makes life 8 times easier with Pandas and Plotly, default True
    """
    if type(keys)!=list:
        keys = [keys]
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
    Interpolates several error curves so that they all lie on the same grid. There is one grid per algorithm name by default. To use with caution as this can significantly bend the error curves.

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


def median_convergence_plot(df_conv, type_x="iterations", err_name="errors", time_name="timings", mean=False):
    """
    Computes the mean or median of several converge plots with respect to either time or iterations (this is determined by ``type_x``). By mean/median of several curve, we mean the mean/median at each ``type_x`` value.

    Parameters
    ----------
    df : pandas dataframe
        input dataframe with error at each iteration split in rows
        requires timings to be aligned by linear interpolation first.
    type_x : string, default ``timings``
        choose if the median is over time or over iterations.
    means : boolean, default False
        if True, compute the mean of the curves, otherwise the median is computed.
    """
    # TODO: use another syntax more similar to above?
    # we use the groupby function; we will groupy by everything except:
    # - errors (we want to median them)
    # - seeds (they don't matter since we use conditional mean over everything else)
    # - timings
    df = df_conv.copy()
    df.pop("seed")
    #if type == "timings":
        ## we store time to put it back at the end
        #timings_saved = df[time_name] 
    if type_x=="iterations":
        df.pop(time_name) # we always pop time, since we made sure it is aligned with iterations
    elif type_x=="timings":
        df.pop("it")
    # else pop custom
    else:
        df.pop(type_x)

    # iterations behave like an index for computing the median
    try:
        df.pop("groups") # good idea?
    except:
        pass

    namelist = list(df.keys())
    namelist.remove(err_name)

    #print(namelist)

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