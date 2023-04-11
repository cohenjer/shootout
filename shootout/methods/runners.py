# A first test for my shoot-out decorators
import itertools
import copy
import os
import pandas
from datetime import datetime
from shootout.methods.post_processors import regroup_columns

def run_and_track(add_track=None, algorithm_names=None, path_store=None, name_store=None,
                    verbose=True, skip=False, **kwa):
    """ 
    This function is the main ingredient of shootout. It is a decorator, to run a python function ``my_script`` with a set of inputs to grid on. The outputs are stored in a pandas DataFrame which is stored locally.
    
    Shootout is designed to make optimization algorithms comparison easier, so the target application is to lift up a code comparing algorithms on single parameter settings to a full comparison and storing all parameters and results.

    A few restrictions are placed on the wrapped function:

        - outputs is one dictionary
        - the errors, timings and any other algorithm dependent quantity must be inside lists of length the number of algorithms.

    In single_method mode, 2. is not applied. We allow the user to simply provide anything that will be added as an object to the df.

    A toy example:

    >>> @run_and_track(nb_seeds=5, n=[5,10], m=[4,8])
    >>> def my_script(n=7, m=9, p=12):
    >>>     return { "something": n+m+p }

    will return a dataframe containing four columns ``n`` , ``m`` , ``something`` and ``seed`` , with 5x2x2 rows (5 runs which are here exactly the same, two values for n and two values for m).    
    Note that the function should output a dictionary.

    **Syntax:**
    In the case of a single algorithm to run with several hyperparameters, the syntax looks like this:
    
    >>> @run_and_track(algorithm_names=['My favorite Algorithm'], hyperparameter=[0.1,1,10], seed=[1,15,975])
    >>> def my_script(n=10, m=10, hyperparameter=0, seed=0):
    >>>     # some random generation using seed, e.g. with numpy randomstate
    >>>     # seed will be populated by numbers in the seed list
    >>>     # and hyperparameter with 0.1, 1 and 10
    >>>     errors, timings, anything_else = my_algorithm(random_data,n,m,hyperparameter)
    >>>     return {"errors": errors, "timings": timings}
    
    where errors and timings must have the same length.
    
    For several algorithms, say two, the syntax for the returns of ``my_script`` function is similar but slightly different:
    
    >>> @run_and_track(algorithm_names=['My Favorite Algorithm', 'my annoying competitor'], seed=[1,17,597], hyperparameter=[0.1,1,10])
    >>> def my_script(n=10, m=10, hyperparameter=0, seed=0):
    >>>     # some random generation using seed, e.g. with numpy randomstate
    >>>     # seed will be populated by numbers from 0 to nb_seeds
    >>>     # and hyperparameter with 0.1, 1 and 10
    >>>     errors, timings, anything_else = my_algorithm(random_data,n,m,hyperparameter)
    >>>     errors2, timings2, anything_else2 = my_competitor(random_data,n,m,hyperparameter)
    >>>     return {"errors": [errors, errors2], "timings": [timings, timings2]}

    The naming of the inputs and outputs is completely free. However, naming errors/loss/utility collected in a list exactly ``errors`` and ``timings`` (time starting from 0) will allow for immediate processing with utilities from shootout such as plotting convergence curves.

    The recommended syntax when many parameters are involved is to define all the parameters in a dictionary, see ``**kwa`` below. For the signature of the inner function, one should also use a dictionary that makes use of the same keys.

    Parameters
    ----------
    **kwa: dictionary, optional
        A dictionary containing all the elements in the signature of the function to run. Typically use in the following fashion:

        >>> vars = dict(
        >>>     {
        >>>         "n": [10,15],
        >>>         "m": [25,30],
        >>>         "noise": 0.25,
        >>>         "r": 3,
        >>>         "seed": [1,2,3,4]
        >>>     }
        >>> )
        >>> @run_and_track(**vars)
        >>> def myalgorithm(**v):
        >>>     xxx = v["n"] + ...
        
        Another syntax is tolerated although not recommended, where the user may mismatch the signatures of "@run_and_track" and the ran function as such:

        >>> @run_and_track(n=[10,15], m=[25,30], noise=0.25, add_track={"mytrack":[7,12,20]}, name_store="test-df", nb_seeds=2, algorithm_names=["method1"])
        >>> def myalgorithm(n=10,m=20,r=3,random=True,noise=0.5):
        >>>     ...

        the entries in run_and_track must then be repeated in the function to be run. Note that both methods cannot be combined, so as to promote the use of the first method which is cleaner (all function parameters used for the run will be stored and the variables are not duplicated).
        By default None.
    add_track : dictionary, optional
        a dictionary of values to store manually. The inputs and outputs are stored anyway so this option should be avoided if possible. By default None
    algorithm_names : list, optional
        provide a list of name for each algorithm that will be run in the script. The names should match the outputs of the script if the outputs are lists (same size and order). Using this input is recommended for easily naming plots in post-processing stages. By default None
    path_store : string, optional
        the path (relative or absolute) to the directory where the DataFrame will be stored, by default None, and default directory is the current working directory.
    name_store : string, optional
        name of the file containing the DataFrame, by default None and the default name is the date in long format.
    verbose : bool, optional
        choose wether run_and_track prints its progress, by default True
    skip : bool, optional
        Set to True to skip computations, by default False.
    """
    #TODO: better error messages/error catching \
    #    - missmatches input names / loop variables
    #    - solo fun vs several
    #    - algorithm_names length does not match outputs
    
    def inner_run_and_track(fun):    
        print(f"verbose works?? {verbose}")
        # Preprocessing things
        if skip:
            return

        # Preprocessing: converting any single value in kwa to a singleton list
        # also tracking names of list parameters for rebuilding columns
        list_params_keys = []
        list_params_lengths = []
        for i in kwa:
            if type(kwa[i])!= list:
                #print("Converting single parameter swipe to singleton")
                kwa[i] = [kwa[i]]
            else:
                # get length of items in the list
                if type(kwa[i][0])==list:
                    list_params_keys.append(i)
                    list_params_lengths.append(len(kwa[i][0]))

        if algorithm_names:
            if len(algorithm_names)>1:
                single_method=False
            else:
                single_method=True
        else:
            # TODO Warning
            print("Consider adding algorithm names to benefit from automatic labeling of runs")

        # Before all, initialize our storage DataFrame
        df = pandas.DataFrame()

        # Now we enter the big loop
        for params in itertools.product(*[kwa[key] for key in kwa]):
            # Discrete printing for following the run
            if verbose:
                print(f"Params values are currently:")
                for i,key in enumerate(kwa):
                    print("   ", key+":", params[i])
            # params are not a dictionary so we make function read them by building a dictionary
            dic = kwa
            for i,elem in enumerate(kwa.keys()):
                dic[elem] = params[i]
            # Calling our script with the correct hyperparameters, everything else is defaults
            outputs = fun(**dic)

            # Storing all results and parameters in a dataframe
            # This is inputs, outputs (both fixed and variable)
            store_dic = {}
            store_dic.update(dic)
            # Optional manual tracking
            if add_track:
                store_dic.update(add_track)
            # A counterintuitive step is to unbracket all the singleton list parameter values in the dictionary, but split other lists. Indeed, pandas will have to consider these values as objects to store several lines simultaneously, and this does the job efficiently.
            dic_copy = copy.copy(store_dic)
            # copy to avoid changing the dictionary online, which fails
            for elem in dic_copy:
                if type(store_dic[elem]) == list:
                    if len(store_dic[elem])==1:
                        store_dic[elem] = store_dic[elem][0]
                    else:
                        # we go over all elements of the list and split them in the dictionary TODO improve
                        for i in range(len(store_dic[elem])):
                            store_dic[elem+"_"+str(i)] = store_dic[elem][i]
                        # Then remove the original list
                        # Otherwise when creating the DataFrame, we have lists of various sizes in the dic and this is not accepted as input
                        del store_dic[elem]
            del dic_copy
            if algorithm_names:
                store_dic.update({"algorithm": algorithm_names})

            if single_method:
                for elem in outputs:
                    outputs[elem] = [outputs[elem]] # double bracket makes pandas objectify this
            store_dic.update(outputs)

            # We can finally update the pandas DataFrame with all the run information
            df_temp = pandas.DataFrame(store_dic)
            df= pandas.concat([df,df_temp], ignore_index=True)

        # update dataframe to reconstruct the input lists as true lists
        for i in range(len(list_params_keys)):
            df = regroup_columns(df, keys=list_params_keys[i], how_many=list_params_lengths[i])

        # To store the dataframe, we use the provided path, otherwise we place at runpath
        if path_store:
            path_store_full = path_store
        else:
            path_store_full = os.getcwd()+"/"
        if name_store:
            path_store_full += name_store
        else:
            path_store_full += "run-{}".format(datetime.today().strftime('%Y-%m-%d_%H-%M'))
        df.to_pickle(path_store_full)
        return fun # This is what the wrapped function will actually be
    return inner_run_and_track