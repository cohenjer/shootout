# A first test for my shoot-out decorators
import inspect
import itertools
import copy
import os
import pandas
from datetime import datetime

def run_and_track(add_track=None, algorithm_names=None, path_store=None, name_store=None,
                    verbose=True, nb_seeds=1, **kwa):
    '''
    AMAZING DOCUMENTATION
    Note: outputs must have nbr of algorithm lengths, but imputs can be objectified
    '''
    # Preprocessing: converting any single value in kwa to a singleton list
    for i in kwa:
        if type(kwa[i])!= list:
            print("Converting single parameter swipe to singleton")
            kwa[i] = [kwa[i]]

    def inner_run_and_track(fun):
        # Before all, initialize our storage DataFrame
        df = pandas.DataFrame()

        # we start by reading kwa contents and match it with inputs
        inputs = inspect.signature(fun).parameters # Get the default inputs of fun
        list_param_values = []
        for name in kwa.keys():
            if name in inputs.keys():
                # we will need to loop over this parameter as requested by the user
                # we use itertools.product to build one big Cartesian product loop, and we add this param to the product
                list_param_values.append(kwa[name])
            else:
                # TODO proper warning
                print("Input argument {} in run_and_track call not in target script, it is ignored".format(name))

        # we get the unlooped parameters by comparing signature with kwa
        default_params = []
        for name in inputs.keys():
            if name not in kwa.keys():
                # store these for adding in data frame later
                default_params.append(name)

        # Now we enter the big loop
        for params in itertools.product(*list_param_values):
            # Discrete printing for following the run
            if verbose:
                print("Params values are currently: {}".format(params))
            # params are not a dictionary so we make function read them by building a dictionary
            dic = kwa
            for i,elem in enumerate(kwa.keys()):
                dic[elem] = params[i]
            # Calling our script with the correct hyperparameters, everything else is defaults
            # User might want several runs for each parameter set (e.g. random initialization inside fun), we do it for him
            for s in range(nb_seeds):
                outputs = fun(**dic)
                # we assume outputs is a dictionary

                # Storing all results and parameters in a dataframe
                # This is inputs, outputs (both fixed and variable)
                store_dic = {}
                store_dic.update(dic)
                # we have variable inputs now we add default inputs inferred earlier
                for name in default_params:
                    store_dic.update({name:inputs[name].default})
                # Optional manual tracking
                if add_track:
                    store_dic.update(add_track)
                # A counterintuitive step is to unbracket all the singleton list parameter values in the dictionary, but split other lists. Indeed, plotly will have to consider these values as objects to store several lines simultaneously, and this does the job efficiently.
                dic_copy = copy.copy(store_dic)
                # copy to avoid changing the dictionary online, which fails
                for elem in dic_copy:
                    if type(store_dic[elem]) == list:
                        if len(store_dic[elem])==1:
                            store_dic[elem] = store_dic[elem][0]
                        else:
                            # we go over all elements of the list and split them in the dictionary
                            for i in range(len(store_dic[elem])):
                                store_dic[elem+"_"+str(i)] = store_dic[elem][i]
                            # Then remove the original list
                            # Otherwise when creating the DataFrame, we have lists of various sizes in the dic and this is not accepted as input
                            del store_dic[elem]
                del dic_copy
                # A static storage: algorithms name provided by user TODO: better solution? annoying to write names :/
                if algorithm_names:
                    store_dic.update({"algorithm": algorithm_names})
                # Storing seed index
                store_dic.update({"seed_idx": s})

                # Now we deal with outputs. 
                # They are trickier because we would like to point out errors(per iteration) and time(per iteration)
                # we must also record an output per algorithm, ie. get the name of the algorithms
                # Thus we make some assumptions on the outputs
                # 1. outputs is one dictionary
                # 2. the error and timing lists must be inside list themselves
                store_dic.update(outputs)
                # Again remove singletons if user used some in outputs; TODO: support list in outputs which are not of fixed size?
                for elem in store_dic:
                    if type(store_dic[elem]) == list and len(store_dic[elem])==1:
                        store_dic[elem] = store_dic[elem][0]

                # We can finally update the pandas DataFrame with all the run information
                df= pandas.concat([df,pandas.DataFrame(store_dic)], ignore_index=True)

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
    return inner_run_and_track # This is the nature of wrapper(nanana)(.) map. Parameters in kwargs and args can parameterize the inner call
