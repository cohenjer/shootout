.. shootout documentation master file, created by
   sphinx-quickstart on Fri Apr  7 10:51:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to shootout's documentation!
====================================

This package is still under development.

Click here to go the :ref:`API`.

**********
Context
**********

Why I wrote this package:
-------------------------
Comparing numerical optimization algorithms is always a pain for me. Things I hate to do: 

- writing lots of loops for testing various hyperparameters.
- writing the code to store the results, painfully changing details all the time.
- comparing algorithms in a fair manner though lengthy plotting scripts.
- comming back to old codes for review updates a year after the simulations, and finding I did not store all the hyperpameters by mistake.
- looking up plotly syntax when updating plots, every single time.
- reading papers that compare algorithms in one run/one set of parameters.

Plus I am a very chaotic person, changing workflow every single paper. So I needed some tools to balance this entropy and make my life easier.

What this does
---------------
- Using a decorator function @run_and_track(), one may run a script many times with user-defined hyperparameters grid; store all the results in clearly formatted pandas dataframe usable by plotly express.
- provide a few helpful functions for processing this dataframe, to produce interesting comparison plots (convergence plots, who is fastest at given threshold plots)

Installation
--------------
The package can be pip installed using::

   pip install shootout-opt

or by cloning the repo and run::

   pip install -e .

with root in the root folder of this package.

Feedback
--------
I wrote this for myself, but if you have some ideas for improvements or new features, feel free to drop an issue or a pull request.


.. toctree::
   :maxdepth: 1
   :hidden:
   
   API
   examples