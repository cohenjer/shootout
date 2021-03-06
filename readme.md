# SHOOTOUT toolbox

## Why I wrote this package
Comparing numerical optimization algorithms is always a pain for me. Things I hate to do: 
- writing lots of loops for testing various hyperparameters.
- writing the code to store the results, painfully changing details all the time.
- comparing algorithms in a fair manner though lengthy plotting scripts.
- comming back to old codes for review updates a year after the simulations, and finding I did not store all the hyperpameters by mistake.
- looking up plotly syntax when updating plots, every single time.
- reading papers that compare algorithms in one run/one set of parameters.

Plus I am a very chaotic person, changing workflow every single paper. So I needed some tool to balance this entropy and make my life easier.

## What this does
- Using a decorator function @run_and_track(), one may run a script many times with user-defined hyperparameters grid; store all the results in clearly formatted pandas dataframe usable by plotly express.
- provide a few helpful functions for processing this dataframe, to produce interesting comparison plots (convergence plots, who is fastest at given threshold plots)
- TODO: a custom unifed API for tweaking Plotly plots and never look on the plotly documentation every single time.

## Installation
For now, no pip installation is provided. You can clone the repo and run
```python
pip install -e .
```
with root in the root folder of this package.

## TODOS
- proper documentation (will use sphinx)
- plotly plots modifiers
- change test dataframe to be much much smaller

## Feedback
I wrote this for myself, but if you have some ideas for improvements or new features, feel free to drop an issue or a pull request.