.. _api:

API
========

There are two submodules in shootout: ``runners`` and ``post_processors``. 

Runners
---------
``runners`` contains methods to run a given script over a user-defined grid of parameters, and store the results in a pandas dataframe.

.. toctree::
   :maxdepth: 2

   runners


Post_processors
----------------
``post_processors`` contains methods to treat the dataframe generated with ``runners`` methods to, for instance, make a convergence plot or average/median several error curves.


.. toctree::
   :maxdepth: 2

   post_processors