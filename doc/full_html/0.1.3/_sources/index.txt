.. picos documentation master file, created by
   sphinx-quickstart on Thu Mar  1 10:03:01 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

PICOS: A Python Interface for Conic Optimization Solvers
========================================================

.. _contents:

**News**

 * 17 Apr. 13: **Picos** :ref:`0.1.3 <download>` **Released** |br|
   Major changes:
     * Function :func:`picos.geomean() <picos.tools.geomean>` implemented, to handle inequalities involving
       a geometric mean and reformulate them automatically as a set of SOCP constraints.
     * Some options were added for the function :func:`solve() <picos.Problem.solve>` ,
       to tell CPLEX to stop the computation as soon as a given value for the
       upper bound (or lower bound) is reached (see the options ``uboundlimit`` and ``lboundlimit``
       documented in :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`).
     * The time used by the solver is now stored in the dictionary
       returned by :func:`solve() <picos.Problem.solve>`.
     * The option ``boundMonitor`` of the function :func:`solve() <picos.Problem.solve>`
       gives access to the values of the lower and upper bounds over time with cplex.
       (this option is documented in :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`).
     * The weak inequalities operators ``<=`` and ``>=`` can now be used (but strict inequalities are
       still interpreted as weak inequalities !).
     * Minor bugs corrected (access to the duals of fixed variables with CPLEX,
       evaluation of constant affine expressions with a zero coefficient appearing
       in the dict of linear terms, number of constraints is now updated in
       :func:`remove_constraint() <picos.Problem.remove_constraint>`).

 * 10 Jan. 13: **Picos** :ref:`0.1.2 <download>` **Released** |br|
   bug-fix release, correcting:
     * The :func:`write_to_file() <picos.Problem.write_to_file>`
       function for sparse SDPA files. The function was writing the
       coefficients of the lower triangular part of the constraint matrices
       instead of the upper triangle.
     * An ``IndexError`` occuring with the function
       :func:`remove_constraint() <picos.Problem.remove_constraint>`.
   
   Thanks to Warren Schudy for pointing out these bugs of the previous release !

 * 08 Dec. 12: **Picos** :ref:`0.1.1 <download>` **Released** |br|
   Major changes:
     * Picos now interfaces GUROBI !
     * You can specify an initial solution to *warm-start* mixed integer optimizers.
       (see the option ``hotstart`` documented in
       :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`)
     * Minor bugs with quadratic expressions corrected
     * It's possible to return a reference to a constraint added
       with add_constraint()


**PICOS Documentation contents**

Welcome to the documentation of PICOS.
The latest version can be downloaded :ref:`here <download>`,
and can be installed by following :ref:`these instructions <requirements>`.
This documentation contains a :ref:`tutorial <tuto>` and some :ref:`examples <examples>`,
which should already be enough for a quick start with PICOS. To go deeper,
have a look at the :ref:`picos reference <api>`, which provides information
on every function of PICOS.

.. toctree::
   :maxdepth: 2

   intro
   tuto
   examples
   api
   download
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

