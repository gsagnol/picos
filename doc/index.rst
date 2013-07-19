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

 * 19 Jul. 13: **Picos** :ref:`1.0.0 <download>` **Released** |br|
   Major Release with following changes:
     * Semidefinite Programming Interface for MOSEK 7.0 !!!
     * New options ``handleBarVars`` and ``handleConeVars`` to customize how SOCP and SDPs are passed to MOSEK
       (When these options are set to ``True`` , PICOS tries to minimize the number of variables of the
       MOSEK instance, see the doc in :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`).
     * The function :func:`dualize() <picos.Problem.dualize>` returns the Lagrangian dual of a Problem.
     * The option ``solve_via_dual`` (documented in
       :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>` ) allows the user to
       pass the dual of a problem to a solver, instead of the primal problem itself.
       This can yield important speed-up for certain problems.
     * In addition to the geometric mean function :func:`picos.geomean() <picos.tools.geomean>` , it is now possible
       to pass rational powers of affine expressions (through an overload of the ``**`` operator), trace of
       matrix powers with :func:`picos.tracepow() <picos.tools.tracepow>` , (generalized) p-norms
       with :func:`picos.norm() <picos.tools.norm>`, and nth root of a determinant with
       :func:`picos.detrootn() <picos.tools.detrootn>`. These functions automatically reformulate the entered  inequalities as a set of equivalent SOCP or SDP constraints.
     * It is now possible to specify variable bounds directly (rather than adding constraints of the type ``x >= 0`` ).
       This can be done with the Keywords ``lower`` and ``upper`` of the function
       :func:`add_variable() <picos.Problem.add_variable>` ,
       or by the methods :func:`set_lower() <picos.Variable.set_lower>` ,
       :func:`set_upper() <picos.Variable.set_upper>` ,
       :func:`set_sparse_lower() <picos.Variable.set_sparse_lower>` , and
       :func:`set_sparse_upper() <picos.Variable.set_sparse_upper>` of the class :class:`Variable <picos.Variable>`.
     * It is now more efficient to update a Problem and resolve it. This is done thanks to the attribute ``passed``
       of the classes :class:`Constraint <picos.Constraint>` and :class:`Variable <picos.Variable>` ,
       that stores which solvers are already aware of a constraint / variable. There is also an
       attribute ``obj_passed`` of the class :class:`Problem <picos.Problem>` , that lists the solver instances
       where the objective function has already been passed. The option ``onlyChangeObjective`` has been
       deprecated.
       
     

 * Former changes are listed :ref:`here <changes>`.


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
   changes
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

