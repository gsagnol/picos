.. picos documentation master file, created by
   sphinx-quickstart on Thu Mar  1 10:03:01 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

PICOS: A Python Interface for Conic Optimization Solvers
========================================================

.. image:: _static/picos_big_trans.gif

Welcome to the documentations of PICOS,
a user-friendly python interface to many linear and conic optimization solvers,
see more about PICOS in the :ref:`introduction <intro>`.

The latest version can be downloaded :ref:`here <download>`,
and can be installed by following :ref:`these instructions <requirements>`.
Alternatively, you can clone the latest development version from **github**:
``$ git clone https://github.com/gsagnol/picos.git``. If you wish to collaborate on PICOS,
please make a pull request on `github <http://github.com/gsagnol/picos>`_.

This documentation contains a :ref:`tutorial <tuto>` and some :ref:`examples <examples>`,
which should already be enough for a quick start with PICOS.
There is also a :ref:`summary <summary>` of useful implemented functions.
To go deeper,
have a look at the :ref:`picos reference <api>`, which provides information
on every function of PICOS.


**News**

 * 4 Jul. 16: **Picos** :ref:`1.1.2 <download>` **Released**
    Major release with following changes:
      * Improved efficiency for the processing of large expressions.
      * It is now possible to dynamically add and remove constraints, e.g. for column generation approaches, cf. :ref:`this paragraph <delcons>` for an example.
        For an easier use, the function :func:`add_constraint() <picos.Problem.add_constraint()>` now returns a handle to the constraint when the option :func:`return_constraints=True <picos.Problem.set_all_options_to_default()>`
        has been passed to the problem. Then, constraints can be deleted by using :func:`constraint.delete() <picos.Constraint.delete()>`.
      * In previous versions, PICOS detected constraints that could be interpreted as a bound on a variable. This was creating a mess to delete constraints,
        so the default behaviour is now to pass all constraints as such. To stick to the old behaviour, use the option :func:`pass_simple_cons_as_bound=True <picos.Problem.set_all_options_to_default()>`.
      * New signature for the function :func:`partial_transpose() <picos.tools.partial_transpose()>`, which can now transpose arbitrary subsystems from a kronecker product.
      * Corrections of minor bugs with complex expressions.
      * Better support for the SDPA solver.

 * 29 Aug. 15: **Picos** :ref:`1.1.1 <download>` **Released**
    Minor release with following changes:
     * Partial trace of an Affine Expression, cf. :func:`partial_trace() <picos.tools.partial_trace>`
     * Bugfix for compatibility with python3 (thanks to `Sergio Callegari <http://www.unibo.it/faculty/sergio.callegari>`_)
     * Initial support for the SDPA solver (with the option ``solver='sdpa'``, picos works as a wrapper around the SDPA executable based on the :func:`write_to_file() <picos.Problem.write_to_file()>` function; thanks to `Petter Wittek <http://peterwittek.com/>`_ )
     * Better PEP8-compliance

 * 15 Apr. 15: **Picos** :ref:`1.1.0 <download>` **Released**
    * PICOS is now compatible with **python 3+** (and remains compatible with python 2.6+). Many thanks to `Sergio Callegari <http://www.unibo.it/faculty/sergio.callegari>`_ for this compatibility layer ! If you plan to work with PICOS and python3, think to install the most recent version of your solver (Mosek, Cplex, Gurobi, or Cvxopt). SCIP is not supported in python3+ at this point (but remains supported with python 2.x).
    
    * PICOS is now available on `github <http://github.com/gsagnol/picos>`_.
    
 * Former changes are listed :ref:`here <changes>`.

.. _contents:

**PICOS Documentation contents**

.. toctree::
   :maxdepth: 2

   intro
   tuto_summary
   examples
   api
   download
   changes
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

