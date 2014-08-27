.. picos documentation master file, created by
   sphinx-quickstart on Thu Mar  1 10:03:01 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br />

PICOS: A Python Interface for Conic Optimization Solvers
========================================================

.. image:: _static/picos_big_trans.gif

Welcome to the documentation of PICOS,
a user-friendly python interface to many linear and conic optimization solvers,
see more about PICOS in the :ref:`introduction <intro>`.

The latest version can be downloaded :ref:`here <download>`,
and can be installed by following :ref:`these instructions <requirements>`.

This documentation contains a :ref:`tutorial <tuto>` and some :ref:`examples <examples>`,
which should already be enough for a quick start with PICOS. To go deeper,
have a look at the :ref:`picos reference <api>`, which provides information
on every function of PICOS.


**News**

 * 27 Aug. 14: **Picos** :ref:`1.0.1 <download>` **Released** |br|
    Major Release with following changes:
     * Support for Semidefinite Programming over the complex domain, see :ref:`here <complex>`.
     * Flow constraints in graphs, cf. :ref:`this section <flowcons>`.
     * Improved implementation of several functionalities, in particular the slicing of affine expressions (``__getitem__``), the processing of large matrix parameters, and the access to primal optimal variables with CPLEX.
     * Improved readibility of the documentation.
     
 * 18 May 14: **Picos** :ref:`1.0.1.dev <download>` **Released** |br|
     Preliminary release of the 1.0.1 (still a few bugs for complex SDPs).
     
        

 * 19 Jul. 13: **Picos** :ref:`1.0.0 <download>` **Released** |br|
        with Semidefinite Programming Interface for MOSEK 7.0 !!!
     
 * Former changes are listed :ref:`here <changes>`.

.. _contents:

**PICOS Documentation contents**

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

