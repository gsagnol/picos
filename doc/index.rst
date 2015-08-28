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

 * 29 Aug. 15: **Picos** :ref:`1.1.1 <download>` **Released**
    Minor release with following changes:
     * Partial trace of an Affine Expression, cf. :func:`partial_trace() <picos.tools.partial_trace>`
     * Bugfix for compatibility with python3 (thanks to `Sergio Callegari <http://www.unibo.it/faculty/sergio.callegari>`_)
     * Initial support for the SDPA solver (with the option ``solver='sdpa'``, picos works as a wrapper around the SDPA executable based on the :func:`write_to_file() <picos.Problem.write_to_file()>` function; thanks to `Petter Wittek <http://peterwittek.com/>`_ )
     * Better PEP8-compliance

 * 15 Apr. 15: **Picos** :ref:`1.1.0 <download>` **Released**
    * PICOS is now compatible with **python 3+** (and remains compatible with python 2.6+). Many thanks to `Sergio Callegari <http://www.unibo.it/faculty/sergio.callegari>`_ for this compatibility layer ! If you plan to work with PICOS and python3, think to install the most recent version of your solver (Mosek, Cplex, Gurobi, or Cvxopt). SCIP is not supported in python3+ at this point (but remains supported with python 2.x).
    
    * PICOS is now available on `github <http://github.com/gsagnol/picos>`_.
    

 * 30 Jan. 15: **Picos** :ref:`1.0.2 <download>` **Released**
    Major release with following new functionalities:
     * Support (read and write) for ``.cbf`` problem files (`conic benchmark format <http://cblib.zib.de/>`_ ), which should be the standard for (mixed integer) conic optimization problems, cf. :func:`write_to_file <picos.Problem.write_to_file>` and :func:`import_cbf <picos.tools.import_cbf>` . 
     * Improved support for complex SDP (more efficient implementation of :func:`to_real() <picos.Problem.to_real>` , corrected bug in the implementation of the scalar product for Hermitian matrices and the conjugate of a complex expression, support for equality constraints involving complex coefficients)
     * Support for inequalities involving the sum of k largest elements of an affine expression, or the k largest eigenvalues of a symmetric matrix expression, cf. the functions :func:`sum_k_largest() <picos.tools.sum_k_largest>` , :func:`sum_k_smallest() <picos.tools.sum_k_smallest>` , :func:`sum_k_largest_lambda() <picos.tools.sum_k_largest_lambda>`, :func:`sum_k_smallest_lambda() <picos.tools.sum_k_smallest_lambda>`, :func:`lambda_max() <picos.tools.lambda_max>` and :func:`lambda_min() <picos.tools.lambda_min>` .
     * Support for inequalities involving the :math:`L_{p,q}-` norm of an affine expresison, cf. :func:`norm() <picos.tools.norm>` .
     * New ``vtype`` for antisymmetric matrix variables ( :attr:`vtype <picos.Variable.vtype>` ``= antisym``).
     * Constraints can be specified as membership in a :class:`Set <picos.Set>` . Sets can be created by the functions :func:`ball() <picos.tools.ball>` , :func:`simplex() <picos.tools.simplex>`, and :func:`truncated_simplex() <picos.tools.truncated_simplex>` .
     * New functions :func:`maximize <picos.Problem.maximize>` and :func:`maximize <picos.Problem.minimize>` to specify the objective function of a problem and solve it. 

    And many thanks to `Petter Wittek <http://peterwittek.com/>`_ for the following improvements, who were motivated by the use of PICOS in the package `ncpol2sdpa <http://peterwittek.github.io/ncpol2sdpa/>`_ for optimization over noncommutative polynomials:
     * More efficient implementation of the writer to the sparse - SDPA file format (:func:`write_to_file <picos.Problem.write_to_file>`)
     * Hadamard (elementwise) product of affine expression is implemented, as an overload of the ``^`` operator,   cf. an example :ref:`here <overloads>` .
     * Partial transposition of an Affine Expression, cf. :func:`partial_transpose() <picos.tools.partial_transpose>` or the :attr:`Tx <picos.AffinExp.Tx>` attribute.
     
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

