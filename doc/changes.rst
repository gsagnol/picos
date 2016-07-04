.. |br| raw:: html

   <br />

.. _changes:

==============
Change History
==============

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

  * 30 Jan. 15: **Picos** :ref:`1.0.2 <download>` **Released** |br|
    
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

        


 * 27 Aug. 14: **Picos** :ref:`1.0.1 <download>` **Released** |br|
   
   Release fixing the missing functionalities of the previous *.dev* version:
     * Improved support for complex SDP (access to dual information and correction of a few bugs, in particular sum of complex affine expression now work correctly)
     * Flow constraints in graphs, including multicommodity flows, cf. :ref:`this section <flowcons>`.
     * Additional ``coef`` argument in the function :func:`picos.tracepow() <picos.tools.tracepow>`, in order to represent constraints of the form :math:`\operatorname{trace}(M X^p) \geq t`.
     * Improved implementation of :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`, which was taking a very long time to process large parameters.
     * Improved implementation of the retrieval of optimal primal variables with CPLEX. With the previous versions there was an important overhead at the end of the solving process to get the optimal values, this is now working much faster. 
     * Nicer documentation.
     
* 18 May 14: **Picos** :ref:`1.0.1.dev <download>` **Released** |br|
   
   Major Release with following changes:
     * Support for Semidefinite Programming over the complex domain, see :ref:`here <complex>`.
     * Flow constraints in graphs, cf. :ref:`this section <flowcons>`.
     * Improved implementation of ``__getitem__`` for affine expressions. The slicing of affine expressions
       was slowing down (a lot!) the processing of the optimization problem.

 * 19 Jul. 13: **Picos** :ref:`1.0.0 <download>` **Released** |br|
   
   Major Release with following changes:
     * Semidefinite Programming Interface for MOSEK 7.0 !!!
     * New options ``handleBarVars`` and ``handleConeVars`` to customize how SOCP and SDPs are passed to MOSEK
       (When these options are set to ``True`` , PICOS tries to minimize the number of variables of the
       MOSEK instance, see the doc in :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`).
     * The function :func:`dualize() <picos.Problem.dualize>` returns the Lagrangian dual of a Problem.
     * The option ``solve_via_dual`` (documented in
       :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>` ) allows the user to pass
       the dual of a problem to a solver, instead of the primal problem itself. This can yield important speed-up for
       certain problems.
     * In addition to the geometric mean function :func:`picos.geomean() <picos.tools.geomean>` , it is now possible
       to pass rational powers of affine expressions (through an overload of the ``**`` operator), trace of
       matrix powers with :func:`picos.tracepow() <picos.tools.tracepow>` , (generalized) p-norms
       with :func:`picos.norm() <picos.tools.norm>`, and nth root of a determinant with
       :func:`picos.detrootn() <picos.tools.detrootn>`. These functions automatically reformulate the entered inequalities as a set of equivalent SOCP or SDP constraints.
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
   
   Bug-fix release, correcting:
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