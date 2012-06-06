.. _intro:

************
Introduction
************

PICOS is a user friendly interface
to several conic and integer programming solvers,
very much like `YALMIP <http://users.isy.liu.se/johanl/yalmip/>`_ under
`MATLAB <http://www.mathworks.com/>`_.

The main motivation for PICOS is to have the possibility to
enter an optimization problem as a *high level model*,
and to be able to solve it with several *different solvers*.
Multidimensional and matrix variables are handled in a natural fashion.
This is very useful to quickly implement some models and
test their validity on simple examples.
Furthermore, with PICOS you can take advantage of the
python programming language to read and write data,
construct a list of constraints by using python list comprehensions,
take slices of multidimensional variables, etc.

It must also be said that PICOS is only a unified interface to other
already existing interfaces of optimization solvers. So you have
to install some additional packages each time you want to use PICOS with a new solver
(see :ref:`a list of supported solvers <solvers>`, and the :ref:`packages you will
have to install <requirements>` to use them). Furthermore, since PICOS is just another
interface layer, one should expect an overhead due to PICOS in the solution time.

Here is a very simple example of the usage of PICOS:

>>> import picos as pic
>>> prob = pic.Problem()
>>> x = prob.add_variable('x',1, vtype='integer') #scalar integer variable
>>> prob.add_constraint(x<5.2)                    #x less or equal to 5.2
>>> prob.set_objective('max',x)                   #maximize x
>>> print prob #doctest: +NORMALIZE_WHITESPACE
---------------------
optimization problem (MIP):
1 variables, 1 affine constraints
x   : (1, 1), integer
    maximize x
such that
  x < 5.2
---------------------
>>> sol = prob.solve(solver='zibopt',verbose=0)
>>> print x                                #optimal value of x #doctest: +NORMALIZE_WHITESPACE
5.0


Currently, PICOS can handle the following class of
optimzation problems. A list of currently
interfaced solvers can be found :ref:`here <solvers>`.

  * Linear Programming (**LP**)
  * Mixed Integer Programming (**MIP**)
  * Convex Quadratically constrained Quadratic Programming (**convex QCQP**)
  * Second Order Cone Programming (**SOCP**)
  * Semidefinite Programming (**SDP**)

.. * General Quadratically constrained Quadratic Programming (**QCQP**) (TODO ??)
.. * Mixed Integer Quadratic Programming (**MIQP**) (TODO ??)


There exists a number of similar projects, so we list below their
main differences with PICOS:

  * `CVXPY <http://www.stanford.edu/~ttinoco/cvxpy/>`_:
    
    This is a python interface
    that can be used to solve any convex optimization
    problem. However, CVXPY interfaces only the open
    source solver `cvxopt <http://abel.ee.ucla.edu/cvxopt/>`_ for disciplined convex programming (**DCP**)

  * `Numberjack <http://numberjack.ucc.ie/home>`_:

    This python package also provides an interface to the integer programming solver `scip <http://zibopt.zib.de/>`_,
    as well as satisfiability (**SAT**) and constraint programming solvers (**CP**). 

  * `puLP <http://packages.python.org/PuLP/>`_:
    
    A user-friendly interface to a bunch of **LP** and **MIP** solvers.

  * `python-zibopt <http://code.google.com/p/python-zibopt/>`_:

    This is a user-friendly interface to the `ZIB optimization suite <http://zibopt.zib.de/>`_
    for solving mixed integer programs (**MIP**). PICOS
    provides an interface to this interface.

  .. Todo::
        
        pyomo and openopt

First Example
=============

We give below a simple example of the use of PICOS, to solve
an SOCP which arises in *optimal experimental design*.
Given some observation matrices :math:`A_1,\ldots,A_s`,
with :math:`A_i \in \mathbb{R}^{m \times l_i}`,
and a coefficient matrix :math:`K \in \mathbb{R}^{m \times r}`,
the problem to solve is:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{\substack{\mu \in \mathbb{R}^s\\ 
                        \forall i \in [s],\ Z_i \in \mathbb{R}^{l_i \times r}}}{\mbox{minimize}}
                      & \sum_{i=1}^s \mu_i\\
   &\mbox{subject to} & \sum_{i=1}^s A_i Z_i = K\\
   &                  & \forall i \in [s],\ \Vert Z_i \Vert_F \leq \mu_i,
   \end{eqnarray*}
   \end{center}

where :math:`\Vert M \Vert_F := \sqrt{\mbox{trace} M M^T}` denotes the 
Frobenius norm of
:math:`M`. This problem can be entered and solved as follows with PICOS:

.. testcode::
        
        import picos as pic
        import cvxopt as cvx
        
        #generate data
        A = [   cvx.sparse([[1 ,2 ,0 ],
                            [2 ,0 ,0 ]]),
                cvx.sparse([[0 ,2 ,2 ]]),
                cvx.sparse([[0 ,2 ,-1],
                            [-1,0 ,2 ],
                            [0 ,1 ,0 ]])
            ]
        K = cvx.sparse([[1 ,1 ,1 ],
                        [1 ,-5,-5]])
        
        #size of the data
        s = len(A)
        m = A[0].size[0]
        l = [ Ai.size[1] for Ai in A ]
        r = K.size[1]
        
        #creates a problem and the optimization variables
        prob = pic.Problem()
        mu = prob.add_variable('mu',s)
        Z  = [prob.add_variable('Z[' + str(i) + ']', (l[i],r))
              for i in range(s)]

        #convert the constants into params of the problem
        A = pic.new_param('A',A)
        K = pic.new_param('K',K)

        #add the constraints
        prob.add_constraint( pic.sum([ A[i]*Z[i] for i in range(s)], #summands
                                    'i',                            #name of the index
                                    '[s]'                           #set to which the index belongs
                                   ) == K
                           )
        prob.add_list_of_constraints( [ abs(Z[i]) < mu[i] for i in range(s)], #constraints
                                      'i',                                    #index of the constraints
                                      '[s]'                                   #set to which the index belongs
                                    )
        
        #sets the objective
        prob.set_objective('min', 1 | mu ) # scalar product of the vector of all ones with mu

        #display the problem
        print prob

        #call to the solver cvxopt
        sol = prob.solve(solver='cvxopt', verbose = 0)

        #show the value of the optimal variable
        print '\n  mu ='
        print mu

        #show the dual variable of the equality constraint
        print'\nThe optimal dual variable of the'
        print prob.get_constraint(0)
        print 'is :'
        print prob.get_constraint(0).dual

This generates the output:

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    ---------------------
    optimization problem (SOCP):
    15 variables, 6 affine constraints, 12 vars in a SO cone
    
    mu  : (3, 1), continuous
    Z   : list of 3 variables, different sizes, continuous
    
        minimize 〈 |1| | mu 〉
    such that
      Σ_{i in [s]} A[i]*Z[i] = K
      ||Z[i]|| < mu[i] for all i in [s]
    ---------------------

      mu =
    [ 6.60e-01]
    [ 2.42e+00]
    [ 1.64e-01]


    The optimal dual variable of the
    # (3x2)-affine constraint : Σ_{i in [s]} A[i]*Z[i] = K #
    is :
    [-3.41e-01]
    [ 9.17e-02]
    [-1.88e-01]
    [-3.52e-01]
    [ 2.32e-01]
    [ 2.59e-01]



.. _solvers:

Solvers
=======

Below is a list of the solvers currently interfaced by PICOS.
We have indicated the classes of optimization problems that
the solver can handle via PICOS. Note however
that the solvers listed below might have other
features that are *not handled by PICOS*.

  * `cvxopt <http://abel.ee.ucla.edu/cvxopt/>`_ (LP, convex QCQP, SOCP, SDP)
  * `smcp <http://abel.ee.ucla.edu/smcp/>`_ (LP, SOCP, SDP)
  * `mosek <http://www.mosek.com>`_ (LP, MIP, SOCP, convex QCQP)
  * `cplex <http://www.ibm.com/software/integration/optimization/cplex-optimizer/>`_ (LP, MIP)
  * `soplex <http://soplex.zib.de/>`_ (LP)
  * `scip <http://scip.zib.de/>`_ (MIP, MIQP)


.. _requirements:

Requirements
============

Installation
============
