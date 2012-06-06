.. _optdes:

*****************************************
Examples from Optimal Experimental Design
*****************************************

Optimal experimental design is a theory
at the interface of statistics and optimization,
which studies how to allocate some experimental
effort within a set of available expeiments.
The goal is to allow for the best possible
estimation of an unknown parameter :math:`\theta`.
In what follows, we assume the standard linear model with
multiresponse experiments: the :math:`i^{\textrm{th}}`
experiment gives a multidimensional observation that
can be written as :math:`y_i = A_i^T \theta+\epsilon_i`,
where :math:`y_i` is of dimension :math:`l_i`,
:math:`A_i` is a :math:`m \times l_i-` matrix,
and the noise vectors :math:`\epsilon_i` are i.i.d. with a unit variance.

Several optimization criterions exist, leading to different SDP, SOCP and LP
formulations.
As such, optimal experimental design problens are natural examples for problems
in conic optimization.
For a review of the different formulations,
see `this article <http://arxiv.org/abs/0912.5467>`_.

The code below initializes the data used in all the examples of this page.
It should be run prior to any of the codes presented in this page.

.. testcode::
        
        import cvxopt as cvx
        import picos as pic
        
        #---------------------------------#
        # First generate some data :      #
        #       _ a list of 8 matrices A  #
        #       _ a vector c              #
        #---------------------------------#
        A=[ cvx.matrix([[1,0,0,0,0],
                        [0,3,0,0,0],
                        [0,0,1,0,0]]),
        cvx.matrix([[0,0,2,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0]]),
        cvx.matrix([[0,0,0,2,0],
                        [4,0,0,0,0],
                        [0,0,1,0,0]]),
        cvx.matrix([[1,0,0,0,0],
                        [0,0,2,0,0],
                        [0,0,0,0,4]]),
        cvx.matrix([[1,0,2,0,0],
                        [0,3,0,1,2],
                        [0,0,1,2,0]]),
        cvx.matrix([[0,1,1,1,0],
                        [0,3,0,1,0],
                        [0,0,2,2,0]]),
        cvx.matrix([[1,2,0,0,0],
                        [0,3,3,0,5],
                        [1,0,0,2,0]]),
        cvx.matrix([[1,0,3,0,1],
                        [0,3,2,0,0],
                        [1,0,0,2,0]])
        ]
        
        c = cvx.matrix([1,2,3,4,5])

c-optimality, multi-response: SOCP
==================================

We compute the c-optimal design (c=[1,2,3,4,5])
for the observation matrices ``A[i].T`` from the variable ``A`` defined above.
The results below suggest that we should allocate 12.8% of the
experimental effort on experiment #5, and 87.2% on experiment #7.

Primal Problem
''''''''''''''

The SOCP for multiresponse c-optimal design is:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{\substack{\mu \in \mathbb{R}^s\\ 
                        \forall i \in [s],\ z_i \in \mathbb{R}^{l_i}}}{\mbox{minimize}}
                      & \sum_{i=1}^s \mu_i\\
   &\mbox{subject to} & \sum_{i=1}^s A_i z_i = c\\
   &                  & \forall i \in [s],\ \Vert z_i \Vert_2 \leq \mu_i,
   \end{eqnarray*}
   \end{center}


.. testcode::
        
        #create the problem, variables and params
        prob_primal_c=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        AA=pic.new_param('A',AA)
        cc=pic.new_param('c',c)
        z=[prob_primal_c.add_variable('z['+str(i)+']',AA[i].size[1]) for i in range(s)]
        mu=prob_primal_c.add_variable('mu',s)

        #define the constraints and objective function
        prob_primal_c.add_list_of_constraints(
                [abs(z[i])<mu[i] for i in range(s)], #constraints
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_primal_c.add_constraint( 
                pic.sum(
                        [AA[i]*z[i] for i in range(s)], #summands
                        'i', #index
                        '[s]' #set to which the index belongs
                        )  
                == cc )
        prob_primal_c.set_objective('min',1|mu)
        
        #solve the problem and retrieve the optimal weights of the optimal design.
        print prob_primal_c
        prob_primal_c.solve(verbose=0,solver='cvxopt')
        
        mu=mu.value
        w=mu/sum(mu) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w

Generated output:

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SOCP):
        32 variables, 5 affine constraints, 24 vars in a SO cone

        z   : list of 8 variables, (3, 1), continuous
        mu  : (8, 1), continuous

                minimize 〈 |1| | mu 〉
        such that
        ||z[i]|| < mu[i] for all i in [s]
        Σ_{i in [s]} A[i]*z[i] = c
        ---------------------

        The optimal deign is:
        [...]
        [...]
        [...]
        [...]
        [ 1.28e-01]
        [...]
        [ 8.72e-01]
        [...]

The ``[...]`` above indicate a numerical zero entry
(*i.e., which can be something like 2.84e-10*).
We use the ellipsis ``...`` instead for clarity and compatibility with **doctest**.

Dual Problem
''''''''''''

This is only to check that we obtain the same solution with the dual problem,
and to provide one additional example in this doc:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{u \in \mathbb{R}^m}{\mbox{maximize}}
                      & c^T u\\
   &\mbox{subject to} & \forall i \in [s],\ \Vert A_i^T u \Vert_2 \leq 1
   \end{eqnarray*}
   \end{center}

.. testcode::
        
        #create the problem, variables and params
        prob_dual_c=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        AA=pic.new_param('A',AA)
        cc=pic.new_param('c',c)
        u=prob_dual_c.add_variable('u',c.size)

        #define the constraints and objective function
        prob_dual_c.add_list_of_constraints(
                [abs(AA[i].T*u)<1 for i in range(s)], #constraints
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_dual_c.set_objective('max', cc|u)
        
        #solve the problem and retrieve the weights of the optimal design 
        print prob_dual_c
        prob_dual_c.solve(verbose=0)
        
        mu = [prob_dual_c.get_constraint(i).dual[0] for i in range(s)] #Lagrangian duals of the SOC constraints
        mu = cvx.matrix(mu)
        w=mu/sum(mu) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w

Generated output:

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SOCP):
        5 variables, 0 affine constraints, 24 vars in a SO cone

        u   : (5, 1), continuous

                maximize 〈 c | u 〉
        such that
        ||A[i].T*u|| < 1 for all i in [s]
        ---------------------
        
        The optimal deign is:
        [...]
        [...]
        [...]
        [...]
        [ 1.28e-01]
        [...]
        [ 8.72e-01]
        [...]


c-optimality, single-response: LP
=================================

When the observation matrices are row vectors (single-response framework),
the SOCP above reduces to a simple LP, because the variables
:math:`z_i` are scalar.
We solve below the LP for the case where there are 12
available experiments, corresponding to the columns of the matrices
``A[4]``, ``A[5]``, ``A[6]``, and ``A[7]`` defined in the preambule.

The optimal design allocates 3.37% to experiment #5 (2nd column of ``A[5]``),
27.9% to experiment #7 (1st column of ``A[6]``),
11.8% to experiment #8 (2nd column of ``A[6]``),
27.6% to experiment #9 (3rd column of ``A[6]``),
and 29.3% to experiment #11 (2nd column of ``A[7]``).

.. testcode::
        
        #create the problem, variables and params
        prob_LP=pic.Problem()
        AA=[cvx.sparse(a[:,i],tc='d') for i in range(3) for a in A[4:]] #12 column vectors
        s=len(AA)
        AA=pic.new_param('A',AA)
        cc=pic.new_param('c',c)
        z=[prob_LP.add_variable('z['+str(i)+']',1) for i in range(s)]
        mu=prob_LP.add_variable('mu',s)
        
        #define the constraints and objective function
        prob_LP.add_list_of_constraints(
                [abs(z[i])<mu[i] for i in range(s)], #constraints handled as -mu_i < z_i< mu_i
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_LP.add_constraint( 
                pic.sum(
                        [AA[i]*z[i] for i in range(s)], #summands
                        'i', #index
                        '[s]' #set to which the index belongs
                        )  
                == cc )
        prob_LP.set_objective('min',1|mu)
        
        #solve the problem and retrieve the weights of the optimal design
        print prob_LP
        prob_LP.solve(verbose=0)
        
        mu=mu.value
        w=mu/sum(mu) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w

Note that there are no cone constraints, because
the constraints of the form :math:`|z_i| \leq \mu_i` are handled as two
inequalities when :math:`z_i` is scalar, so the problem is a LP indeed:

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (LP):
        24 variables, 29 affine constraints

        z   : list of 12 variables, (1, 1), continuous
        mu  : (12, 1), continuous

                minimize 〈 |1| | mu 〉
        such that
        ||z[i]|| < mu[i] for all i in [s]
        Σ_{i in [s]} A[i]*z[i] = c
        ---------------------

        The optimal deign is:
        [...]
        [...]
        [...]
        [...]
        [ 3.37e-02]
        [...]
        [ 2.79e-01]
        [ 1.18e-01]
        [ 2.76e-01]
        [...]
        [ 2.93e-01]
        [...]

A-optimality: SOCP
==================

We compute the A-optimal design
for the observation matrices ``A[i].T`` defined in the preambule.
The optimal design allocates
24.9% on experiment #3,
14.2% on experiment #4,
8.51% on experiment #5,
12.1% on experiment #6,
13.2% on experiment #7,
and 27.0% on experiment #8.

        [ 2.49e-01]
        [ 1.42e-01]
        [ 8.51e-02]
        [ 1.21e-01]
        [ 1.32e-01]
        [ 2.70e-01]

Primal Problem
''''''''''''''

The SOCP for the A-optimal design problem is:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{\substack{\mu \in \mathbb{R}^s\\ 
                        \forall i \in [s],\ Z_i \in \mathbb{R}^{l_i \times m}}}{\mbox{minimize}}
                      & \sum_{i=1}^s \mu_i\\
   &\mbox{subject to} & \sum_{i=1}^s A_i Z_i = I\\
   &                  & \forall i \in [s],\ \Vert Z_i \Vert_F \leq \mu_i,
   \end{eqnarray*}
   \end{center}


.. testcode::

        #create the problem, variables and params
        prob_primal_A=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        AA=pic.new_param('A',AA)
        Z=[prob_primal_A.add_variable('Z['+str(i)+']',AA[i].T.size) for i in range(s)]
        mu=prob_primal_A.add_variable('mu',s)
        
        #define the constraints and objective function
        prob_primal_A.add_list_of_constraints(
                [abs(Z[i])<mu[i] for i in range(s)], #constraints
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_primal_A.add_constraint( 
                pic.sum(
                [AA[i]*Z[i] for i in range(s)], #summands
                'i', #index
                '[s]' #set to which the index belongs
                )  
                == 'I' )
        prob_primal_A.set_objective('min',1|mu)
        
        #solve the problem and retrieve the weights of the optimal design
        print prob_primal_A
        prob_primal_A.solve(verbose=0)
        w=mu.value
        w=w/sum(w) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SOCP):
        128 variables, 25 affine constraints, 120 vars in a SO cone

        Z   : list of 8 variables, (3, 5), continuous
        mu  : (8, 1), continuous

                minimize 〈 |1| | mu 〉
        such that
        ||Z[i]|| < mu[i] for all i in [s]
        Σ_{i in [s]} A[i]*Z[i] = I
        ---------------------

        The optimal deign is:
        [...]
        [...]
        [ 2.49e-01]
        [ 1.42e-01]
        [ 8.51e-02]
        [ 1.21e-01]
        [ 1.32e-01]
        [ 2.70e-01]




Dual Problem
''''''''''''

This is only to check that we obtain the same solution with the dual problem,
and to provide one additional example in this doc:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{U \in \mathbb{R}^{m \times m}}{\mbox{maximize}}
                      &  \mbox{trace}\ U\\
   &\mbox{subject to} & \forall i \in [s],\ \Vert A_i^T U \Vert_2 \leq 1
   \end{eqnarray*}
   \end{center}

.. testcode::

        #create the problem, variables and params
        prob_dual_A=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        m=AA[0].size[0]
        AA=pic.new_param('A',AA)
        U=prob_dual_A.add_variable('U',(m,m))

        #define the constraints and objective function
        prob_dual_A.add_list_of_constraints(
                [abs(AA[i].T*U)<1 for i in range(s)], #constraints
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_dual_A.set_objective('max', 'I'|U)

        #solve the problem and retrieve the weights of the optimal design
        print prob_dual_A
        prob_dual_A.solve(verbose = 0)

        mu = [prob_dual_A.get_constraint(i).dual[0] for i in range(s)] #Lagrangian duals of the SOC constraints
        mu = cvx.matrix(mu)
        w=mu/sum(mu) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SOCP):
        25 variables, 0 affine constraints, 120 vars in a SO cone

        U   : (5, 5), continuous

                maximize trace( U )
        such that
        ||A[i].T*U|| < 1 for all i in [s]
        ---------------------

        The optimal deign is:
        [...]
        [...]
        [ 2.49e-01]
        [ 1.42e-01]
        [ 8.51e-02]
        [ 1.21e-01]
        [ 1.32e-01]
        [ 2.70e-01]

SDP formulation of the c-optimal design problem
===============================================

We give below the SDP for c-optimality, in primal and dual
form. You can observe that we obtain the same results as
with the SOCP presented earlier:
12.8% on experiment #5, and 87.2% on experiment #7.

Primal Problem
''''''''''''''

The SDP formulation of the c-optimal design problem is:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{\mu \in \mathbb{R}^s}{\mbox{minimize}}
                      & \sum_{i=1}^s \mu_i\\
   &\mbox{subject to} & \sum_{i=1}^s \mu_i A_i A_i^T \succeq c c^T,\\
   &                  & \mu \geq 0.
   \end{eqnarray*}
   \end{center}

.. testcode::

        #create the problem, variables and params
        prob_SDP_c_primal=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        AA=pic.new_param('A',AA)
        cc=pic.new_param('c',c)
        mu=prob_SDP_c_primal.add_variable('mu',s)

        #define the constraints and objective function
        prob_SDP_c_primal.add_constraint( 
                pic.sum(
                [mu[i]*AA[i]*AA[i].T for i in range(s)], #summands
                'i', #index
                '[s]' #set to which the index belongs
                )  
                >> cc*cc.T )
        prob_SDP_c_primal.add_constraint(mu>0)
        prob_SDP_c_primal.set_objective('min',1|mu)

        #solve the problem and retrieve the weights of the optimal design
        print prob_SDP_c_primal
        prob_SDP_c_primal.solve(verbose=0)
        w=mu.value
        w=w/sum(w) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SDP):
        8 variables, 8 affine constraints, 15 vars in a SD cone

        mu  : (8, 1), continuous

                minimize 〈 |1| | mu 〉
        such that
        Σ_{i in [s]} mu[i]*A[i]*A[i].T ≽ c*c.T
        mu > |0|
        ---------------------

        The optimal deign is:
        [...]
        [...]
        [...]
        [...]
        [ 1.28e-01]
        [...]
        [ 8.72e-01]
        [...]

Dual Problem
''''''''''''

This is only to check that we obtain the same solution with the dual problem,
and to provide one additional example in this doc:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{X \in \mathbb{R}^{m \times m}}{\mbox{maximize}}
                      &  c^T X c\\
   &\mbox{subject to} & \forall i \in [s],\ \langle A_i A_i^T,\ X \rangle \leq 1,\\
   &                  &  X \succeq 0.
   \end{eqnarray*}
   \end{center}


.. testcode::

        #create the problem, variables and params
        prob_SDP_c_dual=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        AA=pic.new_param('A',AA)
        cc=pic.new_param('c',c)
        m =c.size[0]
        X=prob_SDP_c_dual.add_variable('X',(m,m),vtype='symmetric')

        #define the constraints and objective function
        prob_SDP_c_dual.add_list_of_constraints(
                [(AA[i]*AA[i].T | X ) <1 for i in range(s)], #constraints
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_SDP_c_dual.add_constraint(X>>0)
        prob_SDP_c_dual.set_objective('max', cc.T*X*cc)
        
        #solve the problem and retrieve the weights of the optimal design
        print prob_SDP_c_dual
        prob_SDP_c_dual.solve(verbose=0,solver='smcp')
        mu = [prob_SDP_c_dual.get_constraint(i).dual[0] for i in range(s)] #Lagrangian duals of the SOC constraints
        mu = cvx.matrix(mu)
        w=mu/sum(mu) #normalize mu to get the optimal weights
        print
        print 'The optimal deign is:'
        print w
        print X
        

.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SDP):
        15 variables, 8 affine constraints, 15 vars in a SD cone

        X   : (5, 5), symmetric

                maximize c.T*X*c
        such that
        〈 A[i]*A[i].T | X 〉 < 1.0 for all i in [s]
        X ≽ |0|
        ---------------------

        The optimal deign is:
        [...]
        [...]
        [...]
        [...]
        [ 1.28e-01]
        [...]
        [ 8.72e-01]
        [...]

Exact A-optimal design: MISOCP
==============================

In the exact version of A-optimality, a number :math:`N \in \mathbb{N}`
of experiments is given, and the goal is to find the optimal number of times
:math:`n_i \in \mathbb{N}` that the experiment #i should be performed, 
with :math:`\sum_i n_i =N`.

After a change of variable in the SOCP formulation of A-optimality,
we can express this exact optimal design problem as a MISOCP:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{\substack{t \in \mathbb{R}\\
                        \mathbf{n} \in \mathbb{N}^s\\
                        \forall i \in [s],\ Z_i \in \mathbb{R}^{l_i \times m}}}{\mbox{maximize}}
                      & t\\
   &\mbox{subject to} & \sum_{i=1}^s A_i Z_i = t I\\
   &                  & \forall i \in [s],\ \Vert Z_i \Vert_F \leq n_i,\\
   &                  & \sum_{i=1}^s n_i = N.
   \end{eqnarray*}
   \end{center}

The eact optimal design is :math:`\mathbf{n}=[0,0,5,3,1,3,3,5]`:

.. testcode::

        #create the problem, variables and params
        prob_exact_A=pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
        s=len(AA)
        m=AA[0].size[0]
        AA=pic.new_param('A',AA)
        cc=pic.new_param('c',c)
        N =pic.new_param('N',20) #number of experiments allowed
        I =pic.new_param('I',cvx.spmatrix([1]*m,range(m),range(m),(m,m))) #identity matrix
        Z=[prob_exact_A.add_variable('Z['+str(i)+']',AA[i].T.size) for i in range(s)]
        n=prob_exact_A.add_variable('n',s, vtype='integer')
        t=prob_exact_A.add_variable('t',1)

        #define the constraints and objective function
        prob_exact_A.add_list_of_constraints(
                [abs(Z[i])<n[i] for i in range(s)], #constraints
                'i', #index
                '[s]' #set to which the index belongs
                )
        prob_exact_A.add_constraint( 
                pic.sum(
                [AA[i]*Z[i] for i in range(s)], #summands
                'i', #index
                '[s]' #set to which the index belongs
                )  
                == t*I )
                
        prob_exact_A.add_constraint( 1|n < N )
        prob_exact_A.set_objective('max',t)

        #solve the problem and display the optimal design
        print prob_exact_A
        prob_exact_A.solve(verbose = 0)
        print n
        
.. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (MISOCP):
        129 variables, 26 affine constraints, 120 vars in a SO cone

        Z   : list of 8 variables, (3, 5), continuous
        n   : (8, 1), integer
        t   : (1, 1), continuous

                maximize t
        such that
        ||Z[i]|| < n[i] for all i in [s]
        Σ_{i in [s]} A[i]*Z[i] = t*I
        〈 |1| | n 〉 < N
        ---------------------
        [...]
        [...]
        [ 5.00e+00]
        [ 3.00e+00]
        [ 1.00e+00]
        [ 3.00e+00]
        [ 3.00e+00]
        [ 5.00e+00]

D-optimal design: SOCP
======================

The D-optimal design problem has a convex programming formulation:

.. math::
   :nowrap:   

   \begin{center}
   \begin{eqnarray*}
   &\underset{\substack{\mathbf{L} \in \mathbb{R}^{m \times m}\\
                        \mathbf{w} \in \mathbb{R}^s\\
                        \forall i \in [s],\ V_i \in \mathbb{R}^{l_i \times m}}}{\mbox{maximize}}
                      & \log \prod_{i=1}^m L_{i,i}\\
   &\mbox{subject to} & \sum_{i=1}^s A_i V_i = L,\\
   &                  & L\ \mbox{lower triangular},\\
   &                  & \Vert V_i \Vert_F \leq \sqrt{m}\ w_i,\\
   &                  & \sum_{i=1}^s w_i \leq 1.
   \end{eqnarray*}
   \end{center}

By introducing new SOC constraints, we can create a variable :math:`u_{01234}`
such that :math:`u_{01234}^8 \leq \prod_{i=0}^4 L_{i,i}`. Hence, the
D-optimal problem can be solved by second order cone programming. The example
below allocates respectively 22.7%, 3.38%, 1.65%, 5.44%, 31.8% and 35.1%
to the experiments #3 to #8.

.. testcode::
        
        #create the problem, variables and params
        prob_D = pic.Problem()
        AA=[cvx.sparse(a,tc='d') for a in A]
        s=len(AA)
        m=AA[0].size[0]
        AA=pic.new_param('A',AA)
        mm=pic.new_param('m',m)
        L=prob_D.add_variable('L',(m,m))
        V=[prob_D.add_variable('V['+str(i)+']',AA[i].T.size) for i in range(s)]
        w=prob_D.add_variable('w',s)
        #additional variables to handle the product of the diagonal elements of L
        u={}
        for k in ['01','23','4.','0123','4...','01234']:
                u[k] = prob_D.add_variable('u['+k+']',1)


        #define the constraints and objective function
        prob_D.add_constraint(
                        pic.sum([AA[i]*V[i]
                        for i in range(s)],'i','[s]')
                        == L)
        #L is lower triangular
        prob_D.add_list_of_constraints( [L[i,j] == 0
                                        for i in range(m)
                                        for j in range(i+1,m)],['i','j'],'upper triangle')
        prob_D.add_list_of_constraints([abs(V[i])<(mm**0.5)*w[i]
                                        for i in range(s)],'i','[s]')
        prob_D.add_constraint(1|w<1)
        #SOC constraints to define u['01234'] such that u['01234']**8 < L[0,0] * L[1,1] * ... * L[4,4]
        prob_D.add_constraint(u['01']**2   <L[0,0]*L[1,1])
        prob_D.add_constraint(u['23']**2   <L[2,2]*L[3,3])
        prob_D.add_constraint(u['4.']**2   <L[4,4])
        prob_D.add_constraint(u['0123']**2 <u['01']*u['23'])
        prob_D.add_constraint(u['4...']**2 <u['4.'])
        prob_D.add_constraint(u['01234']**2<u['0123']*u['4...'])

        prob_D.set_objective('max',u['01234'])

        #solve the problem and display the optimal design
        print prob_D
        prob_D.solve(verbose=0)
        print w

.. testoutput::
        .. testoutput::
        :options: +NORMALIZE_WHITESPACE, +ELLIPSIS
        
        ---------------------
        optimization problem  (SOCP):
        159 variables, 36 affine constraints, 126 vars in a SO cone

        V   : list of 8 variables, (3, 5), continuous
        u   : dict of 6 variables, (1, 1), continuous
        L   : (5, 5), continuous
        w   : (8, 1), continuous

                maximize u[01234]
        such that
        L = Σ_{i in [s]} A[i]*V[i]
        L[i,j] = 0 for all (i,j) in upper triangle
        ||V[i]|| < (m)**0.5*w[i] for all i in [s]
        〈 |1| | w 〉 < 1.0
        ||u[01]||^2 < ( L[0,0])( L[1,1])
        ||u[23]||^2 < ( L[2,2])( L[3,3])
        ||u[4.]||^2 < L[4,4]
        ||u[0123]||^2 < ( u[01])( u[23])
        ||u[4...]||^2 < u[4.]
        ||u[01234]||^2 < ( u[0123])( u[4...])
        ---------------------
        [...]
        [...]
        [ 2.27e-01]
        [ 3.38e-02]
        [ 1.65e-02]
        [ 5.44e-02]
        [ 3.18e-01]
        [ 3.51e-01]
