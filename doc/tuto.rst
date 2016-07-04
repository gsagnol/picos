:tocdepth: 3

.. _tutocontent:

************
**Tutorial**
************

First of all, let us import the PICOS module and cvxopt

  >>> import picos as pic
  >>> import cvxopt as cvx

We now generate some arbitrary data, that we will use in this tutorial.

  >>> pairs = [(0,2), (1,4), (1,3), (3,2), (0,4),(2,4)]  #a list of pairs
  >>> A = []
  >>> b = ( [0 ,2 ,0 ,3 ],                               #a tuple of 5 lists, each of length 4
  ...       [1 ,1 ,0 ,5 ],
  ...       [-1,0 ,2 ,4 ],
  ...       [0 ,0 ,-2,-1],
  ...       [1 ,1 ,0 ,0 ]
  ...     )
  >>> for i in range(5):
  ...     A.append(cvx.matrix(range(i-3,i+5),(2,4)))     #A is a list of 2x4 matrices
  >>> D={'Peter': 12,
  ...    'Bob'  : 4,
  ...    'Betty': 7,
  ...    'Elisa': 14
  ...    }

Let us now create an instance ``prob`` of an optimization problem

  >>> prob = pic.Problem()    #create a Problem instance

===========
*Variables*
===========


We will now create the variables of our optimization problem. This is done
by calling the method :func:`add_variable() <picos.Problem.add_variable>`.
This function adds an instance of the class :class:`Variable <picos.Variable>`
in the dictionary ``prob.variables``, and returns a reference
to the freshly added variable.
As we will next see, we
can use
this :class:`Variable <picos.Variable>`
to form affine and quadratic expressions.

  >>> t = prob.add_variable('t',1) #a scalar
  >>> x = prob.add_variable('x',4) #a column vector
  >>> Y = prob.add_variable('Y',(2,4)) #a matrix
  >>> Z = []
  >>> for i in range(5):
  ...     Z.append( prob.add_variable('Z[{0}]'.format(i),(4,2))  )# a list of 5 matrices
  >>> w={}
  >>> for p in pairs:   #a dictionary of (scalar) binary variables, indexed by our pairs
  ...     w[p] = prob.add_variable('w[{0}]'.format(p),1 , vtype='binary')

Now, if we try to display a variable, here is what we get:

  >>> w[2,4]
  # variable w[(2, 4)]:(1 x 1),binary #
  >>> Y
  # variable Y:(2 x 4),continuous #

Also note the use of the
attributes :attr:`name <picos.Variable.name>`, :attr:`value <picos.Variable.value>`,
:attr:`size <picos.Variable.size>`, and :attr:`vtype <picos.Variable.vtype>`:

  >>> w[2,4].vtype
  'binary'
  >>> x.vtype
  'continuous'
  >>> x.vtype='integer'
  >>> x
  # variable x:(4 x 1),integer #
  >>> x.size
  (4, 1)
  >>> Z[1].value = A[0].T
  >>> Z[0].is_valued()
  False
  >>> Z[1].is_valued()
  True
  >>> Z[2].name
  'Z[2]'

The admissible values for the ``vtype`` attribute are documented in :func:`add_variable() <picos.Problem.add_variable>`.

                     
====================
*Affine Expressions*
====================

We will now use our variables to create some affine expressions,
which are stored as instance of the class :class:`AffinExp <picos.AffinExp>`,
and will be the
core to define an optimization problem. Most python operators have been overloaded
to work with instances of :class:`AffinExp <picos.AffinExp>`
(a list of available overloaded operators can be found in the
doc of :class:`AffinExp <picos.AffinExp>`). For example,
you can form the sum of two variables by writing:

  >>> Z[0]+Z[3]
  # (4 x 2)-affine expression: Z[0] + Z[3] #

The transposition of an affine expression is done by appending ``.T``:

  >>> x
  # variable x:(4 x 1),integer #
  >>> x.T
  # (1 x 4)-affine expression: x.T #

Parameters as constant affine expressions
-----------------------------------------

It is also possible to form affine expressions by using parameters
stored in data structures such as a ``list`` or a :func:`cvxopt matrix <cvxopt:cvxopt.matrix>`
(In fact, any type that is recognizable by the function :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`).

  >>> x + b[0]
  # (4 x 1)-affine expression: x + [ 4 x 1 MAT ] #
  >>> x.T + b[0]
  # (1 x 4)-affine expression: x.T + [ 1 x 4 MAT ] #
  >>> A[0] * Z[0] + A[4] * Z[4]
  # (2 x 2)-affine expression: [ 2 x 4 MAT ]*Z[0] + [ 2 x 4 MAT ]*Z[4] #

In the above example, you see that the list ``b[0]`` was correctly converted into
a  :math:`4 \times 1` vector in the first expression, and into
a  :math:`1 \times 4` vector in the second one. This is because the overloaded
operators always try to convert the data into matrices of the appropriate size.

If you want to have better-looking string representations of your affine expressions,
you will need to convert the parameters into constant affine expressions. This can be done
thanks to the function :func:`new_param() <picos.tools.new_param>`:

  >>> A = pic.new_param('A',A)              #this creates a list of constant affine expressions [A[0],...,A[4]]
  >>> b = pic.new_param('b',b)              #this creates a list of constant affine expressions [b[0],...,b[4]]
  >>> D = pic.new_param('D',D)              #this creates a dictionary of constant AffExpr, indexed by 'Peter', 'Bob', ...
  >>> alpha = pic.new_param('alpha',12)     #a scalar parameter
  
  >>> alpha
  # (1 x 1)-affine expression: alpha #
  >>> D['Betty']
  # (1 x 1)-affine expression: D[Betty] #
  >>> b                                         #doctest: +NORMALIZE_WHITESPACE
  [# (4 x 1)-affine expression: b[0] #,
   # (4 x 1)-affine expression: b[1] #,
   # (4 x 1)-affine expression: b[2] #,
   # (4 x 1)-affine expression: b[3] #,
   # (4 x 1)-affine expression: b[4] #]
  >>> print b[0]
  [ 0.00e+00]
  [ 2.00e+00]
  [ 0.00e+00]
  [ 3.00e+00]
  <BLANKLINE>

The above example also illustrates that when a *valued* affine expression ``exp`` is printed,
it is its value that is displayed. For a non-valued affine expression, **__repr__**
and **__str__** produce the same result, a string of the form ``'# (size)-affine expression: string-representation #'``.
Note that the constant affine expressions, as ``b[0]`` in the above example,
are always *valued*.
To assign a value to a non-constant :class:`AffinExp <picos.AffinExp>`,
you must set the :attr:`value <picos.Expression.value>` property of
every variable involved in the affine expression.


  >>> x_minus_1 = x - 1
  >>> x_minus_1                           #note that 1 was recognized as the (4x1)-vector with all ones
  # (4 x 1)-affine expression: x -|1| #
  >>> print x_minus_1
  # (4 x 1)-affine expression: x -|1| #
  >>> x_minus_1.is_valued()
  False
  >>> x.value = [0,1,2,-1]
  >>> x_minus_1.is_valued()
  True
  >>> print x_minus_1
  [-1.00e+00]
  [ 0.00e+00]
  [ 1.00e+00]
  [-2.00e+00]
  <BLANKLINE>

We also point out that :func:`new_param() <picos.tools.new_param>`
converts lists into vectors and lists of lists into matrices (given
in row major order).
In contrast, tuples are converted into list of affine expressions:

   >>> pic.new_param('vect',[1,2,3])                        # [1,2,3] is converted into a vector of dimension 3
   # (3 x 1)-affine expression: vect #
   >>> pic.new_param('mat',[[1,2,3],[4,5,6]])               # [[1,2,3],[4,5,6]] is converted into a (2x3)-matrix
   # (2 x 3)-affine expression: mat #
   >>> pic.new_param('list_of_scalars',(1,2,3))             # (1,2,3) is converted into a list of 3 scalar parameters #doctest: +NORMALIZE_WHITESPACE
   [# (1 x 1)-affine expression: list_of_scalars[0] #,
    # (1 x 1)-affine expression: list_of_scalars[1] #,
    # (1 x 1)-affine expression: list_of_scalars[2] #]
   >>> pic.new_param('list_of_vectors',([1,2,3],[4,5,6]))   # ([1,2,3],[4,5,6]) is converted into a list of 2 vector parameters #doctest: +NORMALIZE_WHITESPACE
   [# (3 x 1)-affine expression: list_of_vectors[0] #,
    # (3 x 1)-affine expression: list_of_vectors[1] #]

Overloaded operators
--------------------

.. _overloads:

OK, so now we have some variables (``t``, ``x``, ``w``, ``Y``, and ``Z``)
and some parameters (``A``, ``b``, ``D`` and ``alpha``). Let us create some
affine expressions with them.

   >>> A[0] * Z[0]                              #left multiplication
   # (2 x 2)-affine expression: A[0]*Z[0] #
   >>> Z[0] * A[0]                              #right multiplication
   # (4 x 4)-affine expression: Z[0]*A[0] #
   >>> A[1] * Z[0] * A[2]                       #left and right multiplication
   # (2 x 4)-affine expression: A[1]*Z[0]*A[2] #
   >>> alpha*Y                                  #scalar multiplication
   # (2 x 4)-affine expression: alpha*Y #
   >>> t/b[1][3] - D['Bob']                     #division by a scalar and substraction
   # (1 x 1)-affine expression: t / b[1][3] -D[Bob] #
   >>> ( b[2] | x )                             #dot product
   # (1 x 1)-affine expression: 〈 b[2] | x 〉 #
   >>> ( A[3] | Y )                             #generalized dot product for matrices: (A|B)=trace(A*B.T)
   # (1 x 1)-affine expression: 〈 A[3] | Y 〉 #
   >>> b[1]^x                                   #hadamard (element-wise) product
   # (4 x 1)-affine expression: b[1]∘x #

We can also take some subelements of affine expressions, by using
the standard syntax of python slices:

   >>> b[1][1:3]                                #2d and 3rd elements of b[1]
   # (2 x 1)-affine expression: b[1][1:3] #
   >>> Y[1,:]                                   #2d row of Y
   # (1 x 4)-affine expression: Y[1,:] #
   >>> x[-1]                                    #last element of x
   # (1 x 1)-affine expression: x[-1] #
   >>> A[2][:,1:3]*Y[:,-2::-2]                  #extended slicing with (negative) steps is allowed
   # (2 x 2)-affine expression: A[2][:,1:3]*( Y[:,-2::-2] ) #

In the last example, we keep only the second and third columns of ``A[2]``, and
the columns of ``Y`` with an even index, considered in the reverse order.
To concatenate affine expressions, the operators ``//`` and ``&`` have been
overloaded:

   >>> (b[1] & b[2] & x & A[0].T*A[0]*x) // x.T                  #vertical (//) and horizontal (&) concatenation
   # (5 x 4)-affine expression: [b[1],b[2],x,A[0].T*A[0]*x;x.T] #
   
When a scalar is added/substracted to a matrix or a vector, we interprete it
as an elementwise addition of the scalar to every element of the matrix or vector.

   >>> 5*x - alpha
   # (4 x 1)-affine expression: 5.0*x + |-alpha| #

.. Warning::
        Note that the string representation ``'|-alpha|'`` does not stand for the
        absolute value of ``-alpha``, but for the vector whose all terms are ``-alpha``.

Summing Affine Expressions
--------------------------

You can take the advantage of python syntax to create sums of affine expressions:

   >>> sum([A[i]*Z[i] for i in range(5)])
   # (2 x 2)-affine expression: A[0]*Z[0] + A[1]*Z[1] + A[2]*Z[2] + A[3]*Z[3] + A[4]*Z[4] #

This works, but you might have very long string representations if there are a lot
of summands. So you'd better use
the function :func:`picos.sum() <picos.tools.sum>`):

   >>> pic.sum([A[i]*Z[i] for i in range(5)],'i','[5]')
   # (2 x 2)-affine expression: Σ_{i in [5]} A[i]*Z[i] #

It is also possible to sum over several indices

   >>> pic.sum([A[i][1,j] + b[j].T*Z[i] for i in range(5) for j in range(4)],['i','j'],'[5]x[4]')
   # (1 x 2)-affine expression: Σ_{i,j in [5]x[4]} |A[i][1,j]| + b[j].T*Z[i] #

A more complicated example, given in two variants: in the first one,
``p`` is a tuple index representing a pair, while in the second case
we explicitely say that the pairs are of the form ``(p0,p1)``:

   >>> pic.sum([w[p]*b[p[1]-1][p[0]] for p in pairs],('p',2),'pairs')
   # (1 x 1)-affine expression: Σ_{p in pairs} w[p]*b[p__1-1][p__0] #
   >>> pic.sum([w[p0,p1]*b[p1-1][p0] for (p0,p1) in pairs],['p0','p1'],'pairs')
   # (1 x 1)-affine expression: Σ_{p0,p1 in pairs} w[(p0, p1)]*b[p1-1][p0] #

It is also possible to sum over string indices (*see the documentation of* :func:`sum() <picos.tools.sum>`):

   >>> pic.sum([D[name] for name in D],'name','people_list')
   # (1 x 1)-affine expression: Σ_{name in people_list} D[name] #


Objective function
------------------

The objective function of the problem
can be defined with the function
:func:`set_objective() <picos.Problem.set_objective>`.
Its first argument should be ``'max'``, ``'min'`` or
``'find'`` (*for feasibility problems*),
and the second argument should be a scalar expression:

    >>> prob.set_objective('max',( A[0] | Y )-t)
    >>> print prob  #doctest: +NORMALIZE_WHITESPACE
    ---------------------
    optimization problem (MIP):
    59 variables, 0 affine constraints
    <BLANKLINE>
    w   : dict of 6 variables, (1, 1), binary
    Z   : list of 5 variables, (4, 2), continuous
    t   : (1, 1), continuous
    Y   : (2, 4), continuous
    x   : (4, 1), integer
    <BLANKLINE>
        maximize 〈 A[0] | Y 〉 -t
    such that
      []
    ---------------------

With this example, you see what happens when a problem is printed:
the list of optimization variables is displayed, then the objective function
and finally a list of constraints (in the case above, there is no constraint).

==============================
*Norm of an affine Expression*
==============================

The norm of an affine expression is an overload of the ``abs()`` function.
If ``x`` is an affine expression, ``abs(x)`` is its Euclidean norm :math:`\sqrt{x^T x}`.

  >>> abs(x)
  # norm of a (4 x 1)- expression: ||x|| #
  
In the case where the affine expression is a matrix, ``abs()`` returns its
Frobenius norm, defined as :math:`\Vert M \Vert_F := \sqrt{\operatorname{trace} (M^T M)}`.

  >>> abs(Z[1]-2*A[0].T)
  # norm of a (4 x 2)- expression: ||Z[1] -2.0*A[0].T|| #

Note that the absolute value of a scalar expression is stored as a norm:

  >>> abs(t)
  # norm of a (1 x 1)- expression: ||t|| #

However, a scalar constraint of the form :math:`|a^T x + b| \leq c^T x + d`
is handled as two linear constraints by PICOS, and so a problem with the latter
constraint
can be solved even if you do not have a SOCP solver available.
Besides, note that the string representation of an absolute value uses the double bar notation.
(Recall that the single bar notation ``|t|`` is used to denote the vector
whose all values are ``t``).

It is also possible to use other :math:`L_p-` norms in picos, cf. :ref:`this paragraph <pnorms>` .


=======================
*Quadratic Expressions*
=======================

Quadratic expressions can be formed in several ways:

  >>> t**2 - x[1]*x[2] + 2*t - alpha                       #sum of linear and quadratic terms
  #quadratic expression: t**2 -x[1]*x[2] + 2.0*t -alpha #
  >>> (x[1]-2) * (t+4)                                     #product of two affine expressions
  #quadratic expression: ( x[1] -2.0 )*( t + 4.0 ) #
  >>> Y[0,:]*x                                             #Row vector multiplied by column vector
  #quadratic expression: Y[0,:]*x #
  >>> (x +2 | Z[1][:,1])                                   #scalar product of affine expressions
  #quadratic expression: 〈 x + |2.0| | Z[1][:,1] 〉 #
  >>> abs(x)**2                                            #recall that abs(x) is the euclidean norm of x
  #quadratic expression: ||x||**2 #
  >>> (t & alpha) * A[1] * x                               #quadratic form
  #quadratic expression: [t,alpha]*A[1]*x #

It is not possible (yet) to make a multidimensional quadratic expression.

=============
*Constraints*
=============

A constraint takes the form of two expressions separated by a relation operator.

Linear (in)equalities
---------------------

Linear (in)equalities are understood elementwise. **The strict operators**
``<`` **and** ``>`` **denote weak inequalities** (*less or equal than*
and *larger or equal than*). For example:

   >>> (1|x) < 2                                                        #sum of the x[i] less or equal than 2
   # (1x1)-affine constraint: 〈 |1| | x 〉 < 2.0 #
   >>> Z[0] * A[0] > b[1]*b[2].T                                        #A 4x4-elementwise inequality
   # (4x4)-affine constraint: Z[0]*A[0] > b[1]*b[2].T #
   >>> pic.sum([A[i]*Z[i] for i in range(5)],'i','[5]') == 0            #A 2x2 equality. The RHS is the all-zero matrix
   # (2x2)-affine constraint: Σ_{i in [5]} A[i]*Z[i] = |0| #

Constraints can be added in the problem with the function
:func:`add_constraint() <picos.Problem.add_constraint>`:

  >>> for i in range(1,5):
  ...      prob.add_constraint(Z[i]==Z[i-1]+Y.T)
  >>> print prob        #doctest: +NORMALIZE_WHITESPACE
  ---------------------
  optimization problem (MIP):
  59 variables, 32 affine constraints
  <BLANKLINE>
  w   : dict of 6 variables, (1, 1), binary
  Z   : list of 5 variables, (4, 2), continuous
  t   : (1, 1), continuous
  Y   : (2, 4), continuous
  x   : (4, 1), integer
  <BLANKLINE>
      maximize 〈 A[0] | Y 〉 -t
  such that
    Z[1] = Z[0] + Y.T
    Z[2] = Z[1] + Y.T
    Z[3] = Z[2] + Y.T
    Z[4] = Z[3] + Y.T
  ---------------------

The constraints of the problem can then be accessed with the function
:func:`get_constraint() <picos.Problem.get_constraint>`:

  >>> prob.get_constraint(2)                      #constraints are numbered from 0
  # (4x2)-affine constraint: Z[3] = Z[2] + Y.T #

An alternative is to pass the constraint with the option ``ret = True``,
which has the effect to return a reference to the constraint you want to add.
In particular, this reference can be useful to access the optimal dual variable
of the constraint, once the problem will have been solved.

  >>> mycons = prob.add_constraint(Z[4]+Z[0] == Y.T, ret = True)
  >>> print mycons
  # (4x2)-affine constraint : Z[4] + Z[0] = Y.T #

Groupping constraints
---------------------

In order to have a more compact string representation of the problem,
it is advised to use the function :func:`add_list_of_constraints() <picos.Problem.add_list_of_constraints()>`,
which works similarly as the function :func:`sum() <picos.tools.sum>`.

    >>> prob.remove_all_constraints()                                                    #we first remove the 4 constraints precedently added
    >>> prob.add_constraint(Y>0)                                                         #a single constraint
    >>> prob.add_list_of_constraints([Z[i]==Z[i-1]+Y.T for i in range(1,5)],'i','1...4') #the same list of constraints as above
    >>> print prob    #doctest: +NORMALIZE_WHITESPACE
    ---------------------
    optimization problem (MIP):
    59 variables, 40 affine constraints
    <BLANKLINE>
    w   : dict of 6 variables, (1, 1), binary
    Z   : list of 5 variables, (4, 2), continuous
    t   : (1, 1), continuous
    Y   : (2, 4), continuous
    x   : (4, 1), integer
    <BLANKLINE>
        maximize 〈 A[0] | Y 〉 -t
    such that
      Y > |0|
      Z[i] = Z[i-1] + Y.T for all i in 1...4
    ---------------------

Now, the constraint ``Z[3] = Z[2] + Y.T``, which has been entered
in 4th position, can either be accessed by  ``prob.get_constraint(3)`` (``3`` because
constraints are numbered from ``0``), or by

  >>> prob.get_constraint((1,2))
  # (4x2)-affine constraint: Z[3] = Z[2] + Y.T #

where ``(1,2)`` means *the 3rd constraint of the 2d group of constraints*,
with zero-based numbering.

Similarly, the constraint ``Y > |0|`` can be accessed by
``prob.get_constraint(0)`` (first constraint),
``prob.get_constraint((0,0))`` (first constraint of the first group), or
``prob.get_constraint((0,))`` (unique constraint of the first group).

.. _delcons:

Removing constraints
--------------------

It can be useful to remove some constraints, especially for
dynamic approaches such as column generation. Re-creating an instance from
scratch after each iteration would be inefficient. Instead, PICOS allows
one to modify the solver instance and to re-solve it on the fly
(for ``mosek``, ``cplex`` and ``gurobi``). It is not possible to change
directly a constraint, but you can delete a constraint from model, and
then re-add the modified constraint.

To delete a constraint, you must have a handle to this constraint. To this end,
you can pass the option ``return constraints`` when you create the instance of the problem.
The next code shows an example in which a variable ``x2`` is added to the model,
which appears as ``+3*x2`` in the objective function and as ``+x2`` in the LHS of a constraint.

  >>> prb = pic.Problem(return_constraints=True)
  >>> x1 = prb.add_variable('x1',1)
  >>> lhs = 2*x1
  >>> obj = 5*x1
  >>> cons = prb.add_constraint(lhs <= 1)
  >>> sol = prb.maximize(obj,verbose=0)
  >>> #--------------------------------------
  >>> #at this place, the user can use his favorite method to solve the 'pricing problem'.
  >>> #Let us assume that this phase suggests to add a new variable 'x2' in the model,
  >>> #which appears as `+3*x2` in the objective function, and as `+x2` in the LHS of the constraint.
  >>> x2 = prb.add_variable('x2',1)
  >>> lhs += x2
  >>> obj += 3*x2
  >>> cons.delete()
  >>> newcons = prb.add_constraint(lhs <= 1)
  >>> print prb   #doctest: +NORMALIZE_WHITESPACE
  ---------------------
  optimization problem  (LP):
  2 variables, 1 affine constraints
  <BLANKLINE>
  x2  : (1, 1), continuous
  x1  : (1, 1), continuous
  <BLANKLINE>
      maximize 5.0*x1 + 3.0*x2
  such that
    2.0*x1 + x2 < 1.0
  ---------------------

.. _flowcons:

Flow constraints in Graphs
--------------------------

Flow constraints in graphs are entered using a Networkx_ Graph. The following example finds a (trivial) maximal flow from ``'S'`` to ``'T'`` in ``G``.

.. _Networkx: https://networkx.github.io/

  >>> import networkx as nx
  >>> G = nx.DiGraph()
  >>> G.add_edge('S','A', capacity=1); G.add_edge('A','B', capacity=1); G.add_edge('B','T', capacity=1)
  >>> pb = pic.Problem()
  >>> # Adding the flow variables
  >>> f={}
  >>> for e in G.edges():
  ...      f[e]=pb.add_variable('f[{0}]'.format(e),1)
  >>> # A variable for the value of the flow
  >>> F = pb.add_variable('F',1)
  >>> # Creating the flow constraint
  >>> flowCons = pic.flow_Constraint(G, f, source='S', sink='T', capacity='capacity', flow_value= F, graphName='G')
  >>> pb.addConstraint(flowCons)
  >>> pb.set_objective('max',F)
  >>> sol = pb.solve(verbose=0)
  >>> flow = pic.tools.eval_dict(f)



Picos allows you to define single source - multiple sinks problems.
You can use the same syntax as for a single source - single sink problems.
Just add a list of sinks and a list of flows instead.

.. warning::
        The function :func:`flow_Constraint() <picos.tools.flow_Constraint>`
        cannot take both multiple sources and multiple sinks.
        Multicommodity flows will be supported in the next release.

.. testcode::

        import picos as pic
        import networkx as nx

        G=nx.DiGraph()
        G.add_edge('S','A', capacity=2); G.add_edge('S','B', capacity=2)
        G.add_edge('A','T1', capacity=2); G.add_edge('B','T2', capacity=2)

        pbMultipleSinks=pic.Problem()
        # Flow variable
        f={}
        for e in G.edges():
                  f[e]=pbMultipleSinks.add_variable('f[{0}]'.format(e),1)

        # Flow value
        F1=pbMultipleSinks.add_variable('F1',1)
        F2=pbMultipleSinks.add_variable('F2',1)

        flowCons = pic.flow_Constraint(G, f, source='S', sink=['T1','T2'], capacity='capacity', flow_value=[F1, F2], graphName='G')
        
        pbMultipleSinks.add_constraint(flowCons)
        pbMultipleSinks.set_objective('max',F1+F2)

        # Solve the problem
        pbMultipleSinks.solve(verbose=0)

        print pbMultipleSinks
        print 'The optimal flow F1 has value {0}'.format(F1)
        print 'The optimal flow F2 has value {0}'.format(F2)


.. testoutput::
        :options: +NORMALIZE_WHITESPACE
        
        ---------------------
        optimization problem  (LP):
        6 variables, 12 affine constraints

        f       : dict of 4 variables, (1, 1), continuous
        F1      : (1, 1), continuous
        F2      : (1, 1), continuous

                  maximize F1 + F2
        such that
          ** One Source, Multiple Sinks ** 
          Flow conservation in G from S to T1 with value F1
          Flow conservation in G from S to T2 with value F2

        ---------------------
        The optimal flow F1 has value 2.0
        The optimal flow F2 has value 2.0


A similar syntax can be used for multiple sources-single sink flows.

..
        .. testcode::

                import picos as pic
                import networkx as nx

                G=nx.DiGraph()
                G.add_edge('S1','A', capacity=1); G.add_edge('S2','B', capacity=2)
                G.add_edge('A','T', capacity=2); G.add_edge('B','T', capacity=2)

                pbMultipleSources=pic.Problem()
                # Flow variable
                f={}
                for e in G.edges():
                        f[e]=pbMultipleSources.add_variable('f[{0}]'.format(e),1)

                # Flow value
                F1=pbMultipleSources.add_variable('F1',1)
                F2=pbMultipleSources.add_variable('F2',1)

                flowCons = pic.flow_Constraint(G, f, source=['S1', 'S2'], sink='T', capacity='capacity', flowValue=[F1, F2], graphName='G')
                pbMultipleSources.addConstraint(flowCons)

                pbMultipleSources.set_objective('max',F1+F2)

                # Solve the problem
                pbMultipleSources.solve(verbose=0)

                print pbMultipleSources
                print 'The optimal flow F1 has value {0}'.format(F1)
                print 'The optimal flow F2 has value {0}'.format(F2)

        .. testoutput::
                :options: +NORMALIZE_WHITESPACE
                
                ---------------------
                optimization problem  (LP):
                6 variables, 13 affine constraints

                F1      : (1, 1), continuous
                F2      : (1, 1), continuous
                f       : dict of 4 variables, (1, 1), continuous

                        maximize F1 + F2
                such that
                ** Multiple Sources, One Sink **
                Flow conservation in G from S1 to T with value F1
                Flow conservation in G from S2 to T with value F2

                ---------------------
                The optimal flow F1 has value 1.0
                The optimal flow F2 has value 2.0



Quadratic constraints
---------------------

Quadratic inequalities are entered in the following way:

  >>> t**2 > 2*t - alpha + x[1]*x[2]
  #Quadratic constraint -t**2 + 2.0*t -alpha + x[1]*x[2] < 0 #
  >>> (t & alpha) * A[1] * x + (x +2 | Z[1][:,1]) < 3*(1|Y)-alpha
  #Quadratic constraint [t,alpha]*A[1]*x + 〈 x + |2.0| | Z[1][:,1] 〉 -(3.0*〈 |1| | Y 〉 -alpha) < 0 #

Note that PICOS does not check the convexity of convex constraints.
The solver will raise an Exception if it does not support
non-convex quadratics.


Second Order Cone Constraints
-----------------------------

There are two types of second order cone constraints supported in PICOS.

  * The constraints of the type :math:`\Vert x \Vert \leq t`, where :math:`t`
    is a scalar affine expression and :math:`x` is
    a multidimensional affine expression (possibly a matrix, in which case the
    norm is Frobenius). This inequality forces
    the vector :math:`[x;t]` to belong to a Lorrentz-Cone (also called
    *ice-cream cone*)
  * The constraints of the type :math:`\Vert x \Vert^2 \leq t u,\ t \geq 0`, where
    :math:`t` and :math:`u` are scalar affine expressions and
    :math:`x` is a multidimensional affine expression, which constrain
    the vector :math:`[x,t,u]` inside a rotated version of the Lorretz cone.
    When a constraint of the form ``abs(x)**2 < t*u`` is passed to PICOS, **it
    is implicitely assumed that** ``t`` **is nonnegative**, and the constraint is
    handled as the equivalent, standard ice-cream cone constraint
    :math:`\Vert \ [2x,t-u]\  \Vert \leq t+u`.
    
A few examples:

  >>> abs(x) < (2|x-1)                                                                  #A simple ice-cream cone constraint
  # (4x1)-SOC constraint: ||x|| < 〈 |2.0| | x -|1| 〉 #
  >>> abs(Y+Z[0].T) < t+alpha                                                           #SOC constraint with Frobenius norm
  # (2x4)-SOC constraint: ||Y + Z[0].T|| < t + alpha #
  >>> abs(Z[1][:,0])**2 < (2*t-alpha)*(x[2]-x[-1])                                      #Rotated SOC constraint
  # (4x1)-Rotated SOC constraint: ||Z[1][:,0]||^2 < ( 2.0*t -alpha)( x[2] -(x[-1])) #
  >>> t**2 < D['Elisa']+t                                                               #t**2 is understood as the squared norm of [t]
  # (1x1)-Rotated SOC constraint: ||t||^2 < D[Elisa] + t #
  >>> 1 < (t-1)*(x[2]+x[3])                                                             #1 is understood as the squared norm of [1]
  # (1x1)-Rotated SOC constraint: 1.0 < ( t -1.0)( x[2] + x[3]) #

Semidefinite Constraints
-------------------------

Linear matrix inequalities (LMI) can be entered thanks to an overload of the operators
``<<`` and ``>>``. For example, the LMI

.. math::
   :nowrap:

   \begin{equation*}
   \sum_{i=0}^3 x_i b_i b_i^T \succeq b_4 b_4^T,
   \end{equation*}

where :math:`\succeq` is used to denote the Löwner ordering,
is passed to PICOS by writing:

  >>> pic.sum([x[i]*b[i]*b[i].T for i in range(4)],'i','0...3') >> b[4]*b[4].T
  # (4x4)-LMI constraint Σ_{i in 0...3} x[i]*b[i]*b[i].T ≽ b[4]*b[4].T #

Note the difference with

  >>> pic.sum([x[i]*b[i]*b[i].T for i in range(4)],'i','0...3') > b[4]*b[4].T
  # (4x4)-affine constraint: Σ_{i in 0...3} x[i]*b[i]*b[i].T > b[4]*b[4].T #

which yields an elementwise inequality.


For convenience, it is possible to add a symmetric matrix variable ``X``,
by specifying the option ``vtype=symmetric``. This has the effect to
store all the affine expressions which depend on ``X`` as a function
of its lower triangular elements only.

    >>> sdp = pic.Problem()
    >>> X = sdp.add_variable('X',(4,4),vtype='symmetric')
    >>> sdp.add_constraint(X >> 0)
    >>> print sdp   #doctest: +NORMALIZE_WHITESPACE
    ---------------------
    optimization problem (SDP):
    10 variables, 0 affine constraints, 10 vars in 1 SD cones
    <BLANKLINE>
    X   : (4, 4), symmetric
    <BLANKLINE>
        find vars
    such that
      X ≽ |0|
    ---------------------

In this example, you see indeed that the problem has 10=(4*5)/2 variables,
which correspond to the lower triangular elements of ``X``.

.. Warning::
     When a constraint of the form ``A >> B`` is passed to PICOS, it is not
     assumed that A-B is symmetric. Instead, the symmetric matrix whose lower
     triangular elements are those of A-B is forced to be positive semidefnite.
     So, in the cases where A-B is not implicitely forced to be symmetric, you
     should add a constraint of the form ``A-B==(A-B).T`` in the problem.

Inequalities involving geometric means
--------------------------------------

It is possible to enter an inequality of the form

.. math::
   t \leq \prod_{i=1}^n x_i^{1/n}

in PICOS, where :math:`t`
is a scalar affine expression and :math:`x` is an affine expression
of dimension :math:`n` (possibly a matrix, in which case
:math:`x_i` is counted in column major order).
This inequality is internally converted to an equivalent set of
second order cone inequalities, by using standard techniques
(cf. e.g. :ref:`[1] <tuto_refs>`).

Many convex constraints can be formulated using inequalities that involve
a geometric mean. For example, :math:`t \leq x_1^{2/3}` is equivalent
to :math:`t \leq t^{1/4} x_1^{1/4} x_1^{1/4}`, which can be entered in PICOS
thanks to the function :func:`picos.geomean() <picos.tools.geomean>` :

  >>> t < pic.geomean(t //x[1] //x[1] //1)
  # geometric mean ineq : t<geomean( [t;x[1];x[1];1.0])#

Note that the latter example can also be passed to picos in a more simple way,
thanks to an overloading of the ``**`` exponentiation operator:

  >>> t < x[1]**(2./3)
  # pth power ineq : ( x[1])**2/3>t#

Inequalities involving geometric means are stored in a temporary object
of the class :class:`GeoMeanConstraint <picos.GeoMeanConstraint>`,
which can be passed to a problem with :func:`add_constraint() <picos.Problem.add_constraint>`:

  >>> geom_ineq = prob.add_constraint(t<pic.geomean(Y[:6]), ret=True)

When the option ``ret = True`` is used to pass an inequality with a geometric mean,
the object of the class :class:`GeoMeanConstraint <picos.GeoMeanConstraint>` is returned.
This object has an attribute ``Ptmp`` which contains all the SOC inequalities that
are used internally to represent the geometric mean:

  >>> geom_ineq.Ptmp.constraints  #doctest: +NORMALIZE_WHITESPACE
  [# (1x1)-Rotated SOC constraint: ||u[1:0-1]||^2 < ( Y[:6][0])( Y[:6][1]) #,
   # (1x1)-Rotated SOC constraint: ||u[1:2-3]||^2 < ( Y[:6][2])( Y[:6][3]) #,
   # (1x1)-Rotated SOC constraint: ||u[1:4-5]||^2 < ( Y[:6][4])( Y[:6][5]) #,
   # (1x1)-Rotated SOC constraint: ||u[2:0-3]||^2 < ( u[1:0-1])( u[1:2-3]) #,
   # (1x1)-Rotated SOC constraint: ||u[2:4-x]||^2 < ( u[1:4-5])( t) #,
   # (1x1)-Rotated SOC constraint: ||t||^2 < ( u[2:0-3])( u[2:4-x]) #]


Inequalities involving real powers or trace of matrix powers
------------------------------------------------------------

As mentionned above, the ``**`` exponentiation operator has been overloaded
to support real exponents. A rational approximation of the exponent is used,
and the inequality are internally reformulated as a set of equivalent SOC inequalities.
Note that only inequalities defining a convex regions can be passed:

   >>> t**0.6666 > x[0]
   # pth power ineq : ( t)**2/3>x[0]#
   >>> t**-0.5 < x[0]
   # pth power ineq : ( t)**-1/2<x[0]#
   >>> try:
   ...      t**-0.5 > x[0]
   ... except Exception as ex:
   ...      print 'Exception: '+str(ex) #doctest: +NORMALIZE_WHITESPACE
   Exception: >= operator can be used only when the function is concave (0<p<=1)
   >>> t**2 < x[1]+x[2]   
   # (1x1)-Rotated SOC constraint: ||t||^2 < x[1] + x[2] #
   
More generally, inequalities involving trace of matrix powers can be passed to PICOS,
by using the :func:`picos.tracepow() <picos.tools.tracepow>` function. The following example
creates the constraint
   
   .. math::
        
        \operatorname{trace}\ \big(x_0 A_0 A_0^T + x_2 A_2 A_2^T\big)^{2.5} \leq 3.   
        


>>> pic.tracepow(x[0] * A[0]*A[0].T + x[2] * A[2]*A[2].T, 2.5) <= 3
# trace of pth power ineq : trace( x[0]*A[0]*A[0].T + x[2]*A[2]*A[2].T)**5/2<3.0#
   
   .. Warning::
   
        when a power expression :math:`x^p` (resp. the trace of matrix power :math:`\operatorname{trace}\ X^p` )
        is used, the base :math:`x` is forced to be nonnegative (resp. the base :math:`X` is
        forced to be positive semidefinite) by picos.
        
When the exponent is :math:`0<p<1`, 
it is also possible to represent constraints of the form
:math:`\operatorname{trace}(M X^p) \geq t`
with SDPs, where :math:`M\succeq 0`, see :ref:`[2] <tuto_refs>`.

>>> pic.tracepow(X, 0.6666, coef = A[0].T*A[0]) >= t
# trace of pth power ineq : trace[ A[0].T*A[0] *(X)**2/3]>t#

As for geometric means, inequalities involving real powers are 
stored in a temporary object of the class :class:`TracePow_Constraint <picos.TracePow_Constraint>`,
which contains a field ``Ptmp`` , a Problem instance with all the SOC or SDP constraints
used to represent the original inequality.

.. _pnorms:

Inequalities involving generalized p-norm
-----------------------------------------

Inequalities of the form :math:`\Vert x \Vert_p \leq t` can be entered by using the
function :func:`picos.norm() <picos.tools.norm>`. This function is also defined for :math:`p < 1`
by the usual formula :math:`\Vert x \Vert_p :=  \Big(\sum_i |x_i|^p \Big)^{1/p}`.
The norm function is convex over :math:`\mathbb{R}^n` for all :math:`p\geq 1`, and
concave over the set of vectors with nonnegative coordinates for :math:`p \leq 1`.

>>> pic.norm(x,3) < t
# p-norm ineq : norm_3( x)<t#
>>> pic.norm(x,'inf') < 2
# p-norm ineq : norm_inf( x)<2.0#
>>> pic.norm(x,0.5) > x[0]-x[1]
# generalized p-norm ineq : norm_1/2( x)>x[0] -x[1]#

.. Warning::

        Note that when a constraint of the form ``norm(x,p) >= t`` is entered (with :math:`p \leq 1` ),
        PICOS forces the vector ``x`` to be nonnegative (componentwise).

Inequalities involving the generalized :math:`L_{p,q}` norm of
a matrix can also be handled with picos, cf. the documentation of
:func:`picos.norm() <picos.tools.norm>` .
        
As for geometric means, inequalities involving p-norms are 
stored in a temporary object of the class :class:`NormP_Constraint <picos.NormP_Constraint>`,
which contains a field ``Ptmp`` , a Problem instance with all the SOC constraints
used to represent the original inequality.
        
Inequalities involving the nth root of a determinant
----------------------------------------------------

The function :func:`picos.detrootn() <picos.tools.detrootn>`
can be used to enter the :math:`n` th root of the determinant of a
:math:`(n \times n)-`symmetric positive semidefinite matrix:

>>> M = sdp.add_variable('M',(5,5),'symmetric')
>>> t < pic.detrootn(M)
# nth root of det ineq : det( M)**1/5>t#

.. Warning::

        Note that when a constraint of the form ``t < pic.detrootn(M)`` is entered (with :math:`p \leq 1` ),
        PICOS forces the matrix ``M`` to be positive semidefinite.
        
As for geometric means, inequalities involving the nth root of a determinant are 
stored in a temporary object of the class :class:`DetRootN_Constraint <picos.DetRootN_Constraint>`,
which contains a field ``Ptmp`` , a Problem instance with all the SOC and SDP constraints
used to represent the original inequality.

================
*Set membership*
================

Since Picos 1.0.2, there is a :class:`Set <picos.Set>` class
that can be used to pass constraints as membership of an affine expression to a set.

Following sets are currently supported:

 * :math:`L_p-` balls representing the set :math:`\{x: \Vert x \Vert_p \leq r \}`  
   can be constructed with the function :func:`pic.ball() <picos.tools.ball>`
 * The standard simplex (scaled by a factor :math:`\gamma` ) :math:`\{x \geq 0: \sum_i x_i \leq r \}`
   can be constructed with the function :func:`pic.simplex() <picos.tools.simplex>`
 * Truncated simplexes :math:`\{0 \leq x \leq 1: \sum_i x_i \leq r \}`
   and symmetrized Truncated simplexes :math:`\{x: \Vert x \Vert_\infty \leq 1, \Vert x \Vert_1\leq r \}`
   can be constructed with the function :func:`pic.truncated_simplex() <picos.tools.truncated_simplex>`
 
Membership of an affine expression to a set can be expressed with the overloaded operator ``<<``.
This returns a temporary object that can be passed to a picos problem with the function
:func:`add_constraint() <picos.Problem.add_constraint>` .

>>> x << pic.simplex(1)
# (5x1)-affine constraint: x in standard simplex #
>>> x << pic.truncated_simplex(2)
# (9x1)-affine constraint: x in truncated simplex of radius 2 #
>>> x << pic.truncated_simplex(2,sym=True)
# symmetrized truncated simplex constraint : ||x||_{infty;1} <= {1;2}#
>>> x << pic.ball(3)
# (4x1)-SOC constraint: ||x|| < 3.0 #
>>> pic.ball(2,'inf') >> x
# p-norm ineq : norm_inf( x)<2.0#
>>> x << pic.ball(4,1.5)
# p-norm ineq : norm_3/2( x)<4.0#
        
        
===========================
*Write a Problem to a file*
===========================

It is possible to write a problem to a file, thanks to the
function :func:`write_to_file() <picos.Problem.write_to_file>`.
Several file formats and file writers are available, have a look at the doc
of :func:`write_to_file() <picos.Problem.write_to_file>` for more explanations.

Below is a *hello world* example, which writes a simple MIP to a **.lp** file:


.. testcode::
        
        import picos as pic
        prob = pic.Problem(pass_simple_cons_as_bound = True)
        #with this option, x>1.5 is recognized as a variable bound
        #instead of a constraint.
        y = prob.add_variable('y',1, vtype='integer')
        x = prob.add_variable('x',1)
        prob.add_constraint(x>1.5)
        prob.add_constraint(y-x>0.7)
        prob.set_objective('min',y)
        #let first picos display the problem
        print prob
        print
        #now write the problem to a .lp file...
        prob.write_to_file('helloworld.lp')
        print 
        #and display the content of the freshly created file:
        print open('helloworld.lp').read()

Generated output:

.. testoutput::
        :options: +NORMALIZE_WHITESPACE
        
        ---------------------
        optimization problem  (MIP):
        2 variables, 2 affine constraints
        
        y   : (1, 1), integer
        x   : (1, 1), continuous
        
                minimize y
        such that
        x > 1.5
        y -x > 0.7
        ---------------------
        
        writing problem in helloworld.lp...
        done.
        
        \* file helloworld.lp generated by picos*\
        Minimize
        obj : 1 y
        Subject To
        in0 : -1 y+ 1 x <= -0.7
        Bounds
        y free
        1.5 <= x<= +inf
        Generals
        y
        Binaries
        End

        

.. testcleanup::
        
        import os
        os.system('rm -f helloworld.lp')

=================
*Solve a Problem*
=================

To solve a problem, you have to use the method :func:`solve() <picos.Problem.solve>`
of the class :class:`Problem <picos.Problem>`. 
Alternatively, the functions :func:`maximize(obj) <picos.Problem.maximize>`
and :func:`minimize(obj) <picos.Problem.minimize>`
can be used to specify the objective function and call the solver in a single statement. 
These method accept several options. In particular the solver can be specified by passing 
an option of the form ``solver='solver_name'``. For a list of available
parameters with their default values, see the doc of the function
:func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`.

Once a problem has been solved, the optimal values of the variables are
accessible with the :attr:`value <picos.Expression.value>` property.
Depending on the solver, you
can also obtain the slack and the optimal dual variables
of the constraints thanks to the properties
:attr:`dual<picos.Constraint.dual>` and
:attr:`slack<picos.Constraint.slack>` of the class
:class:`Constraint <picos.Constraint>`.
See the doc of :attr:`dual<picos.Constraint.dual>` for more explanations
on the dual variables for second order cone programs (SOCP) and
semidefinite programs (SDP).

The class :class:`Problem <picos.Constraint>` also has
two interesting properties: :attr:`type <picos.Problem.type>`, which
indicates the class of the optimization problem ('LP', 'SOCP', 'MIP', 'SDP',...),
and :attr:`status <picos.Problem.status>`, which indicates if the
problem has been solved (the default is ``'unsolved'``; after a call to
:func:`solve() <picos.Problem.solve>` this property can take the value of any
code returned by a solver, such as ``'optimal'``, ``'unbounded'``, ``'near-optimal'``,
``'primal infeasible'``, ``'unknown'``, ...).


Below is a simple example, to solve the linear programm:

.. math::
   :nowrap:   

   \begin{center}
   $\begin{array}{ccc}
   \underset{x \in \mathbb{R}^2}{\mbox{minimize}}
                      & 0.5 x_1 + x_2 &\\
   \mbox{subject to} & x_1 &\geq x_2\\
                     & \left[
                        \begin{array}{cc}
                        1 & 0\\
                        1 & 1
                        \end{array}
                        \right] x &\leq 
                        \left[
                        \begin{array}{c} 3 \\4 \end{array}
                        \right].
   \end{array}$
   \end{center}

More examples can be found :ref:`here <examples>`.

.. testcode::

   P = pic.Problem()
   A = pic.new_param('A', cvx.matrix([[1,1],[0,1]]) )
   x = P.add_variable('x',2)
   P.add_constraint(x[0]>x[1])
   P.add_constraint(A*x<[3,4])
   objective = 0.5 * x[0] + x[1]
   P.set_objective('max', objective) #or directly P.maximize(objective)
   
   #display the problem and solve it
   print P
   print 'type:   '+P.type
   print 'status: '+P.status
   P.solve(solver='cvxopt',verbose=False)
   print 'status: '+P.status
   
   #--------------------#
   #  objective value   #
   #--------------------#
 
   print 'the optimal value of this problem is:'
   print P.obj_value()                      #"print objective" would also work, because objective is valued

   #--------------------#
   #  optimal variable  #
   #--------------------#
   x_opt = x.value
   print 'The solution of the problem is:'
   print x_opt                              #"print x" would also work, since x is now valued
   print

   #--------------------#
   #  slacks and duals  #
   #--------------------#
   c0=P.get_constraint(0)
   print 'The dual of the constraint'
   print c0
   print 'is:'
   print c0.dual
   print 'And its slack is:'
   print c0.slack
   print

   c1=P.get_constraint(1)
   print 'The dual of the constraint'
   print c1
   print 'is:'
   print c1.dual
   print 'And its slack is:'
   print c1.slack

.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    ---------------------
    optimization problem  (LP):
    2 variables, 3 affine constraints

    x   : (2, 1), continuous

        maximize 0.5*x[0] + x[1]
    such that
      x[0] > x[1]
      A*x < [ 2 x 1 MAT ]
    ---------------------
    type:   LP
    status: unsolved
    status: optimal

    the optimal value of this problem is:
    3.0000000002
    The solution of the problem is:
    [ 2.00e+00]
    [ 2.00e+00]


    The dual of the constraint
    # (1x1)-affine constraint : x[0] > x[1] #
    is:
    [ 2.50e-01]

    And its slack is:
    [ 1.83e-09]


    The dual of the constraint
    # (2x1)-affine constraint : A*x < [ 2 x 1 MAT ] #
    is:
    [ 4.56e-10]
    [ 7.50e-01]

    And its slack is:
    [ 1.00e+00]
    [-8.71e-10]


.. _noteduals:

A note on dual variables
------------------------

For second order cone constraints of the form :math:`\Vert \mathbf{x} \Vert \leq t`,
where :math:`\mathbf{x}` is a vector of dimension :math:`n`,
the dual variable is a vector of dimension :math:`n+1` of the form
:math:`[\lambda; \mathbf{z}]`, where the :math:`n-` dimensional vector
:math:`\mathbf{z}` satisfies :math:`\Vert \mathbf{z} \Vert \leq \lambda`.

Since *rotated* second order cone constraints of the form
:math:`\Vert \mathbf{x} \Vert^2 \leq t u,\ t \geq 0`,
are handled as the equivalent ice-cream constraint
:math:`\Vert [2 \mathbf{x}; t-u ] \Vert \leq t+u`,
the dual is given with respect to this reformulated, standard SOC constraint.

In general, a linear problem with second order cone constraints (both standard and rotated)
and semidefinite constraints can be written under the form:

.. math::
   :nowrap:   

   \begin{center}
   $\begin{array}{cclc}
   \underset{\mathbf{x} \in \mathbb{R}^n}{\mbox{minimize}}
                      & \mathbf{c}^T \mathbf{x} & &\\
   \mbox{subject to} & A^e \mathbf{x} + \mathbf{b^e} & = 0 &\\
                     & A^l \mathbf{x} + \mathbf{b^l} & \leq 0 &\\
                     & \Vert A^s_i \mathbf{x} + \mathbf{b^s_i} \Vert &\leq \mathbf{f^s_i}^T \mathbf{x} +d^s_i, & \forall i \in I\\
                     & \Vert A^r_j \mathbf{x} + \mathbf{b^r_j} \Vert^2 &\leq (\mathbf{f^{r_1}_j}^T \mathbf{x} +d^{r_1}_j) (\mathbf{f^{r_2}_j}^T \mathbf{x} +d^{r_2}_j), & \forall j \in J\\
                     & 0        & \leq \mathbf{f^{r_1}_j}^T \mathbf{x} +d^{r_1}_j, & \forall j \in J\\
                     & \sum_{i=1}^n x_i M_i & \succeq M_0
   \end{array}$
   \end{center}

where

        * :math:`\mathbf{c}, \big\{\mathbf{f^s_i}\big\}_{i\in I}, \big\{\mathbf{f^{r_1}_j}\big\}_{j \in J}, \big\{\mathbf{f^{r_2}_j}\big\}_{j \in J}`
          are vectors of dimension :math:`n`;

        * :math:`\big\{d^s_i\big\}_{i \in I}, \big\{d^{r_1}_j\big\}_{j \in J}, \big\{d^{r_2}_j\big\}_{j \in J}`
          are scalars;

        * :math:`\big\{\mathbf{b^s_i}\big\}_{i\in I}` are vectors of dimension :math:`n^s_i` and
          :math:`\big\{A^s_i\big\}_{i\in I}` are matrices of size :math:`n^s_i \times n`;

        * :math:`\big\{\mathbf{b^r_j}\big\}_{j\in J}` are vectors of dimension :math:`n^r_j` and
          :math:`\big\{A^r_j\big\}_{j\in J}` are matrices of size :math:`n^r_j \times n`;

        * :math:`\mathbf{b^e}` is a vector of dimension :math:`n^e` and
          :math:`A^e` is a matrix of size :math:`n^e \times n`;

        * :math:`\mathbf{b^l}` is a vector of dimension :math:`n^l` and
          :math:`A^l` is a matrix of size :math:`n^l \times n`;

        * :math:`\big\{M_k\big\}_{k=0,\ldots,n}` are :math:`m \times m` symmetric
          matrices (:math:`M_k \in \mathbb{S}_m`).

Its dual problem can be written as:

.. math::
   :nowrap:   

   \begin{center}
   $\begin{array}{cll}
   \mbox{maximize}   & \mathbf{b^e}^T \mathbf{\mu^e}
                       + \mathbf{b^l}^T \mathbf{\mu^l} &
                       + \sum_{i\in I} \big( \mathbf{b^s_i}^T \mathbf{z^s_i} - d^s_i \lambda_i \big)
                       + \sum_{j\in J} \big( \mathbf{b^r_j}^T \mathbf{z^r_j} - d^{r_1}_j \alpha_j - d^{r_2}_j \beta_j \big)
                       + \langle M_0, X \rangle\\
   \mbox{subject to} & c + {A^e}^T \mathbf{\mu^e} + {A^l}^T \mathbf{\mu^l} &
                         + \sum_{i\in I} \big( {A^s_i}^T \mathbf{z^s_i} -\lambda_i \mathbf{f^s_i} \big)
                         + \sum_{j\in J} \big( {A^r_j}^T \mathbf{z^r_j} -\alpha_j \mathbf{f^{r_1}_j} - \beta_j \mathbf{f^{r_2}_j} \big)
                        = \mathcal{M} \bullet X \\
                     & \mu_l \geq 0 &\\
                     & \Vert \mathbf{z^s_i} \Vert \leq \lambda_i, &\forall i \in I\\
                     & \Vert \mathbf{z^r_j} \Vert^2 \leq 4 \alpha_j \beta_j, &\forall j \in J\\
                     & \ \ 0 \ \ \ \leq \alpha_j, &\forall j \in J\\
                     & X \succeq 0
   \end{array}$
   \end{center}

where :math:`\mathcal{M} \bullet X` stands for the vector of dimension :math:`n` 
with :math:`\langle M_i, X \rangle` on the :math:`i` th coordinate, and the dual variables
are

        * :math:`\mu^e \in \mathbb{R}^{n_e}`

        * :math:`\mu^l \in \mathbb{R}^{n_l}`

        * :math:`z^s_i \in \mathbb{R}^{n^s_i},\ \forall i \in I`

        * :math:`\lambda_i \in \mathbb{R},\ \forall i \in I`

        * :math:`z^r_j \in \mathbb{R}^{n^r_j},\ \forall j \in J`

        * :math:`(\alpha_j,\beta_j) \in \mathbb{R}^2,\ \forall j \in J`

        * :math:`X \in \mathbb{S}_m`

When quering the dual of a constraint of the above primal problem, **picos will
return**

        * :math:`\mu^e` for the constraint :math:`A^e \mathbf{x} + \mathbf{b^e} = 0`;

        * :math:`\mu^l` for the constraint :math:`A^l \mathbf{x} + \mathbf{b^l} \geq 0`;

        * The :math:`(n^s_i+1)-` dimensional vector :math:`\mu^s_i\ :=\ [\lambda_i;\mathbf{z^s_i}]\ ` for the constraint
        
                :math:`\Vert A^s_i \mathbf{x} + \mathbf{b^s_i} \Vert \leq \mathbf{f^s_i}^T \mathbf{x} +d^s_i`;

        * The :math:`(n^r_j+2)-` dimensional vector :math:`\mu^r_j\:=\ \frac{1}{2}[\ (\beta_j+\alpha_j) ;\ \mathbf{z^r_j} ;\ (\beta_j-\alpha_j)\ ]\ ` for the constraint
        
                :math:`\Vert A^r_j \mathbf{x} + \mathbf{b^r_j} \Vert^2 \leq (\mathbf{f^{r_1}_j}^T \mathbf{x} +d^{r_1}_j) (\mathbf{f^{r_2}_j}^T \mathbf{x} +d^{r_2}_j)`
                
          In other words, if the dual vector returned by picos is of the form
          :math:`\mu^r_j\ = [\sigma^1_j;\mathbf{u_j};\sigma^2_j]`, where :math:`\mathbf{u_j}` is of dimension :math:`n^r_j`,
          then the dual variables of the rotated conic constraint
          are :math:`\alpha_j = \sigma^1_j - \sigma^2_j,\ \beta_j = \sigma^1_j + \sigma^2_j`
          and :math:`\mathbf{z^r_j} = 2 \mathbf{u_j}`;

        * The symmetric positive definite matrix :math:`X` for the constraint

                 :math:`\sum_{i=1}^n x_i M_i \succeq M_0`.

.. _tuto_refs:

References
==========

        1. "`Applications of second-order cone programming`",
           M.S. Lobo, L. Vandenberghe, S. Boyd and H. Lebret,
           *Linear Algebra and its Applications*,
           284, p. *193-228*, 1998.
           
        2. "`On the semidefinite representations of real functions applied to symmetric
           matrices <http://opus4.kobv.de/opus4-zib/frontdoor/index/index/docId/1751>`_", G. Sagnol,
           *Linear Algebra and its Applications*,
           439(10), p. *2829-2843*, 2013.
