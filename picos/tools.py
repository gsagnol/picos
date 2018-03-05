# coding: utf-8

#-------------------------------------------------------------------
# Picos 1.1.3.dev : A pyton Interface To Conic Optimization Solvers
# Copyright (C) 2012  Guillaume Sagnol
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# For any information, please contact:
# Guillaume Sagnol
# sagnol@zib.de
# Konrad Zuse Zentrum für Informationstechnik Berlin (ZIB)
# Takustrasse 7
# D-14195 Berlin-Dahlem
# Germany
#-------------------------------------------------------------------

from __future__ import print_function, division

import cvxopt as cvx
import numpy as np

from six.moves import range, builtins
import six

__all__ = ['_retrieve_matrix',
           '_svecm1_identity',
           'eval_dict',
           'putIndices',
           '_blocdiag',
           'svec',
           'svecm1',
           'ltrim1',
           '_utri',
           'lowtri',
           'sum',
           '_bsum',
           'diag',
           'new_param',
           'available_solvers',
           'offset_in_lil',
           'import_cbf',
           'diag_vect',
           '_quad2norm',
           '_copy_exp_to_new_vars',
           'ProgressBar',
           '_NonWritableDict',
           'QuadAsSocpError',
           'NotAppropriateSolverError',
           'NonConvexError',
           'DualizationError',
           'geomean',
           '_break_cols',
           '_break_rows',
           '_block_idx',
           '_flatten',
           '_remove_in_lil',
           'norm',
           '_read_sdpa',
           'tracepow',
           'trace',
           'partial_transpose',
           'partial_trace',
           'detrootn',
           'ball',
           'sum_k_largest',
           'sum_k_largest_lambda',
           'lambda_max',
           'sum_k_smallest',
           'sum_k_smallest_lambda',
           'lambda_min',
           'simplex',
           'truncated_simplex',
           '_cplx_mat_to_real_mat',
           '_cplx_vecmat_to_real_vecmat',
           '_is_idty',
           'kron',
           '_is_integer',
           '_is_realvalued',
           '_is_numeric',
           'spmatrix'
           ]


#----------------------------------------------------
#        Grouping constraints, summing expressions
#----------------------------------------------------

def sum(lst, it=None, indices=None):
    u"""sum of a list of affine expressions.
    This fonction can be used with python list comprehensions
    (see the example below).

    :param lst: list of :class:`AffinExp <picos.AffinExp>`.
    :param it: Description of the letters which should
                    be used to replace the dummy indices.
                    The function tries to find a template
                    for the string representations of the
                    affine expressions in the list.
                    If several indices change in the
                    list, their letters should be given as a
                    list of strings, in their order of appearance in
                    the resulting string. For example, if three indices
                    change in the summands, and you want them to be
                    named ``'i'``, ``'j'`` and ``'k'``, set ``it = ['i','j','k']``.
                    You can also group two indices which always appear together,
                    e.g. if ``'i'`` always appear next to ``'j'`` you
                    could set ``it = [('ij',2),'k']``. Here, the number 2
                    indicates that ``'ij'`` replaces 2 indices.
                    If ``it`` is set to ``None``, or if the function is not
                    able to find a template, the string of
                    the first summand will be used for
                    the string representation of the sum.
    :type it: None or str or list.
    :param indices: a string to denote the set where the indices belong to.
    :type indices: str.

    **Example:**

    >>> import picos as pic
    >>> prob=pic.Problem()
    >>> x={}
    >>> names=['foo','bar','baz']
    >>> for n in names:
    ...   x[n]=prob.add_variable( 'x[{0}]'.format(n),(3,5) )
    >>> x #doctest: +NORMALIZE_WHITESPACE
    {'baz': # variable x[baz]:(3 x 5),continuous #,
     'foo': # variable x[foo]:(3 x 5),continuous #,
     'bar': # variable x[bar]:(3 x 5),continuous #}
    >>> pic.sum([x[n] for n in names],'n','names')
    # (3 x 5)-affine expression: Σ_{n in names} x[n] #
    >>> pic.sum([(i+1) * x[n] for i,n in enumerate(names)],['i','n'],'[3] x names') #two indices
    # (3 x 5)-affine expression: Σ_{i,n in [3] x names} i*x[n] #
    >>> IJ = [(1,2),(2,4),(0,1),(1,3)]
    >>> pic.sum([x['foo'][ij] for ij in IJ],[('ij',2)],'IJ') #double index
    # (1 x 1)-affine expression: Σ_{ij in IJ} x[foo][ij] #

    """
    from .expression import Expression
    from .expression import AffinExp
    if len(lst) == 0:
        return AffinExp(
            {}, constant=cvx.matrix(
                [0.], (1, 1)), size=(
                1, 1), string='0')
    if not(all([isinstance(exi, Expression) for exi in lst])):
        return builtins.sum(lst)
    #if 'z' in [m.typecode for exp in lst for m in exp.factors.values()
    #           ]:  # complex expression
    if any([exp.has_complex_coef() for exp in lst]):
        affSum = new_param('', cvx.matrix(0., lst[0].size, tc='z'))
    else:
        affSum = new_param('', cvx.matrix(0., lst[0].size, tc='d'))
    for lsti in lst:
        affSum += lsti
    if not it is None:
        sumstr = '_'
        if not indices is None:
            sumstr += '{'
        if isinstance(it, tuple) and len(it) == 2 and _is_integer(it[1]):
            it = (it,)
        if isinstance(it, list):
            it = tuple(it)
        if not isinstance(it, tuple):
            it = (it,)
        if isinstance(it[0], tuple):
            sumstr += str(it[0][0])
        else:
            sumstr += str(it[0])
        for k in [k for k in range(len(it)) if k > 0]:
            if isinstance(it[k], tuple):
                sumstr += ',' + str(it[k][0])
            else:
                sumstr += ',' + str(it[k])
        if not indices is None:
            sumstr += ' in ' + indices + '}'
        try:
            indstr = putIndices([l.affstring() for l in lst], it)
        except Exception:
            indstr = '[' + str(len(lst)) + \
                ' expressions (first: ' + lst[0].string + ')]'
        sumstr += ' ' + indstr
        sigma = 'Σ'  # 'u'\u03A3'.encode('utf-8')
        affSum.string = sigma + sumstr
    return affSum


def _bsum(lst):
    """builtin sum operator"""
    return builtins.sum(lst)


def _break_cols(mat, sizes):
    n = len(sizes)
    I, J, V = [], [], []
    for i in range(n):
        I.append([])
        J.append([])
        V.append([])
    cumsz = np.cumsum(sizes)
    import bisect
    for i, j, v in zip(mat.I, mat.J, mat.V):
        block = bisect.bisect(cumsz, j)
        I[block].append(i)
        V[block].append(v)
        if block == 0:
            J[block].append(j)
        else:
            J[block].append(j - cumsz[block - 1])
    return [spmatrix(V[k], I[k], J[k], (mat.size[0], sz))
            for k, sz in enumerate(sizes)]


def _break_rows(mat, sizes):
    n = len(sizes)
    I, J, V = [], [], []
    for i in range(n):
        I.append([])
        J.append([])
        V.append([])
    cumsz = np.cumsum(sizes)
    import bisect
    for i, j, v in zip(mat.I, mat.J, mat.V):
        block = bisect.bisect(cumsz, i)
        J[block].append(j)
        V[block].append(v)
        if block == 0:
            I[block].append(i)
        else:
            I[block].append(i - cumsz[block - 1])
    return [spmatrix(V[k], I[k], J[k], (sz, mat.size[1]))
            for k, sz in enumerate(sizes)]


def _block_idx(i, sizes):
    # if there are blocks of sizes n1,...,nk and i is
    # the index of an element of the big vectorized variable,
    # returns the block of i and its index inside the sub-block.
    cumsz = np.cumsum(sizes)
    import bisect
    block = bisect.bisect(cumsz, i)
    return block, (i if block == 0 else i - cumsz[block - 1])


def geomean(exp):
    """returns a :class:`GeoMeanExp <picos.GeoMeanExp>` object representing the geometric mean of the entries of ``exp[:]``.
    This can be used to enter inequalities of the form ``t <= geomean(x)``.
    Note that geometric mean inequalities are internally reformulated as a
    set of SOC inequalities.

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> x = prob.add_variable('x',1)
    >>> y = prob.add_variable('y',3)
    >>> # the following line adds the constraint x <= (y0*y1*y2)**(1./3) in the problem:
    >>> prob.add_constraint(x<pic.geomean(y))

    """
    from .expression import AffinExp
    from .expression import GeoMeanExp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    return GeoMeanExp(exp)


def norm(exp, num=2, denom=1):
    """returns a :class:`NormP_Exp <picos.NormP_Exp>` object representing the (generalized-) p-norm of the entries of ``exp[:]``.
    This can be used to enter constraints of the form :math:`\Vert x \Vert_p \leq t` with :math:`p\geq1`.
    Generalized norms are also defined for :math:`p<1`, by using the usual formula
    :math:`\operatorname{norm}(x,p) := \Big(\sum_i x_i^p\Big)^{1/p}`. Note that this function
    is concave (for :math:`p<1`) over the set of vectors with nonnegative coordinates.
    When a constraint of the form :math:`\operatorname{norm}(x,p) > t` with :math:`p\leq1` is entered, PICOS implicitely assumes that :math:`x` is a nonnegative vector.

    This function can also be used to represent the Lp,q- norm of a matrix (for :math:`p,q \geq 1`):
    :math:`\operatorname{norm}(X,(p,q)) := \Big(\sum_i (\sum_j x_{ij}^q )^{p/q}\Big)^{1/p}`,
    that is, the p-norm of the vector formed with the q-norms of the rows of :math:`X`.

    The exponent :math:`p` of the norm must be specified either by
    a couple numerator (2d argument) / denominator (3d arguments),
    or directly by a float ``p`` given as second argument. In the latter case a rational
    approximation of ``p`` will be used. It is also possible to pass ``'inf'``  as
    second argument for the infinity-norm (aka max-norm).

    For the case of :math:`(p,q)`-norms, ``p`` and ``q`` must be specified by a tuple of floats
    in the second argument (rational approximations will be used), and the third argument will
    be ignored.

    **Example:**

    >>> import picos as pic
    >>> P = pic.Problem()
    >>> x = P.add_variable('x',1)
    >>> y = P.add_variable('y',3)
    >>> pic.norm(y,7,3) < x
    # p-norm ineq : norm_7/3( y)<x#
    >>> pic.norm(y,-0.4) > x
    # generalized p-norm ineq : norm_-2/5( y)>x#
    >>> X = P.add_variable('X',(3,2))
    >>> pic.norm(X,(1,2)) < 1
    # pq-norm ineq : norm_1,2( X)<1.0#
    >>> pic.norm(X,('inf',1)) < 1
    # pq-norm ineq : norm_inf,1( X)<1.0#

    """
    from .expression import AffinExp
    from .expression import NormP_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    if num == 2 and denom == 1:
        return abs(exp)
    if isinstance(num, tuple) and len(num) == 2:
        if denom != 1:
            raise ValueError(
                'tuple as 2d argument for L(p,q)-norm and 3d argument is !=1')
        return NormP_Exp(exp, num[0], 1, num[1], 1)
    p = float(num) / float(denom)
    if p == 0:
        raise Exception('undefined for p=0')
    if p == float('inf'):
        return NormP_Exp(exp, float('inf'), 1)
    elif p == float('-inf'):
        return NormP_Exp(exp, float('-inf'), 1)
    else:
        from fractions import Fraction
        frac = Fraction(p).limit_denominator(1000)
        return NormP_Exp(exp, frac.numerator, frac.denominator)


def tracepow(exp, num=1, denom=1, coef=None):
    """Returns a :class:`TracePow_Exp <picos.TracePow_Exp>` object representing the trace of the pth-power of the symmetric matrix ``exp``, where ``exp`` is an :class:`AffinExp <picos.AffinExp>` which we denote by :math:`X`.
    This can be used to enter constraints of the form :math:`\operatorname{trace} X^p \leq t` with :math:`p\geq1` or :math:`p < 0`, or :math:`\operatorname{trace} X^p \geq t` with :math:`0 \leq p \leq 1`.
    Note that :math:`X` is forced to be positive semidefinite when a constraint of this form is entered in PICOS.

    It is also possible to specify a ``coef`` matrix (:math:`M`) of the same size as ``exp``, in order to represent the expression  :math:`\operatorname{trace} (M X^p)`.
    The constraint :math:`\operatorname{trace} (M X^p)\geq t` can be reformulated with SDP constraints if :math:`M` is positive
    semidefinite and :math:`0<p<1`.

    Trace of power inequalities are internally reformulated as a set of Linear Matrix Inequalities (SDP),
    or second order cone inequalities if ``exp`` is a scalar.

    The exponent :math:`p` of the norm must be specified either by
    a couple numerator (2d argument) / denominator (3d arguments),
    or directly by a float ``p`` given as second argument. In the latter case a rational
    approximation of ``p`` will be used.

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> X = prob.add_variable('X',(3,3),'symmetric')
    >>> t = prob.add_variable('t',1)
    >>> pic.tracepow(X,7,3) < t
    # trace of pth power ineq : trace( X)**7/3<t#
    >>> pic.tracepow(X,0.6) > t
    # trace of pth power ineq : trace( X)**3/5>t#

    >>> A = cvx.normal(3,3);A=A*A.T #A random semidefinite positive matrix
    >>> A = pic.new_param('A',A)
    >>> pic.tracepow(X,0.25,coef=A) > t
    # trace of pth power ineq : trace[ A *(X)**1/4]>t#
    """
    from .expression import AffinExp
    from .expression import TracePow_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    if not(coef is None) and not isinstance(coef, AffinExp):
        M, Mstr = _retrieve_matrix(coef)
        coef = AffinExp({}, constant=M[:], size=M.size, string=Mstr)
    if num == denom:
        return ('I' | exp)
    p = float(num) / float(denom)
    if p == 0:
        raise Exception('undefined for p=0')
    from fractions import Fraction
    frac = Fraction(p).limit_denominator(1000)
    return TracePow_Exp(exp, frac.numerator, frac.denominator, coef)


def trace(exp):
    """
    trace of a square AffinExp
    """
    return tracepow(exp)


def sum_k_largest(exp, k):
    """returns a :class:`Sum_k_Largest_Exp <picos.Sum_k_Largest_Exp>` object representing the sum
    of the ``k`` largest elements of an affine expression ``exp``.
    This can be used to enter constraints of the form :math:`\sum_{i=1}^k x_{i}^{\downarrow} \leq t`.
    This kind of constraints is reformulated internally as a set of linear inequalities.

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> x = prob.add_variable('x',3)
    >>> t = prob.add_variable('t',1)
    >>> pic.sum_k_largest(x,2) < 1
    # sum_k_largest constraint : sum_2_largest(x)<1.0#
    >>> pic.sum_k_largest(x,1) < t
    # (3x1)-affine constraint: max(x)<=t #

    """
    from .expression import AffinExp
    from .expression import Sum_k_Largest_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    return Sum_k_Largest_Exp(exp, k, False)


def sum_k_largest_lambda(exp, k):
    """returns a :class:`Sum_k_Largest_Exp <picos.Sum_k_Largest_Exp>` object representing the sum
    of the ``k`` largest eigenvalues of a square matrix affine expression ``exp``.
    This can be used to enter constraints of the form :math:`\sum_{i=1}^k \lambda_{i}^{\downarrow}(X) \leq t`.
    This kind of constraints is reformulated internally as a set of linear matrix inequalities (SDP).
    Note that ``exp`` is assumed to be symmetric (picos does not check).

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> X = prob.add_variable('X',(3,3),'symmetric')
    >>> t = prob.add_variable('t',1)
    >>> pic.sum_k_largest_lambda(X,3) < 1 #this is simply the trace of X
    # (1x1)-affine constraint: 〈 I | X 〉 < 1.0 #
    >>> pic.sum_k_largest_lambda(X,2) < t
    # sum_k_largest constraint : sum_2_largest_lambda(X)<t#

    """
    from .expression import AffinExp
    from .expression import Sum_k_Largest_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    return Sum_k_Largest_Exp(exp, k, True)


def lambda_max(exp):
    """
    largest eigenvalue of a square matrix expression (cf. :func:`pic.sum_k_largest(exp,1) <picos.tools.sum_k_largest_lambda>`)

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> x = prob.add_variable('X',(3,3),'symmetric')
    >>> pic.lambda_max(X) < 2
    # (3x3)-LMI constraint lambda_max(X)<=2.0 #
    """
    return sum_k_largest_lambda(exp, 1)


def sum_k_smallest(exp, k):
    """returns a :class:`Sum_k_Smallest_Exp <picos.Sum_k_Smallest_Exp>` object representing the sum
    of the ``k`` smallest elements of an affine expression ``exp``.
    This can be used to enter constraints of the form :math:`\sum_{i=1}^k x_{i}^{\\uparrow} \geq t`.
    This kind of constraints is reformulated internally as a set of linear inequalities.

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> x = prob.add_variable('x',3)
    >>> t = prob.add_variable('t',1)
    >>> pic.sum_k_smallest(x,2) > t
    # sum_k_smallest constraint : sum_2_smallest(x)>t#
    >>> pic.sum_k_smallest(x,1) > 3
    # (3x1)-affine constraint: min(x)>=3.0 #

    """
    from .expression import AffinExp
    from .expression import Sum_k_Smallest_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    return Sum_k_Smallest_Exp(exp, k, False)


def sum_k_smallest_lambda(exp, k):
    """returns a :class:`Sum_k_Smallest_Exp <picos.Sum_k_Smallest_Exp>` object representing the sum
    of the ``k`` smallest eigenvalues of a square matrix affine expression ``exp``.
    This can be used to enter constraints of the form :math:`\sum_{i=1}^k \lambda_{i}^{\\uparrow}(X) \geq t`.
    This kind of constraints is reformulated internally as a set of linear matrix inequalities (SDP).
    Note that ``exp`` is assumed to be symmetric (picos does not check).

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> X = prob.add_variable('X',(3,3),'symmetric')
    >>> t = prob.add_variable('t',1)
    >>> pic.sum_k_smallest_lambda(X,1) > 1
    # (3x3)-LMI constraint lambda_min(X)>=1.0 #
    >>> pic.sum_k_smallest_lambda(X,2) > t
    # sum_k_smallest constraint : sum_2_smallest_lambda(X)>t#

    """
    from .expression import AffinExp
    from .expression import Sum_k_Smallest_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    return Sum_k_Smallest_Exp(exp, k, True)


def lambda_min(exp):
    """
    smallest eigenvalue of a square matrix expression (cf. :func:`pic.sum_k_smallest(exp,1) <picos.tools.sum_k_smallest_lambda>`)

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> x = prob.add_variable('X',(3,3),'symmetric')
    >>> pic.lambda_min(X) > -1
    # (3x3)-LMI constraint lambda_min(X)>=-1.0 #
    """
    return sum_k_smallest_lambda(exp, 1)


def partial_transpose(exp, dims_1=None, subsystems = None, dims_2=None):
    r"""Partial transpose of an Affine Expression, with respect to
    given subsystems. If ``X`` is a matrix
    :class:`AffinExp <picos.AffinExp>`
    that can be written as :math:`X = A_0 \otimes \cdots \otimes A_{n-1}`
    for some matrices :math:`A_0,\ldots,A_{n-1}`
    of respective sizes ``dims_1[0] x dims_2[0]``, ... , ``dims_1[n-1] x dims_2[n-1]``,
    this function returns the matrix
    :math:`Y = B_0 \otimes \cdots \otimes B_{n-1}`,
    where :math:`B_i=A_i^T` if ``i in subsystems``, and  :math:`B_i=A_i` otherwise.

    The optional parameters ``dims_1`` and ``dims_2`` are tuples specifying the dimension
    of each subsystem :math:`A_i`. The argument ``subsystems`` must be a ``tuple`` (or an ``int``) with the
    index of all subsystems to be transposed.

    The default value ``dims_1=None`` automatically computes the size of the subblocks,
    assuming that ``exp`` is a :math:`n^k \times n^k`-square matrix,
    for the *smallest* appropriate value of :math:`k \in [2,6]`, that is ``dims_1=(n,)*k``.

    If ``dims_2`` is not specified, it is assumed that the subsystems :math:`A_i` are square,
    i.e., ``dims_2=dims_1``. If ``subsystems`` is not specified, the default assumes that
    only the last system must be transposed, i.e., ``subsystems = (len(dims_1)-1,)``

    **Example:**

    >>> import picos as pic
    >>> import cvxopt as cvx
    >>> P = pic.Problem()
    >>> X = P.add_variable('X',(4,4))
    >>> X.value = cvx.matrix(range(16),(4,4))
    >>> print X #doctest: +NORMALIZE_WHITESPACE
    [ 0.00e+00  4.00e+00  8.00e+00  1.20e+01]
    [ 1.00e+00  5.00e+00  9.00e+00  1.30e+01]
    [ 2.00e+00  6.00e+00  1.00e+01  1.40e+01]
    [ 3.00e+00  7.00e+00  1.10e+01  1.50e+01]
    >>> print X.Tx #standard partial transpose (with respect to the 2x2 blocks and 2d subsystem) #doctest: +NORMALIZE_WHITESPACE
    [ 0.00e+00  1.00e+00  8.00e+00  9.00e+00]
    [ 4.00e+00  5.00e+00  1.20e+01  1.30e+01]
    [ 2.00e+00  3.00e+00  1.00e+01  1.10e+01]
    [ 6.00e+00  7.00e+00  1.40e+01  1.50e+01]
    >>> print pic.partial_transpose(X,(2,2),0) #(now with respect to the first subsystem) #doctest: +NORMALIZE_WHITESPACE
    [ 0.00e+00  4.00e+00  2.00e+00  6.00e+00]
    [ 1.00e+00  5.00e+00  3.00e+00  7.00e+00]
    [ 8.00e+00  1.20e+01  1.00e+01  1.40e+01]
    [ 9.00e+00  1.30e+01  1.10e+01  1.50e+01]

    """
    return exp.partial_transpose(dims_1,subsystems,dims_2)


def partial_trace(X, k=1, dim=None):
    r"""Partial trace of an Affine Expression, with respect to the ``k`` th subsystem for a tensor product of dimensions ``dim``.
    If ``X`` is a matrix
    :class:`AffinExp <picos.AffinExp>`
    that can be written as :math:`X = A_0 \otimes \cdots \otimes A_{n-1}`
    for some matrices :math:`A_0,\ldots,A_{n-1}`
    of respective sizes ``dim[0] x dim[0]``, ... , ``dim[n-1] x dim[n-1]`` (``dim`` is a list of ints if all matrices are square),
    or ``dim[0][0] x dim[0][1]``, ...,``dim[n-1][0] x dim[n-1][1]`` (``dim`` is a list of 2-tuples if any of them except the ``k`` th one is rectangular),
    this function returns the matrix
    :math:`Y = \operatorname{trace}(A_k)\quad A_0 \otimes \cdots A_{k-1} \otimes A_{k+1} \otimes \cdots \otimes A_{n-1}`.

    The default value ``dim=None`` automatically computes the size of the subblocks,
    assuming that ``X`` is a :math:`n^2 \times n^2`-square matrix
    with blocks of size :math:`n \times n`.

    **Example:**

    >>> import picos as pic
    >>> import cvxopt as cvx
    >>> P = pic.Problem()
    >>> X = P.add_variable('X',(4,4))
    >>> X.value = cvx.matrix(range(16),(4,4))
    >>> print X #doctest: +NORMALIZE_WHITESPACE
    [ 0.00e+00  4.00e+00  8.00e+00  1.20e+01]
    [ 1.00e+00  5.00e+00  9.00e+00  1.30e+01]
    [ 2.00e+00  6.00e+00  1.00e+01  1.40e+01]
    [ 3.00e+00  7.00e+00  1.10e+01  1.50e+01]
    >>> print pic.partial_trace(X) #partial trace with respect to second subsystem (k=1) #doctest: +NORMALIZE_WHITESPACE
    [ 5.00e+00  2.10e+01]
    [ 9.00e+00  2.50e+01]
    >>> print pic.partial_trace(X,0) #and now with respect to first subsystem (k=0) #doctest: +NORMALIZE_WHITESPACE
    [ 1.00e+01  1.80e+01]
    [ 1.20e+01  2.00e+01]

    """
    return X.partial_trace(k, dim)


def detrootn(exp):
    """returns a :class:`DetRootN_Exp <picos.DetRootN_Exp>` object representing the determinant of the
    :math:`n` th-root of the symmetric matrix ``exp``, where :math:`n` is the dimension of the matrix.
    This can be used to enter constraints of the form :math:`(\operatorname{det} X)^{1/n} \geq t`.
    Note that :math:`X` is forced to be positive semidefinite when a constraint of this form is entered in PICOS.
    Determinant inequalities are internally reformulated as a set of Linear Matrix Inequalities (SDP).

    **Example:**

    >>> import picos as pic
    >>> prob = pic.Problem()
    >>> X = prob.add_variable('X',(3,3),'symmetric')
    >>> t = prob.add_variable('t',1)
    >>> t < pic.detrootn(X)
    # nth root of det ineq : det( X)**1/3>t#

    """
    from .expression import AffinExp
    from .expression import DetRootN_Exp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    return DetRootN_Exp(exp)


def ball(r, p=2):
    """returns a :class:`Ball <picos.expression.Ball>` object representing:

      * a L_p Ball of radius ``r`` (:math:`\{x: \Vert x \Vert_p \geq r \}`) if :math:`p \geq 1`
      * the convex set :math:`\{x\geq 0: \Vert x \Vert_p \geq r \}` :math:`p < 1`.

    **Example**

    >>> import picos as pic
    >>> P = pic.Problem()
    >>> x = P.add_variable('x', 3)
    >>> x << pic.ball(2,3)  #doctest: +NORMALIZE_WHITESPACE
    # p-norm ineq : norm_3( x)<2.0#
    >>> x << pic.ball(1,0.5)
    # generalized p-norm ineq : norm_1/2( x)>1.0#

    """
    from .expression import Ball
    return Ball(p, r)


def simplex(gamma=1):
    """returns a :class:`Truncated_Simplex <picos.expression.Truncated_Simplex>` object representing the set :math:`\{x\geq 0: ||x||_1 \leq \gamma \}`.

    **Example**

    >>> import picos as pic
    >>> P = pic.Problem()
    >>> x = P.add_variable('x', 3)
    >>> x << pic.simplex(1)
    # (4x1)-affine constraint: x in standard simplex #
    >>> x << pic.simplex(2)
    # (4x1)-affine constraint: x in simplex of radius 2 #

    """
    from .expression import Truncated_Simplex
    return Truncated_Simplex(radius=gamma, truncated=False, nonneg=True)


def truncated_simplex(gamma=1, sym=False):
    """returns a :class:`Truncated_Simplex <picos.expression.Truncated_Simplex>` object representing object representing the set:

      * :math:`\{x \geq  0: ||x||_\infty \leq 1,\ ||x||_1 \leq \gamma \}` if ``sym=False`` (default)
      * :math:`\{x: ||x||_\infty \leq 1,\ ||x||_1 \leq \gamma \}` if ``sym=True``.

    **Example**

    >>> import picos as pic
    >>> P = pic.Problem()
    >>> x = P.add_variable('x', 3)
    >>> x << pic.truncated_simplex(2)
    # (7x1)-affine constraint: x in truncated simplex of radius 2 #
    >>> x << pic.truncated_simplex(2,sym=True)
    # symmetrized truncated simplex constraint : ||x||_{infty;1} <= {1;2}#

    """
    from .expression import Truncated_Simplex
    return Truncated_Simplex(radius=gamma, truncated=True, nonneg=not(sym))


def allIdent(lst):
    if len(lst) <= 1:
        return(True)
    return (np.array([lst[i] == lst[i + 1]
                      for i in range(len(lst) - 1)]).all())


def putIndices(lsStrings, it):
    # for multiple indices
    toMerge = []
    for k in it:
        if isinstance(k, tuple):
            itlist = list(it)
            ik = itlist.index(k)
            itlist.remove(k)
            for i in range(k[1]):
                itlist.insert(ik, k[0] + '__' + str(i))
                ik += 1
            toMerge.append((k[0], itlist[ik - k[1]:ik]))
            it = tuple(itlist)
    # main function
    fr = cut_in_frames(lsStrings)
    frame = put_indices_on_frames(fr, it)
    # merge multiple indices
    import re
    import string
    for x in toMerge:
        rexp = '(\(( )*' + string.join(x[1], ',( )*') + \
            '( )*\)|(' + string.join(x[1], ',( )*') + '))'
        m = re.search(rexp, frame)
        while(m):
            frame = frame[:m.start()] + x[0] + frame[m.end():]
            m = re.search(rexp, frame)
    return frame


def is_index_char(char):
    return char.isalnum() or char == '_' or char == '.'


def findEndOfInd(string, curInd):
    indx = ''
    while curInd < len(string) and is_index_char(string[curInd]):
        indx += string[curInd]
        curInd += 1
    if indx == '':
        raise Exception('empty index')
    return curInd, indx


def cut_in_frames(lsStrings):
    n = len(lsStrings)
    curInd = n * [0]
    frame = []
    while curInd[0] < len(lsStrings[0]):
        tmpName = [None] * n
        currentFramePiece = ''
        piece_of_frame_found = False
        # go on while we have the same char
        while allIdent([lsStrings[k][curInd[k]] for k in range(n)]):
            currentFramePiece += lsStrings[0][curInd[0]]
            piece_of_frame_found = True
            curInd = [c + 1 for c in curInd]
            if curInd[0] >= len(lsStrings[0]):
                break
        if not piece_of_frame_found:
            # there was no template frame between successive indices
            if curInd[0] == 0:  # we are still at the beginning
                pass
            else:
                raise Exception('unexpected template')
        # go back until we get a non index char
        #import pdb;pdb.set_trace()
        if curInd[0] < len(lsStrings[0]):
            while curInd[0] > 0 and is_index_char(lsStrings[0][curInd[0] - 1]):
                currentFramePiece = currentFramePiece[:-1]
                curInd = [c - 1 for c in curInd]
        frame.append(currentFramePiece)
        if curInd[0] < len(lsStrings[0]):
            for k in range(n):
                curInd[k], tmpName[k] = findEndOfInd(lsStrings[k], curInd[k])
            frame.append(tmpName)
    return frame


def put_indices_on_frames(frames, indices):
    frames_index = []
    # find indices of frames
    for i, f in enumerate(frames):
        if isinstance(f, list):
            frames_index.append(i)
    replacement_index = []
    index_types = []
    non_replaced_index = []

    # find index types
    for num_index, fi in enumerate(frames_index):
        alpha = []
        num = 0
        for t in frames[fi]:
            tsp = t.split('.')
            if (len(tsp) <= 2 and
                all([s.isdigit() for s in tsp if len(s) > 0])
                ):
                num += 1
            else:
                alpha.append(t)
        # we have a mix of numeric and alpha types in a frame,
        # with always the same alpha:
        # -> patch for sub sums with only one term,
        # that was not replaced by its index
        if len(alpha) > 0 and num > 0:
            if allIdent(alpha):
                # patch
                replacement_index.append(alpha[0])
                index_types.append('resolved')
                non_replaced_index.append(num_index)
            else:
                raise Exception('mix of numeric and alphabetic indices' +
                                'for the index number {0}'.format(num_index))
        elif len(alpha) > 0 and num == 0:
            replacement_index.append(None)
            index_types.append('alpha')
        elif len(alpha) == 0 and num > 0:
            replacement_index.append(None)
            index_types.append('num')
        else:
            raise Exception('unexpected index type' +
                            'for the index number {0}'.format(num_index))

    # set a replacement index
    previous_numeric_index = []
    previous_alphabetic_index = []
    ind_next_index = 0
    for num_index, fi in enumerate(frames_index):
        if replacement_index[num_index] is None:
            if index_types[num_index] == 'num':
                # check if we have a constant offset with a previous index
                for j in previous_numeric_index:
                    prev_frame = frames[frames_index[j]]
                    diff = [
                        float(i) -
                        float(p) for (
                            p,
                            i) in zip(
                            prev_frame,
                            frames[fi])]
                    if allIdent(diff):
                        ind = replacement_index[j]
                        if diff[0] > 0:
                            ind += '+'
                        if diff[0] != 0:
                            offset = diff[0]
                            if offset == int(offset):
                                offset = int(offset)
                            ind += str(offset)
                        replacement_index[num_index] = ind
                        break
                if not(replacement_index[num_index] is None):
                    continue
            elif index_types[num_index] == 'alpha':
                # check if we have the same index
                for j in previous_alphabetic_index:
                    prev_frame = frames[frames_index[j]]
                    same = [
                        st == st2 for st, st2 in zip(
                            prev_frame, frames[fi])]
                    if all(same):
                        replacement_index[num_index] = replacement_index[j]
                        break
                if not(replacement_index[num_index] is None):
                    continue

            if ind_next_index >= len(indices):
                raise Exception('too few indices')
            replacement_index[num_index] = indices[ind_next_index]
            ind_next_index += 1
            if index_types[num_index] == 'num':
                previous_numeric_index.append(num_index)
            if index_types[num_index] == 'alpha':
                previous_alphabetic_index.append(num_index)

    if len(indices) != ind_next_index:
        raise Exception('too many indices')

    # return the complete frame
    ret_string = ''
    for num_index, fi in enumerate(frames_index):
        frames[fi] = replacement_index[num_index]
    for st in frames:
        ret_string += st
    return ret_string


def eval_dict(dict_of_variables):
    """
    if ``dict_of_variables`` is a dictionary
    mapping variable names (strings) to :class:`variables <picos.Variable>`,
    this function returns the dictionary ``names -> variable values``.
    """
    valued_dict = {}

    for k in dict_of_variables:
        valued_dict[k] = dict_of_variables[k].eval()
        if valued_dict[k].size == (1, 1):
            valued_dict[k] = valued_dict[k][0]
    return valued_dict


#---------------------------------------------
#        Tools of the interface
#---------------------------------------------

def _blocdiag(X, n):
    """
    makes diagonal blocs of X, for indices in [sub1,sub2[
    n indicates the total number of blocks (horizontally)
    """
    if not isinstance(X, cvx.base.spmatrix):
        X = cvx.sparse(X)
    if n==1:
        return X
    else:
        Z = spmatrix([],[],[],X.size)
        mat = []
        for i in range(n):
            col = [Z]*(n-1)
            col.insert(i,X)
            mat.append(col)
        return cvx.sparse(mat)

def lse(exp):
    """
    shorter name for the constructor of the class :class:`LogSumExp <picos.LogSumExp>`

    **Example**

    >>> import picos as pic
    >>> import cvxopt as cvx
    >>> prob=pic.Problem()
    >>> x=prob.add_variable('x',3)
    >>> A=pic.new_param('A',cvx.matrix([[1,2],[3,4],[5,6]]))
    >>> pic.lse(A*x)<0
    # (2x1)-Geometric Programming constraint LSE[ A*x ] < 0 #

    """
    from .expression import LogSumExp
    return LogSumExp(exp)


def diag(exp, dim=1):
    r"""
    if ``exp`` is an affine expression of size (n,m),
    ``diag(exp,dim)`` returns a diagonal matrix of size ``dim*n*m`` :math:`\times` ``dim*n*m``,
    with ``dim`` copies of the vectorized expression ``exp[:]`` on the diagonal.

    In particular:

      * when ``exp`` is scalar, ``diag(exp,n)`` returns a diagonal
        matrix of size :math:`n \times n`, with all diagonal elements equal to ``exp``.

      * when ``exp`` is a vector of size :math:`n`, ``diag(exp)`` returns the diagonal
        matrix of size :math:`n \times n` with the vector ``exp`` on the diagonal


    **Example**

    >>> import picos as pic
    >>> prob=pic.Problem()
    >>> x=prob.add_variable('x',1)
    >>> y=prob.add_variable('y',1)
    >>> pic.tools.diag(x-y,4)
    # (4 x 4)-affine expression: Diag(x -y) #
    >>> pic.tools.diag(x//y)
    # (2 x 2)-affine expression: Diag([x;y]) #

    """
    from .expression import AffinExp
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    (n, m) = exp.size
    expcopy = AffinExp(exp.factors.copy(), exp.constant, exp.size,
                       exp.string)
    idx = cvx.spdiag([1.] * dim * n * m)[:].I
    for k in exp.factors.keys():
        # ensure it's sparse
        mat = cvx.sparse(expcopy.factors[k])
        I, J, V = list(mat.I), list(mat.J), list(mat.V)
        newI = []
        for d in range(dim):
            for i in I:
                newI.append(idx[i + n * m * d])
        expcopy.factors[k] = spmatrix(
            V * dim, newI, J * dim, ((dim * n * m)**2, exp.factors[k].size[1]))
    expcopy.constant = cvx.matrix(0., ((dim * n * m)**2, 1))
    if not exp.constant is None:
        for k, i in enumerate(idx):
            expcopy.constant[i] = exp.constant[k % (n * m)]
    expcopy._size = (dim * n * m, dim * n * m)
    expcopy.string = 'Diag(' + exp.string + ')'
    return expcopy


def diag_vect(exp):
    """
    Returns the vector with the diagonal elements of the matrix expression ``exp``

    **Example**

    >>> import picos as pic
    >>> prob=pic.Problem()
    >>> X=prob.add_variable('X',(3,3))
    >>> pic.tools.diag_vect(X)
    # (3 x 1)-affine expression: diag(X) #

    """
    from .expression import AffinExp
    (n, m) = exp.size
    n = min(n, m)
    idx = cvx.spdiag([1.] * n)[:].I
    expcopy = AffinExp(exp.factors.copy(), exp.constant, exp.size,
                       exp.string)
    proj = spmatrix([1.] * n, range(n), idx,
                        (n, exp.size[0] * exp.size[1]))
    for k in exp.factors.keys():
        expcopy.factors[k] = proj * expcopy.factors[k]
    if not exp.constant is None:
        expcopy.constant = proj * expcopy.constant
    expcopy._size = (n, 1)
    expcopy.string = 'diag(' + exp.string + ')'
    return expcopy


def _retrieve_matrix(mat, exSize=None):
    """
    parses the variable *mat* and convert it to a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
    If the variable **exSize** is provided, the function tries
    to return a matrix that matches this expected size, or raise an
    error.

    .. WARNING:: If there is a conflit between the size of **mat** and
                 the expected size **exsize**, the function might still
                 return something without raising an error !

    :param mat: The value to be converted into a cvx.spmatrix.
                The function will try to parse this variable and
                format it to a vector/matrix. *mat* can be of one
                of the following types:

                    * ``list`` [creates a vecor of dimension len(list)]
                    * :func:`cvxopt matrix <cvxopt:cvxopt.matrix>`
                    * :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`
                    * :func:`numpy array <numpy:numpy.array>`
                    * ``int`` or ``real`` [creates a vector/matrix of the size exSize *(or of size (1,1) if exSize is None)*,
                      whith all entries equal to **mat**.
                    * following strings:

                            * '``|a|``' for a matrix with all terms equal to a
                            * '``|a|(n,m)``' for a matrix forced to be of size n x m, with all terms equal to a
                            * '``e_i(n,m)``' matrix of size (n,m), with a 1 on the ith coordinate (and 0 elsewhere)
                            * '``e_i,j(n,m)``' matrix  of size (n,m), with a 1 on the (i,j)-entry (and 0 elsewhere)
                            * '``I``' for the identity matrix
                            * '``I(n)``' for the identity matrix, forced to be of size n x n.
                            * '``a%s``', where ``%s`` is one of the above string: the matrix that
                              should be returned when **mat** == ``%s``, multiplied by the scalar a.
    :returns: A tuple of the form (**M**, **s**), where **M** is the conversion of **mat** into a
              :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`, and **s**
              is a string representation of **mat**

    **Example:**

    >>> import picos as pic
    >>> pic.tools._retrieve_matrix([1,2,3])
    (<3x1 sparse matrix, tc='d', nnz=3>, '[ 3 x 1 MAT ]')
    >>> pic.tools._retrieve_matrix('e_5(7,1)')
    (<7x1 sparse matrix, tc='d', nnz=1>, 'e_5')
    >>> print pic.tools._retrieve_matrix('e_11(7,2)')[0] #doctest: +NORMALIZE_WHITESPACE
    [   0        0       ]
    [   0        0       ]
    [   0        0       ]
    [   0        0       ]
    [   0        1.00e+00]
    [   0        0       ]
    [   0        0       ]
    >>> print pic.tools._retrieve_matrix('5.3I',(2,2))
    (<2x2 sparse matrix, tc='d', nnz=2>, '5.3I')

    """
    retstr = None
    from .expression import Expression

    if isinstance(mat, Expression) and mat.is_valued():
        if isinstance(
                mat.value,
                cvx.base.spmatrix) or isinstance(
                mat.value,
                cvx.base.matrix):
            retmat = mat.value
        else:
            retmat = cvx.matrix(mat.value)
    elif isinstance(mat, np.ndarray):
        if np.iscomplex(mat).any():
            try:
                retmat = cvx.matrix(mat, tc='z')
            except:
                retmat = cvx.matrix(np.matrix(mat), tc='z')
        else:
            try:
                retmat = cvx.matrix(mat, tc='d')
            except:
                retmat = cvx.matrix(np.matrix(mat).real.tolist(), tc='d')

    elif isinstance(mat, cvx.base.matrix):
        if mat.typecode == 'd':
            retmat = mat
        elif mat.typecode == 'z':
            retmat = mat
        else:
            retmat = cvx.matrix(mat, tc='d')
    elif isinstance(mat, cvx.base.spmatrix):
        retmat = mat
    elif isinstance(mat, list):
        if any([isinstance(v, complex) for v in mat]):
            tc = 'z'
        else:
            tc = 'd'
        if (isinstance(exSize, tuple)  # it matches the expected size
                and len(exSize) == 2
                and not(exSize[0] is None)
                and not(exSize[1] is None)
                and len(mat) == exSize[0] * exSize[1]):
            retmat = cvx.matrix(np.array(mat), exSize, tc=tc)
        else:  # no possible match
            retmat = cvx.matrix(np.array(mat), tc=tc)
    elif _is_numeric(mat):
        if 'complex' in str(type(mat)):
            mat = complex(mat)
            if mat.imag:
                tc = 'z'
            else:
                mat = mat.real
                tc = 'd'
        else:
            mat = float(mat)
            tc = 'd'
        if not mat:  # ==0
            if exSize is None:
                # no exSize-> scalar
                retmat = cvx.matrix(0., (1, 1))
            elif _is_integer(exSize):
                # exSize is an int -> 0 * identity matrix
                retmat = spmatrix([], [], [], (exSize, exSize))
            elif isinstance(exSize, tuple):
                # exSize is a tuple -> zeros of desired size
                retmat = spmatrix([], [], [], exSize)
            retstr = ''
        else:
            retstr = str(mat)
            if exSize is None:
                # no exSize-> scalar
                retmat = cvx.matrix(mat, (1, 1), tc=tc)
            elif _is_integer(exSize):
                # exSize is an int -> alpha * identity matrix
                retmat = mat * cvx.spdiag([1.] * exSize)
                retstr += 'I'
            elif isinstance(exSize, tuple):
                # exSize is a tuple -> ones of desired size
                retmat = mat * cvx.matrix(1., exSize)

    elif isinstance(mat, str):
        retstr = mat
        if mat[0] == '-':
            alpha = -1.
            mat = mat[1:]
        else:
            alpha = 1.
        ind = 1
        try:
            while True:
                junk = float(mat[:ind])
                ind += 1
        except Exception:
            ind -= 1
            if ind > 0:
                alpha *= float(mat[:ind])
            mat = mat[ind:]
        transpose = False
        if mat[-2:] == '.T':
            transpose = True
            mat = mat[:-2]
        #|alpha| for a matrix whith all alpha
        #|alpha|(n,m) for a matrix of size (n,m)
        if (mat.find('|') >= 0):
            i1 = mat.find('|')
            i2 = mat.find('|', i1 + 1)
            if i2 < 0:
                raise Exception('There should be a 2d bar')
            if 'j' in mat[i1 + 1:i2]:
                fact = complex(mat[i1 + 1:i2])
                if not fact.imag:
                    fact = fact.real
            else:
                fact = float(mat[i1 + 1:i2])
            i1 = mat.find('(')
            if i1 >= 0:
                i2 = mat.find(')')
                ind = mat[i1 + 1:i2]
                i1 = ind.split(',')[0]
                # checks
                try:
                    i2 = ind.split(',')[1]
                except IndexError:
                    raise Exception('index of |1| should be i,j')
                if not i1.isdigit():
                    raise Exception('first index of |1| should be int')
                if not i2.isdigit():
                    raise Exception('second index of |1| should be int')
                i1 = int(i1)
                i2 = int(i2)
            elif isinstance(exSize, tuple):
                i1, i2 = exSize
            else:
                raise Exception('size unspecified')
            retmat = fact * cvx.matrix(1., (i1, i2))
        # unit vector
        elif (mat.find('e_') >= 0):
            mspl = mat.split('e_')
            if len(mspl[0]) > 0:
                raise NameError('unexpected case')
            mind = mspl[1][:mspl[1].index('(')]
            if (mind.find(',') >= 0):
                idx = mind.split(',')
                idx = (int(idx[0]), int(idx[1]))
            else:
                idx = int(mind)
            i1 = mat.find('(')
            if i1 >= 0:
                i2 = mat.find(')')
                ind = mat[i1 + 1:i2]
                i1 = ind.split(',')[0]
                # checks
                try:
                    i2 = ind.split(',')[1]
                except IndexError:
                    raise Exception('index of e_ should be i,j')
                if not i1.isdigit():
                    raise Exception('first index of e_ should be int')
                if not i2.isdigit():
                    raise Exception('second index of e_ should be int')
                i1 = int(i1)
                i2 = int(i2)
            elif isinstance(exSize, tuple):
                i1, i2 = exSize
            else:
                raise Exception('size unspecified')
            retmat = spmatrix([], [], [], (i1, i2))
            retmat[idx] = 1
        # identity
        elif (mat.startswith('I')):
            if len(mat) > 1 and mat[1] == '(':
                if mat[-1] != ')':
                    raise Exception('this string shlud have the format "I(n)"')
                szstr = mat[2:-1]
                if not(szstr.isdigit()):
                    raise Exception('this string shlud have the format "I(n)"')
                sz = int(szstr)
                if (not exSize is None) and ((_is_integer(exSize) and exSize != sz) or (
                        isinstance(exSize, tuple) and ((exSize[0] != sz) or (exSize[1] != sz)))):
                    raise Exception('exSize does not match the n in "I(n)"')
                exSize = (sz, sz)
                retstr = 'I'
            if exSize is None:
                raise Exception('size unspecified')
            if isinstance(exSize, tuple):
                if exSize[0] != exSize[1]:
                    raise Exception('matrix should be square')
                retmat = cvx.spdiag([1.] * exSize[0])
            else:  # we have an integer
                retmat = cvx.spdiag([1.] * exSize)
        else:
            raise NameError('unexpected mat variable')
        if transpose:
            retmat = retmat.T
        retmat *= alpha
    else:
        raise NameError('unexpected mat variable')

    # make sure it's sparse
    if not isinstance(mat, cvx.base.spmatrix):
        retmat = cvx.sparse(retmat)

    # look for a more appropriate string...
    if retstr is None:
        retstr = '[ {0} x {1} MAT ]'.format(*retmat.size)
    if not retmat:  # |0|
        if retmat.size == (1, 1):
            retstr = '0'
        else:
            retstr = '|0|'
    elif retmat.size == (1, 1):
        retstr = str(retmat[0])
    elif (len(retmat.V) == retmat.size[0] * retmat.size[1]) and not(
            bool(retmat - retmat.V[0])):  # |alpha|
        if retmat[0] == 0:
            retstr = '|0|'
        elif retmat[0] == 1:
            retstr = '|1|'
        else:
            retstr = '|' + str(retmat[0]) + '|'
    elif retmat.I.size[0] == 1:  # e_x
        spm = cvx.sparse(retmat)
        i = spm.I[0]
        j = spm.J[0]
        retstr = ''
        if spm.V[0] != 1:
            retstr = str(spm.V[0]) + '*'
        if retmat.size[1] > 1:
            retstr += 'e_' + str(i) + ',' + str(j)
        else:
            retstr += 'e_' + str(i)
    #(1,1) matrix but not appropriate size
    if retmat.size == (1, 1) and (exSize not in [(1, 1), 1, None]):
        return _retrieve_matrix(retmat[0], exSize)

    return retmat, retstr


def svec(mat, ignore_sym=False):
    """
    returns the svec representation of the cvx matrix ``mat``.
    (see `Dattorro, ch.2.2.2.1 <http://meboo.convexoptimization.com/Meboo.html>`_)

    If ``ignore_sym = False`` (default), the function raises an Exception if ``mat`` is not symmetric.
    Otherwise, elements in the lower triangle of ``mat`` are simply ignored.
    """
    if not isinstance(mat, cvx.spmatrix):
        mat = cvx.sparse(mat)

    s0 = mat.size[0]
    if s0 != mat.size[1]:
        raise ValueError('mat must be square')

    I = []
    J = []
    V = []
    for (i, j, v) in zip((mat.I), (mat.J), (mat.V)):
        if not ignore_sym:
            if abs(mat[j, i] - v) > 1e-6:
                raise ValueError('mat must be symmetric')
        if i <= j:
            isvec = j * (j + 1) // 2 + i
            J.append(0)
            I.append(isvec)
            if i == j:
                V.append(v)
            else:
                V.append(np.sqrt(2) * v)

    return spmatrix(V, I, J, (s0 * (s0 + 1) // 2, 1))


def svecm1(vec, triu=False):
    if vec.size[1] > 1:
        raise ValueError('should be a column vector')
    v = vec.size[0]
    n = int(np.sqrt(1 + 8 * v) - 1) // 2
    if n * (n + 1) // 2 != v:
        raise ValueError('vec should be of dimension n(n+1)/2')
    if not isinstance(vec, cvx.spmatrix):
        vec = cvx.sparse(vec)
    I = []
    J = []
    V = []
    for i, v in zip(vec.I, vec.V):
        c = int(np.sqrt(1 + 8 * i) - 1) // 2
        r = i - c * (c + 1) // 2
        I.append(r)
        J.append(c)
        if r == c:
            V.append(v)
        else:
            if triu:
                V.append(v / np.sqrt(2))
            else:
                I.append(c)
                J.append(r)
                V.extend([v / np.sqrt(2)] * 2)
    return spmatrix(V, I, J, (n, n))


def ltrim1(vec, uptri=True,offdiag_fact=1.):
    """
    If ``vec`` is a vector or an affine expression of size n(n+1)/2, ltrim1(vec) returns a (n,n) matrix with
    the elements of vec in the lower triangle.
    If ``uptri == False``, the upper triangle is 0, otherwise the upper triangle is the symmetric of the lower one.
    """
    if vec.size[1] > 1:
        raise ValueError('should be a column vector')
    from .expression import AffinExp
    v = vec.size[0]
    n = int(np.sqrt(1 + 8 * v) - 1) // 2
    if n * (n + 1) // 2 != v:
        raise ValueError('vec should be of dimension n(n+1)/2')
    if isinstance(vec, cvx.matrix) or isinstance(vec, cvx.spmatrix):
        if not isinstance(vec, cvx.matrix):
            vec = cvx.matrix(vec)
        M = cvx.matrix(0., (n, n))
        r = 0
        c = 0
        for v in vec:
            if r == n:
                c += 1
                r = c
            if r!=c:
                v *= offdiag_fact
            M[r, c] = v
            if r > c and uptri:
                M[c, r] = v
            r += 1

        return M
    elif isinstance(vec, AffinExp):
        I, J, V = [], [], []
        r = 0
        c = 0
        for i in range(v):
            if r == n:
                c += 1
                r = c
            I.append(r + n * c)
            J.append(i)
            V.append(1)
            if r > c and uptri:
                I.append(c + n * r)
                J.append(i)
                V.append(1)
            r += 1
        H = spmatrix(V, I, J, (n**2, v))
        Hvec = H * vec
        newfacs = Hvec.factors
        newcons = Hvec.constant
        if uptri:
            return AffinExp(newfacs, newcons, (n, n),
                            'ltrim1_sym(' + vec.string + ')')
        else:
            return AffinExp(newfacs, newcons, (n, n),
                            'ltrim1(' + vec.string + ')')
    else:
        raise Exception('expected a cvx vector or an affine expression')


def lowtri(exp):
    r"""
    if ``exp`` is a square affine expression of size (n,n),
    ``lowtri(exp)`` returns the (n(n+1)/2)-vector of the lower triangular elements of ``exp``.

    **Example**

    >>> import picos as pic
    >>> import cvxopt as cvx
    >>> prob=pic.Problem()
    >>> X=prob.add_variable('X',(4,4),'symmetric')
    >>> pic.tools.lowtri(X)
    # (10 x 1)-affine expression: lowtri(X) #
    >>> X0 = cvx.matrix(range(16),(4,4))
    >>> X.value = X0 * X0.T
    >>> print X#doctest: +NORMALIZE_WHITESPACE
    [ 2.24e+02  2.48e+02  2.72e+02  2.96e+02]
    [ 2.48e+02  2.76e+02  3.04e+02  3.32e+02]
    [ 2.72e+02  3.04e+02  3.36e+02  3.68e+02]
    [ 2.96e+02  3.32e+02  3.68e+02  4.04e+02]
    >>> print pic.tools.lowtri(X)#doctest: +NORMALIZE_WHITESPACE
    [ 2.24e+02]
    [ 2.48e+02]
    [ 2.72e+02]
    [ 2.96e+02]
    [ 2.76e+02]
    [ 3.04e+02]
    [ 3.32e+02]
    [ 3.36e+02]
    [ 3.68e+02]
    [ 4.04e+02]
    """
    if exp.size[0] != exp.size[1]:
        raise ValueError('exp must be square')
    from .expression import AffinExp
    from itertools import izip
    if not isinstance(exp, AffinExp):
        mat, name = _retrieve_matrix(exp)
        exp = AffinExp({}, constant=mat[:], size=mat.size, string=name)
    (n, m) = exp.size
    newfacs = {}
    newrow = {}  # dict of new row indices
    nr = 0
    for i in range(n**2):
        col = i // n
        row = i % n
        if row >= col:
            newrow[i] = nr
            nr += 1
    nsz = nr  # this should be (n*(n+1))/2
    for var, mat in six.iteritems(exp.factors):
        I, J, V = [], [], []
        for i, j, v in izip(mat.I, mat.J, mat.V):
            col = i // n
            row = i % n
            if row >= col:
                I.append(newrow[i])
                J.append(j)
                V.append(v)
        newfacs[var] = spmatrix(V, I, J, (nr, mat.size[1]))
    if exp.constant is None:
        newcons = None

    else:
        ncs = []
        for i, v in enumerate(cvx.matrix(exp.constant)):
            col = i // n
            row = i % n
            if row >= col:
                ncs.append(v)
        newcons = cvx.matrix(ncs, (nr, 1))
    return AffinExp(newfacs, newcons, (nr, 1), 'lowtri(' + exp.string + ')')


def _utri(mat):
    """
    return elements of the (strict) upper triangular part of a cvxopt matrix
    """
    m, n = mat.size
    if m != n:
        raise ValueError('mat must be square')
    v = []
    for j in range(1, n):
        for i in range(j):
            v.append(mat[i, j])
    return cvx.sparse(v)


def _svecm1_identity(vtype, size):
    """
    row wise svec-1 transformation of the
    identity matrix of size size[0]*size[1]
    """
    if vtype in ('symmetric',):
        s0 = size[0]
        if size[1] != s0:
            raise ValueError('should be square')
        I = range(s0 * s0)
        J = []
        V = []
        for i in I:
            rc = (i % s0, i // s0)
            (r, c) = (min(rc), max(rc))
            j = c * (c + 1) // 2 + r
            J.append(j)
            if r == c:
                V.append(1)
            else:
                V.append(1 / np.sqrt(2))
        idmat = spmatrix(V, I, J, (s0 * s0, s0 * (s0 + 1) // 2))
    elif vtype == 'antisym':
        s0 = size[0]
        if size[1] != s0:
            raise ValueError('should be square')
        I = []
        J = []
        V = []
        k = 0
        for j in range(1, s0):
            for i in range(j):
                I.append(s0 * j + i)
                J.append(k)
                V.append(1)
                I.append(s0 * i + j)
                J.append(k)
                V.append(-1)
                k += 1
        idmat = spmatrix(V, I, J, (s0 * s0, s0 * (s0 - 1) // 2))
    else:
        sp = size[0] * size[1]
        idmat = spmatrix([1] * sp, range(sp), range(sp), (sp, sp))

    return idmat


def new_param(name, value):
    """
    Declare a parameter for the problem, that will be stored
    as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
    It is possible to give a list or a dictionary of parameters.
    The function returns a constant :class:`AffinExp <picos.AffinExp>`
    (or a ``list`` or a ``dict`` of :class:`AffinExp <picos.AffinExp>`) representing this parameter.

    .. note :: Declaring parameters is optional, since the expression can
                    as well be given by using normal variables. (see Example below).
                    However, if you use this function to declare your parameters,
                    the names of the parameters will be displayed when you **print**
                    an :class:`Expression <picos.Expression>` or a :class:`Constraint <picos.Constraint>`

    :param name: The name given to this parameter.
    :type name: str.
    :param value: The value (resp ``list`` of values, ``dict`` of values) of the parameter.
                    The type of **value** (resp. the elements of the ``list`` **value**,
                    the values of the ``dict`` **value**) should be understandable by
                    the function :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`.
    :returns: A constant affine expression (:class:`AffinExp <picos.AffinExp>`)
                    (resp. a ``list`` of :class:`AffinExp <picos.AffinExp>` of the same length as **value**,
                    a ``dict`` of :class:`AffinExp <picos.AffinExp>` indexed by the keys of **value**)

    **Example:**

    >>> import cvxopt as cvx
    >>> prob=pic.Problem()
    >>> x=prob.add_variable('x',3)
    >>> B={'foo':17.4,'matrix':cvx.matrix([[1,2],[3,4],[5,6]]),'ones':'|1|(4,1)'}
    >>> B['matrix']*x+B['foo']
    # (2 x 1)-affine expression: [ 2 x 3 MAT ]*x + |17.4| #
    >>> #(in the string above, |17.4| represents the 2-dim vector [17.4,17.4])
    >>> B=pic.new_param('B',B)
    >>> #now that B is a param, we have a nicer display:
    >>> B['matrix']*x+B['foo']
    # (2 x 1)-affine expression: B[matrix]*x + |B[foo]| #
    """
    from .expression import AffinExp
    if isinstance(value, list):
        if all([_is_numeric(x) for x in value]):
            # list with numeric data
            term, termString = _retrieve_matrix(value, None)
            return AffinExp({}, constant=term[:], size=term.size, string=name)
        elif (all([isinstance(x, list) for x in value]) and
              all([len(x) == len(value[0]) for x in value]) and
              all([_is_realvalued(xi) for x in value for xi in x])
              ):
            # list of numeric lists of the same length
            sz = len(value), len(value[0])
            term, termString = _retrieve_matrix(value, sz)
            return AffinExp({}, constant=term[:], size=term.size, string=name)
        else:
            L = []
            for i, l in enumerate(value):
                L.append(new_param(name + '[' + str(i) + ']', l))
            return L
    elif isinstance(value, tuple):
        # handle as lists, but ignores numeric list and tables (vectors or
        # matrices)
        L = []
        for i, l in enumerate(value):
            L.append(new_param(name + '[' + str(i) + ']', l))
        return L
    elif isinstance(value, dict):
        D = {}
        for k in value.keys():
            D[k] = new_param(name + '[' + str(k) + ']', value[k])
        return D
    else:
        term, termString = _retrieve_matrix(value, None)
        return AffinExp({}, constant=term[:], size=term.size, string=name)


def available_solvers():
    """Lists all available solvers"""
    lst = []
    try:
        import cvxopt as co
        lst.append('cvxopt')
        del co
    except ImportError:
        pass
    try:
        import swiglpk as gl
        lst.append('glpk')
        del gl
    except ImportError:
        pass
    try:
        import smcp as sm
        lst.append('smcp')
        del sm
    except ImportError:
        pass
    try:
        import mosek7 as mo7
        lst.append('mosek7')
        del mo7
        try:
            import mosek as mo
            version7 = not(hasattr(mo, 'cputype'))
            if not version7:
                lst.append('mosek6')
            del mo
        except ImportError:
            pass
    except ImportError:  # only one default mosek available
        try:
            import mosek as mo
            # True if this is the beta version 7 of MOSEK
            version7 = not(hasattr(mo, 'cputype'))
            del mo
            if version7:
                lst.append('mosek7')
            else:
                lst.append('mosek6')
        except ImportError:
            pass
    try:
        import cplex as cp
        lst.append('cplex')
        del cp
    except (ImportError, SyntaxError):
        pass
    try:
        import pyscipopt as zo
        lst.append('zibopt')
        del zo
    except ImportError:
        pass
    try:
        import gurobipy as grb
        lst.append('gurobi')
        del grb
    except ImportError:
        pass
    # TRICK to force mosek6 during tests
    # if 'mosek7' in lst:
    #        lst.remove('mosek7')
    try:
        sdpa_executable = "sdpa"

        def which(program):
            import os

            def is_exe(fpath):
                return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

            fpath, fname = os.path.split(program)
            if fpath:
                if is_exe(program):
                    return program
            else:
                for path in os.environ["PATH"].split(os.pathsep):
                    path = path.strip('"')
                    exe_file = os.path.join(path, program)
                    if is_exe(exe_file):
                        return exe_file

            return None

        if which(sdpa_executable) is None:
            raise ImportError('sdpa not installed')

        lst.append('sdpa')

    except:
        pass

    return lst


def offset_in_lil(lil, offset, lower):
    """
    substract the ``offset`` from all elements of the
    (recursive) list of lists ``lil``
    which are larger than ``lower``.
    """
    for i, l in enumerate(lil):
        if _is_integer(l):
            if l > lower:
                lil[i] -= offset
        elif isinstance(l, list):
            lil[i] = offset_in_lil(l, offset, lower)
        else:
            raise Exception('elements of lil must be int or list')
    return lil


def import_cbf(filename):
    """
    Imports the data from a CBF file, and creates a :class:`Problem` object.

    The created problem contains one (multidimmensional) variable
    for each cone specified in the section ``VAR`` of the .cbf file,
    and one (multidimmensional) constraint for each cone
    specified in the sections ``CON`` and ``PSDCON``.

    Semidefinite variables defined in the section ``PSDVAR`` of the .cbf file
    are represented by a matrix picos variable ``X`` with ``X.vtype='symmetric'``.

    This function returns a tuple ``(P,x,X,data)``,
    where:

     * ``P`` is the imported picos :class:`Problem` object.
     * ``x`` is a list of :class:`Variable` objects, representing the (multidimmensional) scalar variables.
     * ``X`` is a list of :class:`Variable` objects, representing the symmetric semidefinite positive variables.
     * ``data`` is a dictionary containing picos parameters (:class:`AffinExp` objects) used
       to define the problem. Indexing is with respect to the blocks of variables as defined
       in thes sections ``VAR`` and  ``CON`` of the .cbf file.

    """
    from .problem import Problem
    P = Problem()
    x, X, data = P._read_cbf(filename)
    return (P, x, X, data)


def _flatten(l):
    """ flatten a (recursive) list of list """
    for el in l:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            for sub in _flatten(el):
                yield sub
        else:
            yield el


def _remove_in_lil(lil, elem):
    """ remove the element ``elem`` from a (recursive) list of list ``lil``.
        empty lists are removed if any"""
    if elem in lil:
        lil.remove(elem)
    for el in lil:
        if isinstance(el, list):
            _remove_in_lil(el, elem)
            _remove_in_lil(el, [])
    if [] in lil:
        lil.remove([])


def _quad2norm(qd):
    """
    transform the list of bilinear terms qd
    in an equivalent squared norm
    (x.T Q x) -> ||Q**0.5 x||**2
    """
    # find all variables
    qdvars = []
    for xy in qd:
        p1 = (xy[0].startIndex, xy[0])
        p2 = (xy[1].startIndex, xy[1])
        if p1 not in qdvars:
            qdvars.append(p1)
        if p2 not in qdvars:
            qdvars.append(p2)
    # sort by start indices
    qdvars = sorted(qdvars)
    qdvars = [v for (i, v) in qdvars]
    offsets = {}
    ofs = 0
    for v in qdvars:
        offsets[v] = ofs
        ofs += v.size[0] * v.size[1]

    # construct quadratic matrix
    Q = spmatrix([], [], [], (ofs, ofs))
    I, J, V = [], [], []
    for (xi, xj), Qij in six.iteritems(qd):
        oi = offsets[xi]
        oj = offsets[xj]
        Qtmp = spmatrix(Qij.V, Qij.I + oi, Qij.J + oj, (ofs, ofs))
        Q += 0.5 * (Qtmp + Qtmp.T)
    # cholesky factorization V.T*V=Q
    # remove zero rows and cols
    nz = set(Q.I)
    P = spmatrix(1., range(len(nz)), list(nz), (len(nz), ofs))
    Qp = P * Q * P.T
    try:
        import cvxopt.cholmod
        F = cvxopt.cholmod.symbolic(Qp)
        cvxopt.cholmod.numeric(Qp, F)
        Z = cvxopt.cholmod.spsolve(F, Qp, 7)
        V = cvxopt.cholmod.spsolve(F, Z, 4)
        V = V * P
    except ArithmeticError:  # Singular or Non-convex, we must work on the dense matrix
        import cvxopt.lapack
        sig = cvx.matrix(0., (len(nz), 1), tc='z')
        U = cvx.matrix(0., (len(nz), len(nz)))
        cvxopt.lapack.gees(cvx.matrix(Qp), sig, U)
        sig = sig.real()
        if min(sig) < -1e-7:
            raise NonConvexError('I cannot convert non-convex quads to socp')
        for i in range(len(sig)):
            sig[i] = max(sig[i], 0)
        V = cvx.spdiag(sig**0.5) * U.T
        V = cvx.sparse(V) * P
    allvars = qdvars[0]
    for v in qdvars[1:]:
        if v.size[1] == 1:
            allvars = allvars // v
        else:
            allvars = allvars // v[:]
    return abs(V * allvars)**2


def _copy_dictexp_to_new_vars(dct, cvars, complex=None):
    # cf function _copy_exp_to_new_vars for an explanation of the 'complex'
    # argument
    D = {}
    import copy
    for var, value in six.iteritems(dct):
        if isinstance(var, tuple):  # quad
            if var[0].vtype == 'hermitian' or var[1].vtype == 'hermitian':
                raise Exception('quadratic form involving hermitian variable')
            D[cvars[var[0].name], cvars[var[1].name]] = copy.copy(value)
        else:
            if complex is None:
                D[cvars[var.name]] = copy.copy(value)
                continue

            if var.vtype == 'hermitian' and (var.name + '_RE') in cvars:

                n = int(value.size[1]**(0.5))
                idasym = _svecm1_identity('antisym', (n, n))

                if value.typecode == 'z':
                    vr = value.real()
                    D[cvars[var.name + '_IM_utri']] = -value.imag() * idasym
                    #!BUG corrected, in previous version (1.0.1)
                    #"no minus because value.imag()=-value.H.imag()" ???
                    # But maybe an other error was cancelling this bug...
                else:
                    vr = value
                    if complex:
                        D[cvars[var.name + '_IM_utri']] = spmatrix(
                            [], [], [],
                            (vr.size[0], cvars[var.name + '_IM_utri'].size[0]))

                vv = []
                for i in range(vr.size[0]):
                    v = vr[i, :]
                    AA = cvx.matrix(v, (n, n))
                    AA = (AA + AA.T) * 0.5  # symmetrize
                    vv.append(svec(AA).T)
                D[cvars[var.name + '_RE']] = cvx.sparse(vv)
                if complex:
                    # compute the imaginary part and append it.
                    if value.typecode == 'z':
                        Him = value.real()
                        vi = value.imag()
                    else:
                        Him = copy.copy(value)
                        vi = spmatrix([], [], [], Him.size)

                    n = int(vi.size[1]**(0.5))
                    vv = []
                    for i in range(vi.size[0]):
                        v = vi[i, :]
                        BB = cvx.matrix(v, (n, n))
                        BB = (BB + BB.T) * 0.5  # symmetrize
                        vv.append(svec(BB).T)
                    Hre = cvx.sparse(vv)

                    D[cvars[var.name + '_RE']
                      ] = cvx.sparse([D[cvars[var.name + '_RE']], Hre])
                    D[cvars[var.name + '_IM_utri']
                      ] = cvx.sparse([D[cvars[var.name + '_IM_utri']], Him * idasym])

            else:
                if value.typecode == 'z':
                    vr = value.real()
                    vi = value.imag()
                else:
                    vr = copy.copy(value)
                    vi = spmatrix([], [], [], vr.size)
                if complex:
                    D[cvars[var.name]] = cvx.sparse([vr, vi])
                else:
                    D[cvars[var.name]] = vr
    return D


def _copy_exp_to_new_vars(exp, cvars, complex=None):
    # if complex=None (default), the expression is copied "as is"
    # if complex=False, the exp is assumed to be real_valued and
    #                  only the real part is copied to the new expression)
    # otherwise (complex=True), a new expression is created, which concatenates horizontally
    #           the real and the imaginary part
    from .expression import Variable, AffinExp, Norm, LogSumExp, QuadExp, GeneralFun, GeoMeanExp, NormP_Exp, TracePow_Exp, DetRootN_Exp
    import copy
    if isinstance(exp, Variable):
        if exp.vtype == 'hermitian':  # handle as AffinExp
            return _copy_exp_to_new_vars('I' * exp, cvars, complex=complex)
        return cvars[exp.name]
    elif isinstance(exp, AffinExp):
        newfacs = _copy_dictexp_to_new_vars(
            exp.factors, cvars, complex=complex)
        if exp.constant is None:
            v = spmatrix([], [], [], (exp.size[0] * exp.size[1], 1))
        else:
            v = exp.constant
        if complex is None:
            newcons = copy.copy(v)
            newsize = exp.size
        elif complex:
            if v.typecode == 'z':
                vi = v.imag()
            else:
                vi = spmatrix([], [], [], v.size)
            newcons = cvx.sparse([v.real(), vi])
            newsize = (exp.size[0], 2 * exp.size[1])
        else:
            newcons = v.real()
            newsize = exp.size
        return AffinExp(newfacs, newcons, newsize, exp.string)
    elif isinstance(exp, Norm):
        newexp = _copy_exp_to_new_vars(exp.exp, cvars, complex=complex)
        return Norm(newexp)
    elif isinstance(exp, LogSumExp):
        newexp = _copy_exp_to_new_vars(exp.Exp, cvars, complex=complex)
        return LogSumExp(newexp)
    elif isinstance(exp, QuadExp):
        newaff = _copy_exp_to_new_vars(exp.aff, cvars, complex=complex)
        newqds = _copy_dictexp_to_new_vars(exp.quad, cvars, complex=complex)
        if exp.LR is None:
            return QuadExp(newqds, newaff, exp.string, None)
        else:
            LR0 = _copy_exp_to_new_vars(exp.LR[0], cvars, complex=complex)
            LR1 = _copy_exp_to_new_vars(exp.LR[1], cvars, complex=complex)
            return QuadExp(newqds, newaff, exp.string, (LR0, LR1))
    elif isinstance(exp, GeneralFun):
        newexp = _copy_exp_to_new_vars(exp.Exp, cvars, complex=complex)
        return LogSumExp(exp.fun, newexp, exp.funstring)
    elif isinstance(exp, GeoMeanExp):
        newexp = _copy_exp_to_new_vars(exp.exp, cvars, complex=complex)
        return GeoMeanExp(newexp)
    elif isinstance(exp, NormP_Exp):
        newexp = _copy_exp_to_new_vars(exp.exp, cvars, complex=complex)
        return NormP_Exp(newexp, exp.numerator, exp.denominator)
    elif isinstance(exp, TracePow_Exp):
        newexp = _copy_exp_to_new_vars(exp.exp, cvars, complex=complex)
        return TracePow_Exp(newexp, exp.numerator, exp.denominator)
    elif isinstance(exp, DetRootN_Exp):
        newexp = _copy_exp_to_new_vars(exp.exp, cvars, complex=complex)
        return DetRootN_Exp(newexp)
    elif exp is None:
        return None
    else:
        raise Exception('unknown type of expression')


def _cplx_mat_to_real_mat(M):
    """
    if M = A +iB,
    return the block matrix [A,-B;B,A]
    """
    if not(isinstance(M, cvx.base.spmatrix) or isinstance(M, cvx.base.matrix)):
        raise NameError('unexpected matrix type')
    if M.typecode == 'z':
        A = M.real()
        B = M.imag()
    else:
        A = M
        B = spmatrix([], [], [], A.size)
    return cvx.sparse([[A, B], [-B, A]])


def _cplx_vecmat_to_real_vecmat(M, sym=True, times_i=False):
    """
    if the columns of M are vectorizations of matrices of the form A +iB:
    * if times_i is False (default), return vectorizations of the block matrix [A,-B;B,A]
      otherwise, return vectorizations of the block matrix [-B,-A;A,-B]
    * if sym=True, returns the columns with respect to the sym-vectorization of the variables of the LMI
    """
    if not(isinstance(M, cvx.base.spmatrix) or isinstance(M, cvx.base.matrix)):
        raise NameError('unexpected matrix type')

    if times_i:
        M = M * 1j

    mm = M.size[0]
    m = mm**0.5
    if int(m) != m:
        raise NameError('first dimension must be a perfect square')
    m = int(m)

    vv = []
    if sym:
        nn = M.size[1]
        n = nn**0.5
        if int(n) != n:
            raise NameError('2d dimension must be a perfect square')
        n = int(n)

        for k in range(n * (n + 1) // 2):
            j = int(np.sqrt(1 + 8 * k) - 1) // 2
            i = k - j * (j + 1) // 2
            if i == j:
                v = M[:, n * i + i]
            else:
                i1 = n * i + j
                i2 = n * j + i
                v = (M[:, i1] + M[:, i2]) * (1. / (2**0.5))
            vvv = _cplx_mat_to_real_mat(cvx.matrix(v, (m, m)))[:]
            vv.append([vvv])

    else:
        for i in range(M.size[1]):
            v = M[:, i]
            A = cvx.matrix(v, (m, m))
            vvv = _cplx_mat_to_real_mat(A)[:]  # TODO 0.5*(A+A.H) instead ?
            vv.append([vvv])

    return cvx.sparse(vv)


def _is_idty(mat, vtype='continuous'):
    if vtype == 'continuous':
        if (mat.size[0] == mat.size[1]):
            n = mat.size[0]
            if (list(mat.I) == list(range(n)) and
                    list(mat.J) == list(range(n)) and
                    list(mat.V) == [1.] * n):
                return True
    elif vtype == 'antisym':
        n = int((mat.size[0])**0.5)
        if n != int(n) or n * (n - 1) // 2 != mat.size[1]:
            return False
        if not (_svecm1_identity('antisym', (n, n)) - mat):
            return True
    return False


def _is_integer(x):
    return (isinstance(x,six.integer_types) or
            isinstance(x,np.int64) or
            isinstance(x,np.int32))

def _is_numeric(x):
    return (isinstance(x, float) or
            isinstance(x, six.integer_types) or
            isinstance(x, np.float64) or
            isinstance(x, np.int64) or
            isinstance(x,np.int32) or
            isinstance(x, np.complex128) or
            isinstance(x, complex))

def _is_realvalued(x):
    return (isinstance(x, float) or
            isinstance(x, six.integer_types) or
            isinstance(x, np.float64) or
            isinstance(x, np.int64) or
            isinstance(x,np.int32))

def spmatrix(*args,**kwargs):
    try:
        return cvx.spmatrix(*args,**kwargs)
    except TypeError as ex:
        print(type(ex))
        print(str(ex))
        print('non-numeric' in str(ex))
        if 'non-numeric' in str(ex):#catch exception with int64 bug of cvxopt
            size_tc = {}

            if len(args)>0:
                V = args[0]
            elif 'V' in kwargs:
                V = kwargs['V']
            else:
                V = []
            if len(args)>1:
                I = args[1]
            elif 'I' in kwargs:
                I = kwargs['I']
            else:
                I = []
            if len(args)>2:
                J = args[2]
            elif 'J' in kwargs:
                J = kwargs['J']
            else:
                J = []
            if len(args) > 3:
                size_tc['size'] = args[3]
            elif 'size' in kwargs:
                size_tc['size'] = kwargs['size']
            if len(args) > 4:
                size_tc['tc'] = args[4]
            elif 'tc' in kwargs:
                size_tc['tc'] = kwargs['tc']
            return cvx.spmatrix(V, [int(i) for i in I], [int(j) for j in J],**size_tc)
        else:
            raise
    except Exception as ex:
        print(type(ex))
        import pdb;pdb.set_trace()


def kron(A,B):
    """
    Kronecker product of 2 expression, at least one of which must be constant

    **Example:**

    >>> import picos as pic
    >>> import cvxopt as cvx
    >>> P = pic.Problem()
    >>> X = P.add_variable('X',(4,3))
    >>> X.value = cvx.matrix(range(12),(4,3))
    >>> I = pic.new_param('I',np.eye(2))
    >>> print pic.kron(I,X) #doctest: +NORMALIZE_WHITESPACE
    [ 0.00e+00  4.00e+00  8.00e+00  0.00e+00  0.00e+00  0.00e+00]
    [ 1.00e+00  5.00e+00  9.00e+00  0.00e+00  0.00e+00  0.00e+00]
    [ 2.00e+00  6.00e+00  1.00e+01  0.00e+00  0.00e+00  0.00e+00]
    [ 3.00e+00  7.00e+00  1.10e+01  0.00e+00  0.00e+00  0.00e+00]
    [ 0.00e+00  0.00e+00  0.00e+00  0.00e+00  4.00e+00  8.00e+00]
    [ 0.00e+00  0.00e+00  0.00e+00  1.00e+00  5.00e+00  9.00e+00]
    [ 0.00e+00  0.00e+00  0.00e+00  2.00e+00  6.00e+00  1.00e+01]
    [ 0.00e+00  0.00e+00  0.00e+00  3.00e+00  7.00e+00  1.10e+01]

    """

    from .expression import AffinExp

    if not isinstance(A,AffinExp):
        expA, nameA = _retrieve_matrix(A)
    else:
        expA, nameA = A,A.string

    if not isinstance(B,AffinExp):
        expB, nameB = _retrieve_matrix(B)
    else:
        expB, nameB = B,B.string

    if expA.isconstant():
        AA = np.array(cvx.matrix(expA.value))
        kron_fact = {}
        for x, Bx in six.iteritems(expB.factors):
            #Blst contains matrix such that B=\sum x_i B_i (+constant)
            Blst = []
            AkronB = []
            for k in range(Bx.size[1]):
                Blst.append(np.reshape(cvx.matrix(Bx[:,k]),expB.size[::-1]).T)
                AkronB.append(np.kron(AA,Blst[-1]))
            kron_fact[x] = cvx.sparse([list(AkronB[k].T.ravel()) for k in range(Bx.size[1])])
        kron_cons = None
        if expB.constant:
            Bcons = np.reshape(cvx.matrix(expB.constant),expB.size[::-1]).T
            AkronB = np.kron(AA,Bcons)
            kron_cons = cvx.sparse(list(AkronB.T.ravel()))

        kron_size = (expA.size[0] * expB.size[0], expA.size[1] * expB.size[1])
        kron_string =  nameA+'⊗ ('+nameB+')'
    elif expB.isconstant():
        BB = np.array(cvx.matrix(expB.value))
        kron_fact = {}
        for x, Ax in six.iteritems(expA.factors):
            #Blst contains matrix such that B=\sum x_i B_i (+constant)
            Alst = []
            AkronB = []
            for k in range(Ax.size[1]):
                Alst.append(np.reshape(cvx.matrix(Ax[:,k]),expA.size[::-1]).T)
                AkronB.append(np.kron(Alst[-1],BB))
            kron_fact[x] = cvx.sparse([list(AkronB[k].T.ravel()) for k in range(Ax.size[1])])
        kron_cons = None
        if expA.constant:
            Acons = np.reshape(cvx.matrix(expA.constant),expA.size[::-1]).T
            AkronB = np.kron(Acons,BB)
            kron_cons = cvx.sparse(list(AkronB.T.ravel()))

        kron_size = (expA.size[0] * expB.size[0], expA.size[1] * expB.size[1])
        kron_string =  '('+ nameA+') ⊗ '+nameB
    else:
        raise NotImplementedError('kron product with quadratic terms')
    return AffinExp(kron_fact,constant=kron_cons,size=kron_size,string=kron_string)




def _read_sdpa(filename):
    """TODO, remove dependence; currently relies on smcp sdpa_read
    cone constraints ||x||<t are recognized if they have the arrow form [t,x';x,t*I]>>0
    """
    f = open(filename, 'r')
    import smcp
    from .problem import Problem
    A, bb, blockstruct = smcp.misc.sdpa_read(f)

    # make list of tuples (size,star,end) for each block
    blocks = []
    ix = 0
    for b in blockstruct:
        blocks.append((b, ix, ix + b))
        ix += b

    P = Problem()
    nn = A.size[1] - 1
    x = P.add_variable('x', nn)

    # linear (in)equalities
    Aineq = []
    bineq = []
    Aeq = []
    beq = []
    mm = int((A.size[0])**0.5)
    oldv = None
    for (_, i, _) in [(sz, start, end)
                      for (sz, start, end) in blocks if sz == 1]:
        ix = i * (mm + 1)
        v = A[ix, :]
        if (oldv is not None) and np.linalg.norm((v + oldv).V) < 1e-7:
            Aineq.pop()
            bineq.pop()
            Aeq.append(v[1:].T)
            beq.append(v[0])
        else:
            Aineq.append(v[1:].T)
            bineq.append(v[0])
        oldv = v
    Aeq = cvx.sparse(Aeq)
    Aineq = cvx.sparse(Aineq)
    if Aeq:
        P.add_constraint(Aeq * x == beq)
    if Aineq:
        P.add_constraint(Aineq * x > bineq)

    # sdp and soc constraints
    for (sz, start, end) in [(sz, start, end)
                             for (sz, start, end) in blocks if sz > 1]:
        MM = [cvx.sparse(cvx.matrix(A[:, i], (mm, mm)))[
            start:end, start:end] for i in range(nn + 1)]

        # test whether it is an arrow pattern
        isarrow = True
        issoc = True
        isrsoc = True
        for M in MM:
            for i, j in zip(M.I, M.J):
                if i > j and j > 0:
                    isarrow = False
                    break
            isrsoc = isrsoc and isarrow and all(
                [M[i, i] == M[1, 1] for i in range(2, sz)])
            issoc = issoc and isrsoc and (M[1, 1] == M[0, 0])
            if not(isrsoc):
                break

        if issoc or isrsoc:
            Acone = cvx.sparse([M[:, 0].T for M in MM[1:]]).T
            bcone = MM[0][:, 0]
            if issoc:
                P.add_constraint(
                    abs(Acone[1:, :] * x - bcone[1:]) < Acone[0, :] * x - bcone[0])
            else:
                arcone = cvx.sparse([M[1, 1] for M in MM[1:]]).T
                brcone = MM[0][1, 1]
                P.add_constraint(abs(Acone[1:, :] *
                                     x -
                                     bcone[1:])**2 < (Acone[0, :] *
                                                      x -
                                                      bcone[0]) *
                                 (arcone *
                                  x -
                                  brcone))

        else:
            CCj = MM[0] + MM[0].T - \
                cvx.spdiag([MM[0][i, i] for i in range(sz)])
            MMj = [M + M.T - cvx.spdiag([M[i, i]
                                         for i in range(sz)]) for M in MM[1:]]
            P.add_constraint(sum([x[i] * MMj[i]
                                  for i in range(nn) if MMj[i]], 'i') >> CCj)

    # objective
    P.set_objective('min', bb.T * x)
    return P


def flow_Constraint(
        G,
        f,
        source,
        sink,
        flow_value,
        capacity=None,
        graphName=''):
    """Returns an object of the class :class:`_Flow_Constraint <picos.Constraint._Flow_Constraint>` that can be passed to a
    problem with :func:`add_constraint() <picos.Problem.add_constraint>`.

            ``G`` a directed graph (class DiGraph of `networkx <http://networkx.lanl.gov/index.html>`_)

            ``f`` must be a dictionary of variables indexed by the edges of ``G``

            ``source`` can be eiter a node of ``G``, or a list of nodes in case of a multisource-single sink flow

            ``sink`` can be eiter a node of ``G``, or a list of nodes in case of a single source-multising flow

            ``flow_value`` is the value of the flow, or a list of values in case of a single source - multisink flow. In the latter case,
            the values represent the demands of each sink (resp. of each source for a multisource - single sink flow). The values
            can be either constants or :class:`AffinExp <picos.AffinExp>`.

            ``capacity`` must be either ``None`` or a string. If this is a string, it indicates the key of the edge
            dictionaries of ``G`` that is used for the capacity of the links. Otherwise, edges have an unbounded capacity.

            ``graphName`` is a string used in the string representation of the constraint.


    """
    # checking that we have the good number of variables
    if len(f) != len(G.edges()):
        print('Error: The number of variables does not match with the number of edges.')
        return False

    from .problem import Problem
    Ptmp = Problem()

    if not capacity is None:
        # Adding Edge capacities
        cap = [ed[2][capacity] for ed in G.edges(data=True)]

        c = {}
        for i, e in enumerate(G.edges()):
            c[e] = cap[i]

        cc = new_param('c', c)

        # Adding the capacity constraint
        Ptmp.add_list_of_constraints([f[e] < cc[e]
                                      for e in G.edges()], [('e', 2)], 'edges')

    # nonnegativity of the flows
    Ptmp.add_list_of_constraints(
        [f[e] > 0 for e in G.edges()], [('e', 2)], 'edges')

    #
    # One Source, One Sink
    #
    if not isinstance(source, list) and not isinstance(sink, list):
        # Adding the flow conservation
        Ptmp.add_list_of_constraints([sum([f[p, i] for p in G.predecessors(i)], 'p', 'pred(i)') == sum(
            [f[i, j] for j in G.successors(i)], 'j', 'succ(i)') for i in G.nodes() if i != sink and i != source], 'i', 'nodes-(s,t)')

        # Source flow at S
        Ptmp.add_constraint(sum([f[p,
                                   source] for p in G.predecessors(source)],
                                'p',
                                'pred(s)') + flow_value == sum([f[source,
                                                                  j] for j in G.successors(source)],
                                                               'j',
                                                               'succ(s)'))

        # this constraint was redundant
        # Sink flow at T
        #Ptmp.add_constraint(sum([f[p,sink] for p in G.predecessors(sink)],'p','pred(t)') == sum([f[sink,j] for j in G.successors(sink)],'j','succ(t)') + flow_value)

        if hasattr(flow_value, 'string'):
            fv = flow_value.string
        else:
            fv = str(flow_value)

        if graphName == '':
            comment = "Flow conservation from " + \
                str(source) + " to " + str(sink) + " with value " + fv
        else:
            comment = "Flow conservation in " + \
                str(graphName) + " from " + str(source) + " to " + str(sink) + " with value " + fv

    #
    # One Source, Multiple sink
    #
    elif not isinstance(source, list):
        if(len(sink) != len(flow_value)):
            print('Error: The number sink must match with the number of flows values.')
            return False

        # Adding the flow conservation
        Ptmp.add_list_of_constraints([sum([f[p, i] for p in G.predecessors(i)], 'p', 'pred(i)') == sum(
            [f[i, j] for j in G.successors(i)], 'j', 'succ(i)') for i in G.nodes() if not i in sink and i != source], 'i', 'nodes-(s,t)')

        # this constraint was redundant
        # Source flow at S
        #Ptmp.add_constraint(sum([f[p,source] for p in G.predecessors(source)],'p','pred(s)') + sum([fv for fv in flow_value], 'fv', 'flows') == sum([f[source,j] for j in G.successors(source)],'j','succ('+str(source)+')'))

        comment = "** One Source, Multiple Sinks **\n"
        for k in range(0, len(sink)):

            # Sink flow at T
            Ptmp.add_constraint(sum([f[p,
                                       sink[k]] for p in G.predecessors(sink[k])],
                                    'p',
                                    'pred(' + str(sink[k]) + ')') == sum([f[sink[k],
                                                                            j] for j in G.successors(sink[k])],
                                                                         'j',
                                                                         'succ(t)') + flow_value[k])

            if hasattr(flow_value[k], 'string'):
                fv = flow_value[k].string
            else:
                fv = str(flow_value[k])

            if graphName == '':
                comment = comment + "  Flow conservation from " + \
                    str(source) + " to " + str(sink[k]) + " with value " + fv + "\n"
            else:
                comment = comment + "  Flow conservation in " + \
                    str(graphName) + " from " + str(source) + " to " + str(sink[k]) + " with value " + fv + "\n"

    #
    # Multiple source, One Sink
    #
    elif not isinstance(sink, list):
        if(len(source) != len(flow_value)):
            print('Error: The number sink must match with the number of flows values.')
            return False

        # Adding the flow conservation
        Ptmp.add_list_of_constraints([sum([f[p, i] for p in G.predecessors(i)], 'p', 'pred(i)') == sum(
            [f[i, j] for j in G.successors(i)], 'j', 'succ(i)') for i in G.nodes() if not i in source and i != sink], 'i', 'nodes-(s,t)')

        # this constraint was redundant
        # Sink flow at S
        #Ptmp.add_constraint(sum([f[p,sink] for p in G.predecessors(sink)],'p','pred(s)') + sum([fv for fv in flow_value], 'fv', 'flows') == sum([f[sink,j] for j in G.successors(sink)],'j','succ('+str(sink)+')'))
        #Ptmp.add_constraint(sum([f[p,sink] for p in G.predecessors(sink)],'p','pred('+str(sink)+')') == sum([f[sink,j] for j in G.successors(sink)],'j','succ(t)') + sum([fv for fv in flow_value], 'fv', 'flows'))

        comment = "** Multiple Sources, One Sink **\n"
        for k in range(0, len(source)):

            # Source flow at T
            #Ptmp.add_constraint(sum([f[p,source[k]] for p in G.predecessors(source[k])],'p','pred('+str(source[k])+')') == sum([f[source[k],j] for j in G.successors(source[k])],'j','succ(t)') + flow_value[k])
            Ptmp.add_constraint(sum([f[p,
                                       source[k]] for p in G.predecessors(source[k])],
                                    'p',
                                    'pred(s)') + flow_value[k] == sum([f[source[k],
                                                                         j] for j in G.successors(source[k])],
                                                                      'j',
                                                                      'succ(' + str(source[k]) + ')'))

            if hasattr(flow_value[k], 'string'):
                fv = flow_value[k].string
            else:
                fv = str(flow_value[k])

            if graphName == '':
                comment = comment + "  Flow conservation from " + \
                    str(source[k]) + " to " + str(sink) + " with value " + fv + "\n"
            else:
                comment = comment + "  Flow conservation in " + \
                    str(graphName) + " from " + str(source[k]) + " to " + str(sink) + " with value " + fv + "\n"

    #
    # Multiple source, Multiple Sink
    #
    elif isinstance(sink, list) and isinstance(source, list):
        if(len(source) != len(flow_value)):
            print('Error: The number of sinks must match with the number of flow values.')
            return False
        if(len(sink) != len(source)):
            print('Error: The number of sinks must macht with the number of sources.')
            return False
        if(len(sink) != len(flow_value)):
            print('Error: The number of sinks must match with the numver of flow values.')
            return False

        comment = "** Multiple Sources, Multiple Sinks **\n"

        SS = list(set(source))
        TT = list(set(sink))

        if len(SS) <= len(TT):

            ftmp = {}
            for s in SS:
                ftmp[s] = {}
                sinks_from_s = [
                    t for (
                        i, t) in enumerate(sink) if source[i] == s]
                values_from_s = [v for (i, v) in enumerate(
                    flow_value) if source[i] == s]
                for e in G.edges():
                    ftmp[s][e] = Ptmp.add_variable(
                        'f[{0}][{1}]'.format(s, e), 1)
                Ptmp.add_constraint(
                    flow_Constraint(
                        G,
                        ftmp[s],
                        source=s,
                        sink=sinks_from_s,
                        flow_value=values_from_s,
                        graphName='G'))

            Ptmp.add_list_of_constraints([f[e] == sum(
                [ftmp[s][e] for s in SS], 's', 'sources') for e in G.edges()], 'e', 'E')

        else:
            ftmp = {}
            for t in TT:
                ftmp[t] = {}
                sources_to_t = [
                    s for (
                        i, s) in enumerate(source) if sink[i] == t]
                values_to_t = [v for (i, v) in enumerate(
                    flow_value) if sink[i] == t]
                for e in G.edges():
                    ftmp[t][e] = Ptmp.add_variable(
                        'f[{0}][{1}]'.format(t, e), 1)
                Ptmp.add_constraint(
                    flow_Constraint(
                        G,
                        ftmp[t],
                        source=sources_to_t,
                        sink=t,
                        flow_value=values_to_t,
                        graphName='G'))

            Ptmp.add_list_of_constraints([f[e] == sum(
                [ftmp[t][e] for t in TT], 't', 'sinks') for e in G.edges()], 'e', 'E')

        # comments
        for k in range(0, len(source)):
            if hasattr(flow_value[k], 'string'):
                fv = flow_value[k].string
            else:
                fv = str(flow_value[k])
            if graphName == '':
                comment = comment + "  Flow conservation from " + \
                    str(source[k]) + " to " + str(sink[k]) + " with value " + fv + "\n"
            else:
                comment = comment + "  Flow conservation in " + \
                    str(graphName) + " from " + str(source[k]) + " to " + str(sink[k]) + " with value " + fv + "\n"

    #
    # Handle errors
    #
    else:
        print('Error: unexpected error.')
        return False

    from .constraint import Flow_Constraint
    return Flow_Constraint(G, Ptmp, comment)


def drawGraph(G, capacity='capacity'):
    """"Draw a given Graph"""
    pos = nx.spring_layout(G)
    edge_labels = dict([((u, v,), d[capacity])
                        for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, pos)
    plt.show()


# A Python Library to create a Progress Bar.
# Copyright (C) 2008  BJ Dierkes <wdierkes@5dollarwhitebox.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# This class is an improvement from the original found at:
#
#   http://code.activestate.com/recipes/168639/
#

class ProgressBar:

    def __init__(self, min_value=0, max_value=100, width=None, **kwargs):
        self.char = kwargs.get('char', '#')
        self.mode = kwargs.get('mode', 'dynamic')  # fixed or dynamic
        if not self.mode in ['fixed', 'dynamic']:
            self.mode = 'fixed'

        self.bar = ''
        self.min = min_value
        self.max = max_value
        self.span = max_value - min_value
        if width is None:
            width = self.getTerminalSize()[1] - 10
        self.width = width
        self.amount = 0       # When amount == max, we are 100% done
        self.update_amount(0)

    def increment_amount(self, add_amount=1):
        """
        Increment self.amount by 'add_ammount' or default to incrementing
        by 1, and then rebuild the bar string.
        """
        new_amount = self.amount + add_amount
        if new_amount < self.min:
            new_amount = self.min
        if new_amount > self.max:
            new_amount = self.max
        self.amount = new_amount
        self.build_bar()

    def update_amount(self, new_amount=None):
        """
        Update self.amount with 'new_amount', and then rebuild the bar
        string.
        """
        if not new_amount:
            new_amount = self.amount
        if new_amount < self.min:
            new_amount = self.min
        if new_amount > self.max:
            new_amount = self.max
        self.amount = new_amount
        self.build_bar()

    def get_amount(self):
        return self.amount

    def build_bar(self):
        """
        Figure new percent complete, and rebuild the bar string base on
        self.amount.
        """
        diff = float(self.amount - self.min)
        percent_done = int(round((diff / float(self.span)) * 100.0))

        # figure the proper number of 'character' make up the bar
        all_full = self.width - 2
        num_hashes = int(round((percent_done * all_full) / 100))

        if self.mode == 'dynamic':
            # build a progress bar with self.char (to create a dynamic bar
            # where the percent string moves along with the bar progress.
            self.bar = self.char * num_hashes
        else:
            # build a progress bar with self.char and spaces (to create a
            # fixe bar (the percent string doesn't move)
            self.bar = self.char * num_hashes + ' ' * (all_full - num_hashes)

        percent_str = str(percent_done) + "%"
        self.bar = '[ ' + self.bar + ' ] ' + percent_str

    def __str__(self):
        return str(self.bar)

    def getTerminalSize(self):
        """
        returns (lines:int, cols:int)
        """
        import os
        import struct

        def ioctl_GWINSZ(fd):
            import fcntl
            import termios
            return struct.unpack(
                "hh", fcntl.ioctl(
                    fd, termios.TIOCGWINSZ, "1234"))
        # try stdin, stdout, stderr
        for fd in (0, 1, 2):
            try:
                return ioctl_GWINSZ(fd)
            except:
                pass
        # try os.ctermid()
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            try:
                return ioctl_GWINSZ(fd)
            finally:
                os.close(fd)
        except:
            pass
        # try `stty size`
        try:
            return tuple(
                int(x) for x in os.popen(
                    "stty size",
                    "r").read().split())
        except:
            pass
        # try environment variables
        try:
            return tuple(int(os.getenv(var)) for var in ("LINES", "COLUMNS"))
        except:
            pass
        # i give up. return default.
        return (25, 80)


# a not writable dict
class _NonWritableDict(dict):

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        print('not writable')
        raise Exception('NONO')

    def __delitem__(self, key):
        print('not writable')

    def _set(self, key, value):
        dict.__setitem__(self, key, value)

    def _del(self, key):
        dict.__delitem__(self, key)

    def _reset(self):
        for key in self.keys():
            self._del(key)


class QuadAsSocpError(Exception):
    """
    Exception raised when the problem can not be solved
    in the current form, because quad constraints are not handled.
    User should try to convert the quads as socp.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self): return self.msg

    def __repr__(self): return "QuadAsSocpError('" + self.msg + "')"


class NotAppropriateSolverError(Exception):
    """
    Exception raised when trying to solve a problem with
    a solver which cannot handle it
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self): return self.msg

    def __repr__(self): return "NotAppropriateSolverError('" + self.msg + "')"


class NonConvexError(Exception):
    """
    Exception raised when non-convex quadratic constraints
    are passed to a solver which cannot handle them.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self): return self.msg

    def __repr__(self): return "NonConvexError('" + self.msg + "')"


class DualizationError(Exception):
    """
    Exception raised when a non-standard conic problem is being dualized.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self): return self.msg

    def __repr__(self): return "DualizationError('" + self.msg + "')"
