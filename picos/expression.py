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
import sys
import six
from six.moves import zip, range
import itertools

from .tools import *
from .constraint import *

__all__ = ['Expression',
           'AffinExp',
           'Norm',
           'QuadExp',
           'GeneralFun',
           'LogSumExp',
           '_ConvexExp',
           'GeoMeanExp',
           'NormP_Exp',
           'TracePow_Exp',
           'DetRootN_Exp',
           'Sum_k_Largest_Exp',
           'Variable',
           'Set',
           'Ball',
           'Truncated_Simplex'
           ]

global INFINITY
INFINITY = 1e16

#----------------------------------
#                Expression
#----------------------------------


class Expression(object):
    """The parent class of :class:`AffinExp<picos.AffinExp>`
    (which is the parent class of :class:`Variable<picos.Variable>`),
    :class:`Norm<picos.Norm>`,
    :class:`LogSumExp<picos.LogSumExp>`, and
    :class:`QuadExp<picos.QuadExp>`.
    """
    # and :class:`GeneralFun<picos.GeneralFun>`.

    def __init__(self, string):
        self.string = string
        """String representation of the expression"""

    def eval(self):
        pass

    def set_value(self, value):
        raise ValueError('set_value can only be called on a Variable')

    def del_simple_var_value(self):
        raise ValueError(
            'del_simple_var_value can only be called on a Variable')

    def has_complex_coef(self):
        hcc = False
        if hasattr(self,'quad'):
            hcc = 'z' in [m.typecode for m in self.quad.values()]
        if hasattr(self,'aff'):
            hcc = hcc or ('z' in [m.typecode for m in self.aff.factors.values()])
            if self.aff.constant:
                hcc = hcc or ('z' == self.aff.constant.typecode)
        if hasattr(self,'factors'):
            hcc = hcc or ('z' in [m.typecode for m in self.factors.values()])
            if self.constant:
                hcc = hcc or ('z' == self.constant.typecode)
        return hcc


    value = property(
        eval,
        set_value,
        del_simple_var_value,
        "value of the affine expression")
    """value of the expression. The value of an expression is
           defined in the following three situations:

                     * The expression is **constant**.
                     * The user has assigned a value to each variable
                       involved in the expression.
                       (This can be done by setting the ``value`` property
                       of the instances of the class
                       :class:`Variable<picos.Variable>`
                       or by using the function
                       :func:`set_var_value()<picos.Problem.set_var_value>`
                       of the class
                       :class:`Problem<picos.Problem>`.)
                     * The expression involves variables of a problem
                       that has already been solved, so that the variables
                       are set at their optimal values.
    """

    @property
    def size(self):
        """size of the affine expression"""
        return self._size

    """
        **Example**

        >>> import picos as pic
        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',2)
        >>> x.is_valued()
        False
        >>> print abs(x)
        # norm of a (2 x 1)- expression: ||x|| #
        >>> x.value=[3,4]
        >>> abs(x).is_valued()
        True
        >>> print abs(x)        #now that x is valued, it is the value of the expression which is printed
        5.0
        """

    def is_valued(self):
        """Does the expression have a value ?
        Returns true if all variables involved
        in the expression are not ``None``."""
        try:
            val = self.value
            return not(val is None)
        except Exception:
            return False

    def __le__(self, exp):
        return self.__lt__(exp)

    def __ge__(self, exp):
        return self.__gt__(exp)


#----------------------------------
#                 AffinExp class
#----------------------------------

class AffinExp(Expression):
    u"""A class for defining vectorial (or matrix) affine expressions.
    It derives from :class:`Expression<picos.Expression>`.

    **Overloaded operators**

            :``+``: sum            (with an affine or quadratic expression)
            :``+=``: in-place sum  (with an affine or quadratic expression)
            :``-``: substraction   (with an affine or quadratic expression) or unitary minus
            :``*``: multiplication (by another affine expression or a scalar)
            :``^``: hadamard product (elementwise multiplication with another affine expression,
                        similarly as MATLAB operator ``.*`` )
            :``/``: division       (by a scalar)
            :``|``: scalar product (with another affine expression)
            :``[.]``: slice of an affine expression
            :``abs()``: Euclidean norm (Frobenius norm for matrices)
            :``**``: exponentiation (works with arbitrary powers for constant
                       affine expressions, and any nonzero exponent otherwise).
                       In the case of a nonconstant affine expression, the exponentiation
                       returns a quadratic expression if the exponent is 2, or
                       a :class:`TracePow_Exp<picos.TracePow_Exp>` object for other
                       exponents. A rational approximation of the exponent is used,
                       and the power inequality is internally replaced by an equivalent set of second order
                       cone inequalities.
            :``&``: horizontal concatenation (with another affine expression)
            :``//``: vertical concatenation (with another affine expression)
            :``<``: less **or equal** (than an affine or quadratic expression)
            :``>``: greater **or equal** (than an affine or quadratic expression)
            :``==``: is equal (to another affine expression)
            :``<<``: less than inequality in the Loewner ordering (linear matrix inequality
                       :math:`\preceq`  ); or, if the right hand side is a :class:`Set`,
                       membership in this set.
            :``>>``: greater than inequality in the Loewner ordering (linear matrix inequality :math:`\succeq` )

    .. Warning::

            We recall here the implicit assumptions that are made
            when using relation operator overloads, in the following
            two situations:

                    * the rotated second order cone constraint
                      ``abs(exp1)**2 < exp2 * exp3`` implicitely
                      assumes that the scalar expression ``exp2`` (and hence ``exp3``)
                      **is nonnegative**.

                    * the linear matrix inequality ``exp1 >> exp2`` only tells picos
                      that the symmetric matrix whose lower triangular elements
                      are those of ``exp1-exp2`` is positive semidefinite. The
                      matrix ``exp1-exp2`` **is not constrained to be symmetric**.
                      Hence, you should manually add the constraint
                      ``exp1-exp2 == (exp1-exp2).T`` if it is not clear from the data
                      that this matrix is symmetric.

    """

    def __init__(self, factors=None, constant=None,
                 size=(1, 1),
                 string='0'
                 ):
        if factors is None:
            factors = {}
        Expression.__init__(self, string)
        self.factors = factors
        """
                dictionary storing the matrix of coefficients of the linear
                part of the affine expressions. The matrices of coefficients
                are always stored with respect to vectorized variables (for the
                case of matrix variables), and are indexed by instances
                of the class :class:`Variable<picos.Variable>`.
                """
        self.constant = constant
        """constant of the affine expression,
                stored as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
                """
        
        assert len(size)==2 and _is_integer(size[0]) and _is_integer(size[1])
        #make sure they are of class `int`, otherwise compatibility problem with py3...
        self._size = (int(size[0]), int(size[1]))
        
        """size of the affine expression"""
        # self.string=string

    def __hash__(self):
        return Expression.__hash__(self)

    @property
    def size(self):
        """size of the affine expression"""
        return self._size

    def __str__(self):
        if self.is_valued():
            if self.size == (1, 1):
                return str(self.value[0])
            else:
                return str(self.value)
        else:
            return repr(self)

    def __repr__(self):
        affstr = '# ({0} x {1})-affine expression: '.format(self.size[0],
                                                            self.size[1])
        affstr += self.affstring()
        affstr += ' #'
        return affstr

    def hard_copy(self):
        import copy
        facopy = {}
        for f, m in six.iteritems(
                self.factors):  # copy matrices but not the variables (keys of the dict)
            facopy[f] = copy.deepcopy(m)

        conscopy = copy.deepcopy(self.constant)
        return AffinExp(facopy, conscopy, self.size, self.string)

    def copy(self):
        excopy = 1*self
        excopy.string = self.string
        return excopy

    def soft_copy(self):
        return AffinExp(self.factors, self.constant, self.size, self.string)

    def affstring(self):
        return self.string

    def eval(self, ind=None):
        if self.constant is None:
            val = spmatrix([], [], [], (self.size[0] * self.size[1], 1))
        else:
            val = self.constant
        if self.is0():
            return cvx.matrix(val, self.size)

        for k in self.factors:
            # ignore this factor if the coef is 0
            if not(self.factors[k]):
                continue
            if ind is None:
                if not k.value is None:
                    if k.vtype == 'symmetric':
                        val = val + self.factors[k] * svec(k.value)
                    else:
                        val = val + self.factors[k] * k.value[:]
                else:
                    raise Exception(k + ' is not valued')
            else:
                if ind in k.value_alt:
                    if k.vtype == 'symmetric':
                        val = val + self.factors[k] * svec(k.value_alt[ind])
                    else:
                        val = val + self.factors[k] * k.value_alt[ind][:]
                else:
                    raise Exception(
                        k + ' does not have a value for the index ' + str(ind))
        return cvx.matrix(val, self.size)

    def set_value(self, value):
        # is it a complex variable?
        if self.is_pure_complex_var():
            facs = list(self.factors.keys())
            if facs[0].name.endswith('_RE'):
                Zr = facs[0]
                Zi = facs[1]
            else:
                Zr = facs[1]
                Zi = facs[1]
            Zr.value = value.real()
            if value.typecode == 'z':
                Zi.value = value.imag()
            else:
                Zi.value = 0
            return
        # is it an antisym variable ?
        if self.is_pure_antisym_var():
            facs = list(self.factors.keys())
            vutri = facs[0]
            n = int((vutri.size[0])**0.5)
            value = _retrieve_matrix(value, (n, n))[0]
            vutri.value = _utri(value)
            return
        raise ValueError('set_value can only be called on a Variable')

    def del_simple_var_value(self):
        # is it a complex variable?
        if self.is_pure_complex_var():
            facs = list(self.factors.keys())
            if facs[0].name.endswith('_RE'):
                Zr = facs[0]
                Zi = facs[1]
            else:
                Zr = facs[1]
                Zi = facs[1]
            del Zi.value
            del Zr.value
            return
        # is it an antisym variable ?
        if self.is_pure_antisym_var():
            facs = list(self.factors.keys())
            vutri = facs[0]
            del vutri.value
            return
        raise ValueError(
            'del_simple_var_value can only be called on a Variable')

    value = property(
        eval,
        set_value,
        del_simple_var_value,
        "value of the affine expression")

    def get_type(self):
        # is it a complex variable?
        if self.is_pure_complex_var():
            return 'complex'
        elif self.is_pure_antisym_var():
            return 'antisym'
        raise ValueError('get_type can only be called on a Variable')

    def set_type(self, value):
        raise ValueError('set_type can only be called on a Variable')

    def del_type(self):
        raise ValueError('vtype cannot be deleted')

    vtype = property(
        get_type,
        set_type,
        del_type,
        "vtype (for complex and antisym variables)")

    def get_real(self):
        # is it a complex variable?
        if self.is_pure_complex_var():
            facs = list(self.factors.keys())
            if facs[0].name.endswith('_RE'):
                return facs[0]
            else:
                return facs[1]
        else:
            return 0.5 * (self + self.conj)

    def set_real(self, value):
        raise ValueError('real is not writable')

    def del_real(self):
        raise ValueError('real is not writable')

    real = property(
        get_real,
        set_real,
        del_real,
        "real part (for complex expressions)")

    def get_imag(self):
        # is it a complex variable?
        if self.is_pure_complex_var():
            facs = list(self.factors.keys())
            if facs[0].name.endswith('_RE'):
                return facs[1]
            else:
                return facs[0]
        else:
            return -1j * 0.5 * (self - self.conj)

    def set_imag(self, value):
        raise ValueError('imag is not writable')

    def del_imag(self):
        raise ValueError('imag is not writable')

    imag = property(
        get_imag,
        set_imag,
        del_imag,
        "imaginary part (for complex expressions)")

    def is_pure_complex_var(self):
        if self.constant:
            return False
        if (len(self.factors) == 2):
            facs = list(self.factors.keys())
            if facs[0].name.endswith('_RE') and _is_idty(
                    self.factors[facs[0]]):
                if facs[1].name.endswith(
                        '_IM') and _is_idty(-1j * self.factors[facs[1]]):
                    return True
            if facs[1].name.endswith('_RE') and _is_idty(
                    self.factors[facs[1]]):
                if facs[0].name.endswith(
                        '_IM') and _is_idty(-1j * self.factors[facs[0]]):
                    return True
        return False

    def is_pure_antisym_var(self):
        if self.constant:
            return False
        if (len(self.factors) == 1):
            facs = list(self.factors.keys())
            if facs[0].name.endswith('_utri') and _is_idty(
                    self.factors[facs[0]], 'antisym'):
                return True
        return False

    def is_real(self):
        real = True
        for (x, A) in six.iteritems(self.factors):
            if x.vtype == 'complex':
                return False
            if A.typecode == 'z' and bool(A.imag()):
                if x.vtype != 'hermitian':
                    return False
            if x.vtype == 'hermitian':
                # test if this is an expression of the form P|Z with P
                # Hermitian
                n = int(A.size[1]**(0.5))
                for i in range(A.size[0]):
                    vv = A[i, :]
                    P = cvx.matrix(vv, (n, n))
                    vr = (P.real() - P.real().T)[:]
                    if P.typecode == 'z':
                        vi = (P.imag() + P.imag().T)[:]
                    else:
                        vi = cvx.matrix(0., (1, 1))
                    if (vi.T * vi)[0] + (vr.T * vr)[0] > 1e-6:
                        return False
        if self.constant is None:
            return True
        if self.constant.typecode == 'z' and bool(self.constant.imag()):
            return False
        return True

    def is_valued(self, ind=None):

        try:
            for k in self.factors:
                if ind is None:
                    if k.value is None:
                        return False
                else:
                    if ind not in k.value_alt:
                        return False
        except:
            return False

        # Yes, you can call eval(ind) without any problem.
        return True

    def is0(self):
        """is the expression equal to 0 ?"""
        if bool(self.constant):
            return False
        for f in self.factors:
            if bool(self.factors[f]):
                return False
        return True

    def is1(self):
        if not bool(self.constant):
            return False
        if not(self.size == (1, 1) and self.constant[0] == 1):
            return False
        for f in self.factors:
            if bool(self.factors[f]):
                return False
        return True

    def isconstant(self):
        """is the expression constant (no variable involved) ?"""
        for f in self.factors:
            if bool(self.factors[f]):
                return False
        return True

    def __index__(self):
        if self.is_valued() and self.size == (
                1, 1) and int(
                self.value[0]) == self.value[0]:
            return int(self.value[0])
        else:
            raise Exception(
                'unexpected index (nonvalued, multidimensional, or noninteger)')

    def inplace_transpose(self):
        if isinstance(self, Variable):
            raise Exception(
                'inplace_transpose should not be called on a Variable object')
        for k in self.factors:
            bsize = self.size[0]
            bsize2 = self.size[1]
            I0 = [(i // bsize) + (i % bsize) *
                  bsize2 for i in self.factors[k].I]
            J = self.factors[k].J
            V = self.factors[k].V
            self.factors[k] = spmatrix(V, I0, J, self.factors[k].size)

        if not (self.constant is None):
            self.constant = cvx.matrix(self.constant,
                                       self.size).T[:]
        self._size = (self.size[1], self.size[0])
        if (('*' in self.affstring()) or ('/' in self.affstring())
                or ('+' in self.affstring()) or ('-' in self.affstring())):
            self.string = '( ' + self.string + ' ).T'
        else:
            self.string += '.T'

    def transpose(self):
        selfcopy = self.copy()
        selfcopy.inplace_transpose()
        return selfcopy

    def setT(self, value):
        raise AttributeError("attribute 'T' of 'AffinExp' is not writable")

    def delT(self):
        raise AttributeError("attribute 'T' of 'AffinExp' is not writable")

    T = property(transpose, setT, delT, "transposition")

    def conjugate(self):
        selfcopy = self.copy()
        selfcopy.inplace_conjugate()
        return selfcopy

    def inplace_conjugate(self):
        if isinstance(self, Variable):
            raise Exception(
                'inplace_conjugate should not be called on a Variable object')
        for k in self.factors:
            if k.vtype == 'hermitian':
                fack = self.factors[k]
                I = fack.I
                J = fack.J
                V = fack.V
                J0 = []
                V0 = []
                n = int(fack.size[1]**0.5)
                for j, v in zip(J, V):
                    col, row = divmod(j, n)
                    J0.append(row * n + col)
                    V0.append(v.conjugate())
                self.factors[k] = spmatrix(V0, I, J0, fack.size)

            elif self.factors[k].typecode == 'z':
                Fr = self.factors[k].real()
                Fi = self.factors[k].imag()
                self.factors[k] = Fr - 1j * Fi
        if not self.constant is None:
            if self.constant.typecode == 'z':
                Fr = self.constant.real()
                Fi = self.constant.imag()
                self.constant = Fr - 1j * Fi

        if (('*' in self.affstring()) or ('/' in self.affstring())
                or ('+' in self.affstring()) or ('-' in self.affstring())):
            self.string = '( ' + self.string + ' ).conj'
        else:
            self.string += '.conj'

    def setconj(self, value):
        raise AttributeError("attribute 'conj' of 'AffinExp' is not writable")

    def delconj(self):
        raise AttributeError("attribute 'conj' of 'AffinExp' is not writable")

    conj = property(conjugate, setconj, delconj, "complex conjugate")

    def inplace_Htranspose(self):
        if isinstance(self, Variable):
            raise Exception(
                'inplace_transpose should not be called on a Variable object')
        for k in self.factors:
            if k.vtype == 'hermitian':
                bsize = self.size[0]
                bsize2 = self.size[1]
                vsize = k.size[0]
                vsize2 = k.size[1]
                I0 = [(i // bsize) + (i % bsize) *
                      bsize2 for i in self.factors[k].I]
                J0 = [(j // vsize) + (j % vsize) *
                      vsize2 for j in self.factors[k].J]
                V0 = [v.conjugate() for v in self.factors[k].V]
                self.factors[k] = spmatrix(
                    V0, I0, J0, self.factors[k].size)
            else:
                F = self.factors[k]
                bsize = self.size[0]
                bsize2 = self.size[1]
                I0 = [(i // bsize) + (i % bsize) * bsize2 for i in F.I]
                J = F.J
                V = [v.conjugate() for v in F.V]
                self.factors[k] = spmatrix(V, I0, J, F.size)

        if not (self.constant is None):
            if self.constant.typecode == 'z':
                self.constant = cvx.matrix(self.constant,
                                           self.size).H[:]
            else:
                self.constant = cvx.matrix(self.constant,
                                           self.size).T[:]
        self._size = (self.size[1], self.size[0])
        if (('*' in self.affstring()) or ('/' in self.affstring())
                or ('+' in self.affstring()) or ('-' in self.affstring())):
            self.string = '( ' + self.string + ' ).H'
        else:
            self.string += '.H'

    def Htranspose(self):
        selfcopy = self.copy()
        selfcopy.inplace_Htranspose()
        return selfcopy

    def setH(self, value):
        raise AttributeError("attribute 'H' of 'AffinExp' is not writable")

    def delH(self):
        raise AttributeError("attribute 'H' of 'AffinExp' is not writable")

    H = property(Htranspose, setH, delH, "Hermitian transposition")
    """Hermitian (or conjugate) transposition"""

    def inplace_partial_transpose(self, dim_1 = None,subsystems=None, dim_2=None):
        if isinstance(self, Variable):
            raise Exception(
                'partial_transpose should not be called on a Variable object')
        size = self.size

        if dim_1 is None:
            try :
                s0 = size[0]
                k = [s0**(1./i)==int(s0**(1./i)) for i in range(2,7)].index(True)+2
                subsize = int(s0**(1./k))
                dim_1 = (subsize,)*k
            except ValueError:
                raise ValueError(
                    'partial_transpose can only be applied to n**k x n**k matrices when the dimensions of subsystems are not defined')

        if dim_2 is None:
            dim_2 = dim_1

        assert isinstance(dim_1,tuple)
        assert isinstance(dim_2,tuple)
        assert (len(dim_1)==len(dim_2))

        assert (np.product(dim_1) == size[0]),'the size of subsystems do not match the size of the entire matrix'
        assert (np.product(dim_2) == size[1]),'the size of subsystems do not match the size of the entire matrix'

        N = len(dim_1)

        if subsystems is None:
            subsystems = (N-1,)
        if isinstance(subsystems,int):
            subsystems = (subsystems,)

        assert all([i in range(N) for i in subsystems])

        newdim_1 = ()
        newdim_2 = ()
        for i in range(N):
            if i in subsystems:
                newdim_1 += (dim_2[i],)
                newdim_2 += (dim_1[i],)
            else:
                newdim_2 += (dim_2[i],)
                newdim_1 += (dim_1[i],)

        newsize = (int(np.product(newdim_1)), int(np.product(newdim_2)))

        def block_indices(dims,ii):
            inds = []
            rem = ii
            for k in range(len(dims)):
                blk,rem = divmod(rem,np.product(dims[k+1:]))
                inds.append(int(blk))

            return inds

        for k in self.factors:
            I0 = []
            J = self.factors[k].J
            V = self.factors[k].V
            for i in self.factors[k].I:
                column, row = divmod(i, size[0])
                row_blocks = block_indices(dim_1,row)
                col_blocks = block_indices(dim_2,column)
                new_rblock = [col_blocks[l] if l in subsystems else row_blocks[l] for l in range(N)]
                new_cblock = [row_blocks[l] if l in subsystems else col_blocks[l] for l in range(N)]
                newrow = int(sum([new_rblock[l] * np.product(newdim_1[l+1:]) for l in range(N)]))
                newcol = int(sum([new_cblock[l] * np.product(newdim_2[l+1:]) for l in range(N)]))
                I0.append(newcol * newsize[0] + newrow)
            self.factors[k] = spmatrix(V, I0, J, self.factors[k].size)
        if not (self.constant is None):
            spconstant = cvx.sparse(self.constant)
            J = spconstant.J
            V = spconstant.V
            I0 = []
            for i in spconstant.I:
                column, row = divmod(i, size[0])
                row_blocks = block_indices(dim_1,row)
                col_blocks = block_indices(dim_2,column)
                new_rblock = [col_blocks[l] if l in subsystems else row_blocks[l] for l in range(N)]
                new_cblock = [row_blocks[l] if l in subsystems else col_blocks[l] for l in range(N)]
                newrow = int(sum([new_rblock[l] * np.product(dim_1[l+1:]) for l in range(N)]))
                newcol = int(sum([new_cblock[l] * np.product(dim_2[l+1:]) for l in range(N)]))
                I0.append(newcol * newsize[0] + newrow)
            self.constant = spmatrix(V, I0, J, spconstant.size)
        self._size = newsize
        if (('*' in self.affstring()) or ('/' in self.affstring())
                or ('+' in self.affstring()) or ('-' in self.affstring())):
            self.string = '( ' + self.string + ' ).Tx'
        else:
            self.string += '.Tx'

    def partial_transpose(self, dim_1=None, subsystems=None, dim_2=None):
        selfcopy = self.copy()
        selfcopy.inplace_partial_transpose(dim_1,subsystems,dim_2)
        return selfcopy

    def setTx(self, value):
        raise AttributeError("attribute 'Tx' of 'AffinExp' is not writable")

    def delTx(self):
        raise AttributeError("attribute 'Tx' of 'AffinExp' is not writable")

    Tx = property(partial_transpose, setTx, delTx, "Partial transposition")
    """Partial transposition (for an n**2 x n**2 matrix, assumes subblocks of size n x n).
           cf. doc of :func:`picos.partial_transpose() <picos.tools.partial_transpose>`"""

    def partial_trace(self, k=1, dim=None):
        """partial trace
        cf. doc of :func:`picos.partial_trace() <picos.tools.partial_trace>`
        """
        sz = self.size
        if dim is None:
            if sz[0] == sz[1] and (sz[0]**0.5) == int(sz[0]**0.5) and (sz[1]**0.5) == int(sz[1]**0.5):
                dim = (int(sz[0]**0.5), int(sz[1]**0.5))
            else:
                raise ValueError('The default parameter dim=None assumes X is a n**2 x n**2 matrix')

        # checks if dim is a list (or tuple) of lists (or tuples) of two integers each
        T = [list,tuple]
        if type(dim) in T and all([type(d) in T and len(d) == 2 for d in dim]) and all([type(n) is int for d in dim for n in d]):
            dim = [d for d in zip(*dim)]
            pdim = np.product(dim[0]),np.product(dim[1])

        # if dim is a single list of integers we assume that no subsystem is rectangular
        elif type(dim) in [list,tuple] and all([type(n) is int for n in dim]):
            pdim = np.product(dim),np.product(dim)
            dim = (dim,dim)
        else:
            raise ValueError('Wrong dim variable')

        if len(dim[0]) != len(dim[1]):
            raise ValueError('Inconsistent number of subsystems, fix dim variable')

        if pdim[0] != sz[0] or pdim[1] != sz[1]:
            raise ValueError('The product of the sub-dimensions does not match the size of X')

        if k > len(dim[0])-1:
            raise Exception('There is no k-th subsystem, fix k or dim variable')

        if dim[0][k] != dim[1][k] :
            raise ValueError('The dimensions of the subsystem to trace over don\'t match')

        dim_reduced = [list(d) for d in dim]
        del dim_reduced[0][k]
        del dim_reduced[1][k]
        dim_reduced = tuple(tuple(d) for d in dim_reduced)
        pdimred = tuple([np.product(d) for d in dim_reduced])
        fact = cvx.spmatrix([], [], [], (np.product(pdimred), np.product(pdim)))

        for iii in itertools.product(*[range(i) for i in dim_reduced[0]]):
            for jjj in itertools.product(*[range(j) for j in dim_reduced[1]]):
            # element iii,jjj of the partial trace
            
                row = int(sum([iii[j] * np.product(dim_reduced[0][j + 1:]) for j in range(len(dim_reduced[0]))]))
                col = int(sum([jjj[j] * np.product(dim_reduced[1][j + 1:]) for j in range(len(dim_reduced[1]))]))
                # this corresponds to the element row,col in the matrix basis
                rowij = col * pdimred[0] + row
                # this corresponds to the elem rowij in vectorized form
                
                # computes the partial trace for iii,jjj
                for l in range(dim[0][k]):
                    iili = list(iii)
                    iili.insert(k, l)
                    iili = tuple(iili)

                    jjlj = list(jjj)
                    jjlj.insert(k, l)
                    jjlj = tuple(jjlj)

                    row_l = int(sum([iili[j] * np.product(dim[0][j + 1:]) for j in range(len(dim[0]))]))
                    col_l = int(sum([jjlj[j] * np.product(dim[1][j + 1:]) for j in range(len(dim[1]))]))

                    colij_l = col_l * pdim[0] + row_l
                    fact[int(rowij), int(colij_l)] = 1

        newfacs = {}
        for x in self.factors:
            newfacs[x] = fact * self.factors[x]
        if self.constant:
            cons = fact * self.constant
        else:
            cons = None
        return AffinExp(newfacs, cons, (pdimred[0],pdimred[1]), 'Tr_' + str(k) + '(' + self.string + ')')

    def hadamard(self, fact):
        """hadamard (elementwise) product"""
        return self ^ fact

    def __xor__(self, fact):
        """hadamard (elementwise) product"""
        selfcopy = self.copy()
        if isinstance(fact, AffinExp):
            if fact.isconstant():
                fac, facString = cvx.sparse(fact.eval()), fact.string
            else:
                if self.isconstant():
                    return fact ^ self
                else:
                    raise Exception('not implemented')
        else:
            fac, facString = _retrieve_matrix(fact, self.size[0])
        if fac.size == (1, 1) and selfcopy.size[0] != 1:
            fac = fac[0] * cvx.spdiag([1.] * selfcopy.size[0])
        if self.size == (1, 1) and fac.size[1] != 1:
            oldstring = selfcopy.string
            selfcopy = selfcopy.diag(fac.size[1])
            selfcopy.string = oldstring
        if selfcopy.size[0] != fac.size[0] or selfcopy.size[1] != fac.size[1]:
            raise Exception('incompatible dimensions')
        mm, nn = selfcopy.size
        bfac = spmatrix([], [], [], (mm * nn, mm * nn))
        for i, j, v in zip(fac.I, fac.J, fac.V):
            bfac[j * mm + i, j * mm + i] = v
        for k in selfcopy.factors:
            newfac = bfac * selfcopy.factors[k]
            selfcopy.factors[k] = newfac
        if selfcopy.constant is None:
            newfac = None
        else:
            newfac = bfac * selfcopy.constant
        selfcopy.constant = newfac
        """
                #the following removes 'I' from the string when a matrix is multiplied
                #by the identity. We leave the 'I' when the factor of identity is a scalar
                if len(facString)>0:
                        if facString[-1]=='I' and (len(facString)==1
                                 or facString[-2].isdigit() or facString[-2]=='.') and (
                                 self.size != (1,1)):
                                facString=facString[:-1]
		"""
        sstring = selfcopy.affstring()
        if len(facString) > 0:
            if ('+' in sstring) or ('-' in sstring):
                sstring = '( ' + sstring + ' )'
            if ('+' in facString) or ('-' in facString):
                facString = '( ' + facString + ' )'

            selfcopy.string = facString + '∘' + sstring

        return selfcopy

    def __rxor__(self, fact):
        return self.__xor__(fact)

    def __rmul__(self, fact):

        if isinstance(fact, AffinExp):
            if fact.isconstant():
                fac, facString = cvx.sparse(fact.eval()), fact.string
                if fac.size == (1, 1) and self.size[0] != 1:
                    fac = _blocdiag(fac, self.size[0])
            else:
                raise Exception('not implemented')

        #fast handling for the most standard case (no need to go inside retrieve_matrix)
        if (isinstance(self,Variable) and
          self.vtype not in ('symmetric','antisym',) and
          hasattr(fact,'size') and isinstance(fact.size,tuple) and
          len(fact.size)==2 and fact.size[1]==self.size[0]):
            if not isinstance(fact,AffinExp):
                facString = '[ {0} x {1} MAT ]'.format(*fact.size)
                fac = fact
            bfac = _blocdiag(fac, self.size[1])
            return AffinExp(factors={self: bfac}, size=(fac.size[0], self.size[1]), string=facString+'*'+self.string)

        if not isinstance(fact, AffinExp):
            fac, facString = _retrieve_matrix(fact, self.size[0])
            # the following removes 'I' from the string when a matrix is multiplied
            # by the identity. We leave the 'I' when the factor of identity is a
            # scalar
            if len(facString) > 0:
                if (facString[-1] == 'I' and (len(facString) == 1 or facString[-2].isdigit()
                   or facString[-2] == '.') and (self.size != (1, 1))):
                    facString = facString[:-1]

        if (isinstance(self,Variable) and
          self.vtype not in ('symmetric','antisym',) and
          self.size[0] == fac.size[1]):
            bfac = _blocdiag(fac, self.size[1])
            return AffinExp(factors={self: bfac}, size=(fac.size[0], self.size[1]), string=facString+'*'+self.string)

        selfcopy = self.soft_copy()

        is_scalar_mult = (isinstance(fact, float) or isinstance(fact, six.integer_types) or isinstance(fact, np.float64) or
          isinstance(fact, np.int64) or isinstance(fact, np.complex128) or isinstance(fact, complex) or
          (hasattr(fact,'size') and fact.size==(1,1)) or (hasattr(fact,'shape') and fact.shape in ((1,),(1,1))) )

        if self.size == (1, 1) and fac.size[1] != 1:
            oldstring = selfcopy.string
            selfcopy = selfcopy.diag(fac.size[1])
            selfcopy.string = oldstring
        if selfcopy.size[0] != fac.size[1]:
            raise Exception('incompatible dimensions')
        if is_scalar_mult:
            bfac = fac[0]
        else:
            bfac = _blocdiag(fac, selfcopy.size[1])
        newfac = {}
        for k in selfcopy.factors:
            newfac[k] = bfac * selfcopy.factors[k]
        if selfcopy.constant is None:
            newcons = None
        else:
            newcons = bfac * selfcopy.constant

        selfcopy = AffinExp(factors=newfac,constant=newcons, size=(fac.size[0], selfcopy.size[1]), string=selfcopy.string)

        sstring = selfcopy.affstring()
        if len(facString) > 0:
            if ('+' in sstring) or ('-' in sstring):
                sstring = '( ' + sstring + ' )'
            if ('+' in facString) or ('-' in facString):
                facString = '( ' + facString + ' )'

            selfcopy.string = facString + '*' + sstring

        return selfcopy

    def __mul__(self, fact):
        """product of 2 affine expressions"""
        if isinstance(fact, AffinExp):
            if fact.isconstant():
                fac, facString = cvx.sparse(fact.eval()), fact.string
            elif self.isconstant():
                return fact.__rmul__(self)
            elif self.size[0] == 1 and fact.size[1] == 1 and self.size[1] == fact.size[0]:
                # quadratic expression
                linpart = AffinExp({}, constant=None, size=(1, 1))
                if not self.constant is None:
                    linpart = linpart + self.constant.T * fact
                if not fact.constant is None:
                    linpart = linpart + self * fact.constant
                if not ((fact.constant is None) or (self.constant is None)):
                    linpart = linpart - self.constant.T * fact.constant

                quadpart = {}
                for i in self.factors:
                    for j in fact.factors:
                        quadpart[i, j] = self.factors[i].T * fact.factors[j]
                stleft = self.affstring()
                stright = fact.affstring()
                if ('+' in stleft) or ('-' in stleft):
                    if len(stleft) > 3 and not(
                            stleft[0] == '(' and stleft[-3:] == ').T'):
                        stleft = '( ' + stleft + ' )'
                if ('+' in stright) or ('-' in stright):
                    stright = '( ' + stright + ' )'
                if self.size[1] == 1:
                    return QuadExp(quadpart, linpart, stleft +
                                   '*' + stright, LR=(self, fact))
                else:
                    return QuadExp(quadpart, linpart, stleft + '*' + stright)
            else:
                raise Exception('not implemented')
        elif isinstance(fact, QuadExp):
            return fact * self
        # product with a constant
        else:
            if self.size == (1, 1):  # scalar mult. of the constant
                fac, facString = _retrieve_matrix(fact, None)
            else:  # normal matrix multiplication, we expect a size
                fac, facString = _retrieve_matrix(fact, self.size[1])

        is_scalar_mult = (_is_numeric(fact) or
          (hasattr(fact,'size') and fact.size==(1,1)) or (hasattr(fact,'shape') and fact.shape in ((1,),(1,1))) )

        if is_scalar_mult:
            alpha = fac[0]
            newfacs = {}
            for k, M in six.iteritems(self.factors):
                newfacs[k] = alpha * M
            if self.constant is None:
                newcons = None
            else:
                newcons = alpha * self.constant
            sstring = self.affstring()
            if ('+' in sstring) or ('-' in sstring):
                sstring = '( ' + sstring + ' )'
            return AffinExp(
                newfacs,
                newcons,
                self.size,
                facString +
                '*' +
                sstring)

        selfcopy = self.soft_copy()

        if self.size == (1, 1) and fac.size[0] != 1:
            oldstring = selfcopy.string
            selfcopy = selfcopy.diag(fac.size[0])
            selfcopy.string = oldstring

        prod = (selfcopy.T.__rmul__(fac.T)).T
        prod._size = (selfcopy.size[0], fac.size[1])
        # the following removes 'I' from the string when a matrix is multiplied
        # by the identity. We leave the 'I' when the factor of identity is a
        # scalar
        if len(facString) > 0:
            if facString[-1] == 'I' and (len(facString) == 1 or facString[-2].isdigit(
            ) or facString[-2] == '.') and(self.size != (1, 1)):
                facString = facString[:-1]
        sstring = selfcopy.affstring()
        if len(facString) > 0:
            if ('+' in sstring) or ('-' in sstring):
                sstring = '( ' + sstring + ' )'
            if ('+' in facString) or ('-' in facString):
                facString = '( ' + facString + ' )'
            prod.string = sstring + '*' + facString
        else:
            prod.string = selfcopy.string
        return prod

    def __or__(self, fact):  # scalar product
        selfcopy = self.copy()
        if not(isinstance(fact, AffinExp)):
            fac, facString = _retrieve_matrix(fact, self.size)
            fact = AffinExp(
                {},
                constant=fac[:],
                size=fac.size,
                string=facString)

        # now we must have an AffinExp
        if self.size != fact.size:
            raise Exception('incompatible dimensions')

        dotp = fact[:].H * self[:]
        facString = fact.string
        if facString[-1] == 'I' and (len(facString) == 1
                                     or facString[-2].isdigit() or facString[-2] == '.'):
            dotp.string = facString[:-1] + 'trace( ' + self.string + ' )'
        else:
            dotp.string = '〈 ' + self.string + ' | ' + facString + ' 〉'

        return dotp

    def __ror__(self, fact):  # scalar product
        if not(isinstance(fact, AffinExp)):
            fac, facString = _retrieve_matrix(fact, self.size)
            fact = AffinExp(
                {},
                constant=fac[:],
                size=fac.size,
                string=facString)

        # now we must have an AffinExp
        if self.size != fact.size:
            raise Exception('incompatible dimensions')

        dotp = self[:].H * fact[:]
        facString = fact.string
        if facString[-1] == 'I' and (len(facString) == 1
                                     or facString[-2].isdigit() or facString[-2] == '.'):
            dotp.string = facString[:-1] + 'trace( ' + self.string + ' )'
        else:
            dotp.string = '〈 ' + facString + ' | ' + self.string + ' 〉'

        return dotp

    def __add__(self, term):
        selfcopy = self.copy()
        selfcopy += term
        return selfcopy

    def __radd__(self, term):
        return self.__add__(term)

    # inplace sum
    def __iadd__(self, term):
        if isinstance(term, AffinExp):
            if term.size == (1, 1) and self.size != (1, 1):
                oldstring = term.string
                term = cvx.matrix(1., self.size) * term.diag(self.size[1])
                term.string = '|' + oldstring + '|'
            if self.size == (1, 1) and term.size != (1, 1):
                oldstring = self.string
                selfone = cvx.matrix(1., term.size) * self.diag(term.size[1])
                selfone.string = '|' + oldstring + '|'
                selfone += term
                return selfone
            if term.size != self.size:
                raise Exception('incompatible dimension in the sum')
            for k in term.factors:
                if k in self.factors:
                    try:
                        self.factors[k] += term.factors[k]
                    except TypeError as ex:
                        if str(ex).startswith('incompatible') or str(
                                ex).startswith('invalid'):  # incompatible typecodes
                            self.factors[k] = self.factors[k] + term.factors[k]
                        else:
                            raise
                else:
                    self.factors[k] = term.factors[k]
            if self.constant is None and term.constant is None:
                pass
            else:
                if self.constant is None:
                    if not(
                            term.constant is None) and term.constant.typecode == 'z':
                        newCons = cvx.matrix(0., self.size, 'z')[:]
                    else:
                        newCons = cvx.matrix(0., self.size, 'd')[:]
                    self.constant = newCons
                if not term.constant is None:
                    try:
                        self.constant += term.constant
                    except TypeError as ex:
                        if str(ex).startswith('incompatible') or str(
                                ex).startswith('invalid'):  # incompatible typecodes
                            self.constant = self.constant + term.constant
                        else:
                            raise
                    """old implementation without try-block; the new one is safer !

                                        if ((isinstance(self.constant,spmatrix) and isinstance(term.constant,cvx.matrix))) or (
                                               term.constant.typecode=='z' and self.constant.typecode=='d'):
                                                  #inplace op not defined
                                                  self.constant=cvx.matrix(self.constant,tc=term.constant.typecode)
                                        self.constant+=term.constant
                                        """

            if term.affstring() not in ['0', '', '|0|', '0.0', '|0.0|']:
                if term.string[0] == '-':
                    import re
                    if ('+' not in term.string[1:]) and (
                            '-' not in term.string[1:]):
                        self.string += ' ' + term.affstring()
                    elif (term.string[1] == '(') and (
                            re.search('.*\)((\[.*\])|(.T))*$', term.string)):  # a group in a (...)
                        self.string += ' ' + term.affstring()
                    else:
                        self.string += ' + (' + \
                            term.affstring() + ')'
                else:
                    self.string += ' + ' + term.affstring()
            return self
        elif isinstance(term, QuadExp):
            if self.size != (1, 1):
                raise Exception('LHS must be scalar')
            self = QuadExp({}, self, self.affstring())
            self += term
            return self
        else:  # constant term
            term, termString = _retrieve_matrix(term, self.size)
            self += AffinExp({},
                             constant=term[:],
                             size=term.size,
                             string=termString)
            return self

    def __neg__(self):
        selfneg = (-1) * self
        if self.string != '':
            if self.string[0] == '-':
                import re
                if ('+' not in self.string[1:]
                        ) and ('-' not in self.string[1:]):
                    selfneg.string = self.string[1:]
                elif (self.string[1] == '(') and (
                        re.search('.*\)((\[.*\])|(.T))*$', self.string)):  # a group in a (...)
                    if self.string[-1] == ')':
                        # we remove the parenthesis
                        selfneg.string = self.string[2:-1]
                    else:
                        selfneg.string = self.string[
                            1:]  # we keep the parenthesis
                else:
                    selfneg.string = '-(' + self.string + ')'
            else:
                if ('+' in self.string) or ('-' in self.string):
                    selfneg.string = '-(' + self.string + ')'
                else:
                    selfneg.string = '-' + self.string
        return selfneg

    def __sub__(self, term):
        if isinstance(term, AffinExp) or isinstance(term, QuadExp):
            return self + (-term)
        else:  # constant term
            term, termString = _retrieve_matrix(term, self.size)
            return self - AffinExp({},
                                   constant=term[:],
                                   size=term.size,
                                   string=termString)

    def __rsub__(self, term):
        return term + (-self)

    def __truediv__(self, divisor):
        return self.__div__(divisor)

    def __div__(self, divisor):  # division (by a scalar)
        if isinstance(divisor, AffinExp):
            if divisor.isconstant():
                divi, diviString = divisor.value, divisor.string
            else:
                raise Exception('not implemented')
            if divi.size != (1, 1):
                raise Exception('not implemented')
            divi = divi[0]
            if divi == 0:
                raise Exception('Division By Zero')
            division = self * (1 / divi)
            if ('+' in self.string) or ('-' in self.string):
                division.string = '(' + self.string + ') /' + diviString
            else:
                division.string = self.string + ' / ' + diviString
            return division
        else:  # constant term
            divi, diviString = _retrieve_matrix(divisor, (1, 1))
            return self / \
                AffinExp({}, constant=divi[:], size=(1, 1), string=diviString)

    def __rdiv__(self, divider):
        divi, diviString = _retrieve_matrix(divider, None)
        return AffinExp(
            {},
            constant=divi[:],
            size=divi.size,
            string=diviString) / self

    def __getitem__(self, index):
        def indexstr(idx):
            if isinstance(idx, int):
                return str(idx)
            elif isinstance(idx, Expression):
                return idx.string

        def slicestr(sli):
            # single element
            if not (sli.start is None or sli.stop is None):
                sta = sli.start
                sto = sli.stop
                if isinstance(sta, int):
                    sta = new_param(str(sta), sta)
                if isinstance(sto, int):
                    sto = new_param(str(sto), sto)
                if (sto.__index__() == sta.__index__() + 1):
                    return sta.string
            # single element -1 (Expression)
            if (isinstance(sli.start, Expression) and sli.start.__index__()
                    == -1 and sli.stop is None and sli.step is None):
                return sli.start.string
            # single element -1
            if (isinstance(sli.start, int) and sli.start == -1
                    and sli.stop is None and sli.step is None):
                return '-1'
            ss = ''
            if not sli.start is None:
                ss += indexstr(sli.start)
            ss += ':'
            if not sli.stop is None:
                ss += indexstr(sli.stop)
            if not sli.step is None:
                ss += ':'
                ss += indexstr(sli.step)
            return ss

        if isinstance(index, Expression) or isinstance(index, int):
            ind = index.__index__()
            if ind == -1:  # (-1,0) does not work
                index = slice(ind, None, None)
            else:
                index = slice(ind, ind + 1, None)
        if isinstance(index, slice):
            idx = index.indices(self.size[0] * self.size[1])
            rangeT = list(range(idx[0], idx[1], idx[2]))
            # newfacs={}
            # for k in self.factors:
            # newfacs[k]=self.factors[k][rangeT,:]
            # if not self.constant is None:
            # newcons=self.constant[rangeT]
            # else:
            # newcons=None
            newsize = (len(rangeT), 1)
            indstr = slicestr(index)
        elif isinstance(index, tuple):
            if isinstance(index[0], Expression) or isinstance(index[0], int):
                ind = index[0].__index__()
                if ind == -1:
                    index = (slice(ind, None, None), index[1])
                else:
                    index = (slice(ind, ind + 1, None), index[1])
            if isinstance(index[1], Expression) or isinstance(index[1], int):
                ind = index[1].__index__()
                if ind == -1:
                    index = (index[0], slice(ind, None, None))
                else:
                    index = (index[0], slice(ind, ind + 1, None))
            idx0 = index[0].indices(self.size[0])
            idx1 = index[1].indices(self.size[1])
            rangei = range(idx0[0], idx0[1], idx0[2])
            rangej = range(idx1[0], idx1[1], idx1[2])
            rangeT = []
            for j in rangej:
                rangei_translated = []
                for vi in rangei:
                    rangei_translated.append(
                        vi + (j * self.size[0]))
                rangeT.extend(rangei_translated)

            # newfacs={}
            # for k in self.factors:
                # newfacs[k]=self.factors[k][rangeT,:]
            # if not self.constant is None:
                # newcons=self.constant[rangeT]
            # else:
                # newcons=None
            newsize = (len(range(*idx0)), len(range(*idx1)))
            indstr = slicestr(index[0]) + ',' + slicestr(index[1])

        newfacs = {}
        for k in self.factors:
            Ridx, J, V = self.factors[k].T.CCS  # fast row slicing
            II, VV, JJ = [], [], []
            for l, i in enumerate(rangeT):
                idx = range(Ridx[i], Ridx[i + 1])
                for j in idx:
                    II.append(l)
                    JJ.append(J[j])
                    VV.append(V[j])
            newfacs[k] = spmatrix(
                VV, II, JJ, (len(rangeT), self.factors[k].size[1]))

        if not self.constant is None:
            newcons = self.constant[rangeT]
        else:
            newcons = None

        if ('*' in self.affstring()) or ('+' in self.affstring()) or (
                '-' in self.affstring()) or ('/' in self.affstring()):
            newstr = '( ' + self.string + ' )[' + indstr + ']'
        else:
            newstr = self.string + '[' + indstr + ']'
        # check size
        if newsize[0] == 0 or newsize[1] == 0:
            raise IndexError('slice of zero-dimension')
        return AffinExp(newfacs, newcons, newsize, newstr)

    def __setitem__(self, key, value):
        raise AttributeError('slices of an expression are not writable')

    def __delitem__(self):
        raise AttributeError('slices of an expression are not writable')

    def __lt__(self, exp):
        if isinstance(exp, AffinExp):
            if exp.size == (1, 1) and self.size != (1, 1):
                oldstring = exp.string
                exp = cvx.matrix(1., self.size) * exp.diag(self.size[1])
                exp.string = '|' + oldstring + '|'
            if self.size == (1, 1) and exp.size != (1, 1):
                oldstring = self.string
                selfone = cvx.matrix(1., exp.size) * self.diag(exp.size[1])
                selfone.string = '|' + oldstring + '|'
                return (selfone < exp)
            return Constraint('lin<', None, self, exp)
        elif isinstance(exp, QuadExp):
            if (self.isconstant() and self.size == (1, 1)
                and (not exp.LR is None) and (not exp.LR[1] is None)
                ):
                cst = AffinExp(
                    factors={}, constant=cvx.matrix(
                        np.sqrt(
                            self.eval()), (1, 1)), size=(
                        1, 1), string=self.string)
                return (Norm(cst)**2) < exp
            elif self.size == (1, 1):
                return (-exp) < (-self)
            else:
                raise Exception('not implemented')
        elif isinstance(exp, GeoMeanExp):
            return exp > self
        elif isinstance(exp, NormP_Exp):
            return exp > self
        elif isinstance(exp, TracePow_Exp):
            return exp > self
        elif isinstance(exp, DetRootN_Exp):
            return exp > self
        else:
            term, termString = _retrieve_matrix(exp, self.size)
            exp2 = AffinExp(
                factors={},
                constant=term[:],
                size=self.size,
                string=termString)
            return Constraint('lin<', None, self, exp2)

    def __gt__(self, exp):
        if isinstance(exp, AffinExp):
            if exp.size == (1, 1) and self.size != (1, 1):
                oldstring = exp.string
                exp = cvx.matrix(1., self.size) * exp.diag(self.size[1])
                exp.string = '|' + oldstring + '|'
            if self.size == (1, 1) and exp.size != (1, 1):
                oldstring = self.string
                selfone = cvx.matrix(1., exp.size) * self.diag(exp.size[1])
                selfone.string = '|' + oldstring + '|'
                return (selfone > exp)
            return Constraint('lin>', None, self, exp)
        elif isinstance(exp, QuadExp):
            return exp < self
        elif isinstance(exp, NormP_Exp):
            return exp < self
        elif isinstance(exp, TracePow_Exp):
            return exp < self
        else:
            term, termString = _retrieve_matrix(exp, self.size)
            exp2 = AffinExp(
                factors={},
                constant=term[:],
                size=self.size,
                string=termString)
            return Constraint('lin>', None, self, exp2)

    def __eq__(self, exp):
        if isinstance(exp, AffinExp):
            if exp.size == (1, 1) and self.size != (1, 1):
                oldstring = exp.string
                exp = cvx.matrix(1., self.size) * exp.diag(self.size[1])
                exp.string = '|' + oldstring + '|'
            if self.size == (1, 1) and exp.size != (1, 1):
                oldstring = self.string
                selfone = cvx.matrix(1., exp.size) * self.diag(exp.size[1])
                selfone.string = '|' + oldstring + '|'
                return (selfone == exp)
            return Constraint('lin=', None, self, exp)
        else:
            term, termString = _retrieve_matrix(exp, self.size)
            exp2 = AffinExp(
                factors={},
                constant=term[:],
                size=self.size,
                string=termString)
            return Constraint('lin=', None, self, exp2)

    def __abs__(self):
        return Norm(self)

    def __pow__(self, exponent):
        if (self.size == (1, 1) and self.isconstant()):
            if (isinstance(exponent, AffinExp) and exponent.isconstant()):
                exponent = exponent.value[0]
            if isinstance(exponent, int) or isinstance(exponent, float):
                return AffinExp(
                    factors={},
                    constant=self.eval()[0]**exponent,
                    size=(
                        1,
                        1),
                    string='(' + self.string + ')**{0}'.format(exponent))
            else:
                raise Exception('type of exponent not handled')
        if self.size != (1, 1):
            raise Exception('not implemented')
        if (exponent == 2):
            Q = QuadExp({},
                        AffinExp(),
                        None, None)
            qq = self * self
            Q.quad = qq.quad
            Q.LR = (self, None)
            if ('*' in self.affstring()) or ('+' in self.affstring()) or (
                    '-' in self.affstring()) or ('/' in self.affstring()):
                Q.string = '(' + self.affstring() + ')**2'
            else:
                Q.string = self.affstring() + '**2'
            return Q
        else:
            return tracepow(self, exponent)

    def diag(self, dim):
        if self.size != (1, 1):
            raise Exception('not implemented')
        selfcopy = self.copy()
        idx = cvx.spdiag([1.] * dim)[:].I

        for k in self.factors:
            tc = 'z' if self.factors[k].typecode=='z' else 'd'
            selfcopy.factors[k] = spmatrix(
                [], [], [], (dim**2, self.factors[k].size[1]),tc=tc)
            for i in idx:
                selfcopy.factors[k][i, :] = self.factors[k]
        if not self.constant is None:
            tc = 'z' if self.constant.typecode=='z' else 'd'
            selfcopy.constant = cvx.matrix(0., (dim**2, 1),tc=tc)
            for i in idx:
                selfcopy.constant[i] = self.constant[0]
        else:
            selfcopy.constant = None
        selfcopy._size = (dim, dim)
        selfcopy.string = 'diag(' + selfcopy.string + ')'
        return selfcopy

    def __and__(self, exp):
        """horizontal concatenation"""
        selfcopy = self.copy()
        selfcopy &= exp
        return selfcopy

    def __rand__(self, exp):
        Exp, ExpString = _retrieve_matrix(exp, self.size[0])
        exp2 = AffinExp(
            factors={},
            constant=Exp[:],
            size=Exp.size,
            string=ExpString)
        return (exp2 & self)

    def __iand__(self, exp):
        if isinstance(exp, AffinExp):
            if exp.size[0] != self.size[0]:
                raise Exception('incompatible size for concatenation')
            for k in list(set(exp.factors.keys()).union(
                    set(self.factors.keys()))):
                if (k in self.factors) and (k in exp.factors):
                    if self.factors[k].typecode == 'z' or exp.factors[
                            k].typecode == 'z':
                        facr = self.factors[k].real()
                        if self.factors[k].typecode == 'z':
                            faci = self.factors[k].imag()
                        else:
                            faci = spmatrix([], [], [], facr.size)
                        expr = exp.factors[k].real()
                        if exp.factors[k].typecode == 'z':
                            expi = exp.factors[k].imag()
                        else:
                            expi = spmatrix([], [], [], expr.size)
                        newfac = (cvx.sparse([[facr, expr]]) +
                                  1j * cvx.sparse([[faci, expi]]))
                    else:
                        newfac = cvx.sparse(
                            [[self.factors[k], exp.factors[k]]])
                    self.factors[k] = newfac
                elif k in exp.factors:
                    s1 = self.size[0] * self.size[1]
                    s2 = exp.factors[k].size[1]
                    if exp.factors[
                            k].typecode == 'z':  # it seems there is a bug with sparse with complex inputs
                        newfac = (cvx.sparse([[spmatrix([], [], [], (s1, s2)), exp.factors[k].real()]]) +
                                  1j * cvx.sparse([[spmatrix([], [], [], (s1, s2)), exp.factors[k].imag()]]))
                    else:
                        newfac = cvx.sparse(
                            [[spmatrix([], [], [], (s1, s2)), exp.factors[k]]])
                    self.factors[k] = newfac
                else:
                    s1 = exp.size[0] * exp.size[1]
                    s2 = self.factors[k].size[1]
                    if self.factors[
                            k].typecode == 'z':  # it seems there is a bug with sparse with complex inputs
                        newfac = (cvx.sparse([[self.factors[k].real(), spmatrix([], [], [], (s1, s2))]]) +
                                  1j * cvx.sparse([[self.factors[k].imag(), spmatrix([], [], [], (s1, s2))]]))
                    else:
                        newfac = cvx.sparse(
                            [[self.factors[k], spmatrix([], [], [], (s1, s2))]])
                    self.factors[k] = newfac
            if self.constant is None and exp.constant is None:
                pass
            else:
                s1 = self.size[0] * self.size[1]
                s2 = exp.size[0] * exp.size[1]
                if not self.constant is None:
                    newCons = self.constant
                else:
                    newCons = spmatrix([], [], [], (s1, 1))
                if not exp.constant is None:
                    # it seems there is a bug with sparse with complex inputs
                    if newCons.typecode == 'z' or exp.constant.typecode == 'z':
                        expr = exp.constant.real()
                        if exp.constant.typecode == 'z':
                            expi = exp.constant.imag()
                        else:
                            expi = spmatrix([], [], [], expr.size)
                        csr = newCons.real()
                        if newCons.typecode == 'z':
                            csi = newCons.imag()
                        else:
                            csi = spmatrix([], [], [], csr.size)
                        newCons = (cvx.sparse([[csr, expr]]) +
                                   1j * cvx.sparse([[csi, expi]]))
                    else:
                        newCons = cvx.sparse([[newCons, exp.constant]])
                else:
                    if newCons.typecode == 'z':  # it seems there is a bug with sparse with complex inputs
                        newCons = (cvx.sparse([[newCons.real(), spmatrix([], [], [], (s2, 1))]]) +
                                   1j * cvx.sparse([[newCons.imag(), spmatrix([], [], [], (s2, 1))]]))
                    else:
                        newCons = cvx.sparse(
                            [[newCons, spmatrix([], [], [], (s2, 1))]])
                self.constant = newCons
            self._size = (exp.size[0], exp.size[1] + self.size[1])
            sstring = self.string
            estring = exp.string
            if sstring[0] == '[' and sstring[-1] == ']':
                sstring = sstring[1:-1]
            if estring[0] == '[' and estring[-1] == ']':
                estring = estring[1:-1]
            self.string = '[' + sstring + ',' + estring + ']'
            return self
        else:
            Exp, ExpString = _retrieve_matrix(exp, self.size[0])
            exp2 = AffinExp(
                factors={},
                constant=Exp[:],
                size=Exp.size,
                string=ExpString)
            self &= exp2
            return self

    def __floordiv__(self, exp):
        """vertical concatenation"""
        if isinstance(exp, AffinExp):
            concat = (self.T & exp.T).T
            concat._size = (exp.size[0] + self.size[0], exp.size[1])
            sstring = self.string
            estring = exp.string
            if sstring[
                    0] == '[' and sstring[-1] == ']':  # TODO problem when the [ does not match with ]
                sstring = sstring[1:-1]
            if estring[0] == '[' and estring[-1] == ']':
                estring = estring[1:-1]
            concat.string = '[' + sstring + ';' + estring + ']'
            return concat
        else:
            Exp, ExpString = _retrieve_matrix(exp, self.size[1])
            exp2 = AffinExp(
                factors={},
                constant=Exp[:],
                size=Exp.size,
                string=ExpString)
            return (self // exp2)

    def __ifloordiv__(self, exp):
        """inplace vertical concatenation"""
        if isinstance(exp, AffinExp):
            if exp.size[1] != self.size[1]:
                raise Exception('incompatible size for concatenation')
            for k in list(set(exp.factors.keys()).union(
                    set(self.factors.keys()))):
                if (k in self.factors) and (k in exp.factors):
                    if self.factors[k].typecode == 'z' or exp.factors[
                            k].typecode == 'z':
                        facr = self.factors[k].real()
                        if self.factors[k].typecode == 'z':
                            faci = self.factors[k].imag()
                        else:
                            faci = spmatrix([], [], [], facr.size)
                        expr = exp.factors[k].real()
                        if exp.factors[k].typecode == 'z':
                            expi = exp.factors[k].imag()
                        else:
                            expi = spmatrix([], [], [], expr.size)
                        newfac = (cvx.sparse([facr, expr]) +
                                  1j * cvx.sparse([faci, expi]))
                    else:
                        newfac = cvx.sparse([self.factors[k], exp.factors[k]])
                    self.factors[k] = newfac
                elif k in exp.factors:
                    s1 = self.size[0] * self.size[1]
                    s2 = exp.factors[k].size[1]
                    if exp.factors[
                            k].typecode == 'z':  # it seems there is a bug with sparse with complex inputs
                        newfac = (cvx.sparse([spmatrix([], [], [], (s1, s2)), exp.factors[k].real(
                        )]) + 1j * cvx.sparse([spmatrix([], [], [], (s1, s2)), exp.factors[k].imag()]))
                    else:
                        newfac = cvx.sparse(
                            [spmatrix([], [], [], (s1, s2)), exp.factors[k]])
                    self.factors[k] = newfac
                else:
                    s1 = exp.size[0] * exp.size[1]
                    s2 = self.factors[k].size[1]
                    if self.factors[
                            k].typecode == 'z':  # it seems there is a bug with sparse with complex inputs
                        newfac = (cvx.sparse([self.factors[k].real(), spmatrix([], [], [], (s1, s2))]) +
                                  1j * cvx.sparse([self.factors[k].imag(), spmatrix([], [], [], (s1, s2))]))
                    else:
                        newfac = cvx.sparse(
                            [self.factors[k], spmatrix([], [], [], (s1, s2))])
                    self.factors[k] = newfac

            if self.constant is None and exp.constant is None:
                pass
            else:
                s1 = self.size[0] * self.size[1]
                s2 = exp.size[0] * exp.size[1]
                if not self.constant is None:
                    newCons = self.constant
                else:
                    newCons = spmatrix([], [], [], (s1, 1))
                if not exp.constant is None:
                    # it seems there is a bug with sparse with complex inputs
                    if newCons.typecode == 'z' or exp.constant.typecode == 'z':
                        expr = exp.constant.real()
                        if exp.constant.typecode == 'z':
                            expi = exp.constant.imag()
                        else:
                            expi = spmatrix([], [], [], expr.size)
                        csr = newCons.real()
                        if newCons.typecode == 'z':
                            csi = newCons.imag()
                        else:
                            csi = spmatrix([], [], [], csr.size)
                        newCons = (cvx.sparse([csr, expr]) +
                                   1j * cvx.sparse([csi, expi]))
                    else:
                        newCons = cvx.sparse([newCons, exp.constant])
                else:
                    if newCons.typecode == 'z':  # it seems there is a bug with sparse with complex inputs
                        newCons = (cvx.sparse([newCons.real(), spmatrix([], [], [], (s2, 1))]) +
                                   1j * cvx.sparse([newCons.imag(), spmatrix([], [], [], (s2, 1))]))
                    else:
                        newCons = cvx.sparse(
                            [newCons, spmatrix([], [], [], (s2, 1))])
                self.constant = newCons

            self._size = (exp.size[0] + self.size[0], exp.size[1])
            sstring = self.string
            estring = exp.string
            if sstring[0] == '[' and sstring[-1] == ']':
                sstring = sstring[1:-1]
            if estring[0] == '[' and estring[-1] == ']':
                estring = estring[1:-1]
            self.string = '[' + sstring + ';' + estring + ']'
            return self
        else:
            Exp, ExpString = _retrieve_matrix(exp, self.size[1])
            exp2 = AffinExp(
                factors={},
                constant=Exp[:],
                size=Exp.size,
                string=ExpString)
            self //= exp2
            return self

    def __rfloordiv__(self, exp):
        Exp, ExpString = _retrieve_matrix(exp, self.size[1])
        exp2 = AffinExp(
            factors={},
            constant=Exp[:],
            size=Exp.size,
            string=ExpString)
        return (exp2 // self)

    def apply_function(self, fun):
        return GeneralFun(fun, self, fun())

    def __lshift__(self, exp):

        if isinstance(exp, Set):
            return exp >> self

        if self.size[0] != self.size[1]:
            raise Exception('both sides of << must be square')
        if isinstance(exp, AffinExp):
            return Constraint('sdp<', None, self, exp)
        else:
            n = self.size[0]
            Exp, ExpString = _retrieve_matrix(exp, (n, n))
            exp2 = AffinExp(
                factors={},
                constant=Exp[:],
                size=Exp.size,
                string=ExpString)
            return (self << exp2)

    def __rshift__(self, exp):
        if self.size[0] != self.size[1]:
            raise Exception('both sides of << must be square')
        if isinstance(exp, AffinExp):
            return Constraint('sdp>', None, self, exp)
        else:
            n = self.size[0]
            Exp, ExpString = _retrieve_matrix(exp, (n, n))
            exp2 = AffinExp(
                factors={},
                constant=Exp[:],
                size=Exp.size,
                string=ExpString)
            return (self >> exp2)

#---------------------------------------------
#        Class Norm and ProductOfAffinExp
#---------------------------------------------


class Norm(Expression):
    """
    Euclidean (or Frobenius) norm of an Affine Expression.
    This class derives from :class:`Expression<picos.Expression>`.

    **Overloaded operators**

            :``**``: exponentiation (only implemented when the exponent is 2)
            :``<``: less **or equal** (than a scalar affine expression)
    """

    def __init__(self, exp):
        Expression.__init__(self, '||' + exp.string + '||')
        self.exp = exp
        """The affine expression of which we take the norm"""

    def __repr__(self):
        normstr = '# norm of a ({0} x {1})- expression: ||'.format(
            self.exp.size[0], self.exp.size[1])
        normstr += self.exp.affstring()
        normstr += '||'
        normstr += ' #'
        return normstr

    def __str__(self):
        if self.is_valued():
            return str(self.value[0])
        else:
            return repr(self)

    def eval(self, ind=None):
        vec = self.exp.eval(ind)
        return cvx.matrix(np.linalg.norm(vec), (1, 1))

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __pow__(self, exponent):
        if (exponent != 2):
            raise Exception('not implemented')

        qq = (self.exp[:].T) * (self.exp[:])
        if isinstance(qq, AffinExp):
            qq = QuadExp({}, qq, qq.string)
        Qnorm = QuadExp(qq.quad,
                        qq.aff,
                        '||' + self.exp.affstring() + '||**2',
                        LR=(self.exp, None)
                        )

        return Qnorm

    def __lt__(self, exp):
        if isinstance(exp, AffinExp):
            if self.exp.size != (1, 1):
                return Constraint('SOcone', None, self.exp, exp)
            elif not self.exp.is_real():
                return Constraint('SOcone', None, self.exp.real // self.exp.imag, exp)
            else:
                cons = (self.exp // -self.exp) < (exp // exp)
                if exp.is1():
                    cons.myconstring = '||' + self.exp.string + '|| < 1'
                else:
                    cons.myconstring = '||' + self.exp.string + '|| < ' + exp.string
                cons.myfullconstring = '# (1x1)-SOC constraint ' + \
                    cons.myconstring + ' #'
                return cons
        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self < exp1


class LogSumExp(Expression):
    """Log-Sum-Exp applied to an affine expression.
       If the affine expression ``z`` is of size :math:`N`,
       with elements :math:`z_1, z_2, \ldots, z_N`,
       then LogSumExp(z) represents the expression
       :math:`\log ( \sum_{i=1}^n e^{z_i} )`.
       This class derives from :class:`Expression<picos.Expression>`.


    **Overloaded operator**

            :``<``: less **or equal** than (the rhs **must be 0**, for geometrical programming)

    """

    def __init__(self, exp):

        if not(isinstance(exp, AffinExp)):
            term, termString = _retrieve_matrix(exp, None)
            exp = AffinExp(factors={}, constant=term,
                           size=term.size, string=termString)

        Expression.__init__(self, 'LSE[' + exp.string + ']')
        self.Exp = exp

    def __str__(self):
        if self.is_valued():
            return str(self.value[0])
        else:
            return repr(self)

    def __repr__(self):
        lsestr = '# log-sum-exp of an affine expression: '
        lsestr += self.Exp.affstring()
        lsestr += ' #'
        return lsestr

    def affstring(self):
        return 'LSE[' + self.Exp.affstring() + ']'

    def eval(self, ind=None):
        return cvx.matrix(np.log(np.sum(np.exp(self.Exp.eval(ind)))),
                          (1, 1)
                          )

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value,
        "value of the logsumexp expression")

    def __lt__(self, exp):
        if exp != 0 and not(isinstance(exp, AffinExp) and exp.is0()):
            raise Exception('rhs must be 0')
        else:
            return Constraint('lse', None, self.Exp, 0)


class QuadExp(Expression):
    """
    Quadratic expression.
    This class derives from :class:`Expression<picos.Expression>`.

    **Overloaded operators**

            :``+``: addition (with an affine or a quadratic expression)
            :``-``: substraction (with an affine or a quadratic expression) or unitary minus
            :``*``: multiplication (by a scalar or a constant affine expression)
            :``<``: less **or equal** than (another quadratic or affine expression).
            :``>``: greater **or equal** than (another quadratic or affine expression).

    """

    def __init__(self, quad, aff, string, LR=None):
        Expression.__init__(self, string)
        self._size = (1,1)
        self.quad = quad
        """dictionary of quadratic forms,
                stored as matrices representing bilinear forms
                with respect to two vectorized variables,
                and indexed by tuples of
                instances of the class :class:`Variable<picos.Variable>`.
                """
        self.aff = aff
        """affine expression representing the affine part of the quadratic expression"""
        self.LR = LR
        """stores a factorization of the quadratic expression, if the
                   expression was entered in a factorized form. We have:

                     * ``LR=None`` when no factorization is known
                     * ``LR=(aff,None)`` when the expression is equal to ``||aff||**2``
                     * ``LR=(aff1,aff2)`` when the expression is equal to ``aff1*aff2``.
                """

    def __str__(self):
        if self.is_valued():
            return str(self.value[0])
        else:
            return repr(self)

    def __repr__(self):
        return '#quadratic expression: ' + self.string + ' #'

    def copy(self):
        import copy
        qdcopy = {}
        for ij, m in six.iteritems(self.quad):
            qdcopy[ij] = copy.deepcopy(m)

        if self.aff is None:
            affcopy = None
        else:
            affcopy = self.aff.copy()

        if self.LR is None:
            lrcopy = None
        else:
            if self.LR[1] is None:
                lrcopy = (self.LR[0].copy(), None)
            else:
                lrcopy = (self.LR[0].copy(), self.LR[1].copy())
        return QuadExp(qdcopy, affcopy, self.string, lrcopy)

    def eval(self, ind=None):
        if not self.LR is None:
            ex1 = self.LR[0].eval(ind)
            if self.LR[1] is None:
                val = (ex1.T * ex1)
            else:
                if self.LR[0].size != (1, 1) or self.LR[1].size != (1, 1):
                    raise Exception(
                        'QuadExp of size (1,1) only are implemented')
                else:
                    ex2 = self.LR[1].eval(ind)
                    val = (ex1 * ex2)

        else:
            if not self.aff is None:
                val = self.aff.eval(ind)
            else:
                val = cvx.matrix(0., (1, 1))

            for i, j in self.quad:
                if ind is None:
                    if i.value is None:
                        raise Exception(i + ' is not valued')
                    if j.value is None:
                        raise Exception(j + ' is not valued')
                    xi = i.value[:]
                    xj = j.value[:]
                else:
                    if ind not in i.value_alt:
                        raise Exception(
                            i + ' does not have a value for the index ' + str(ind))
                    if ind not in j.value_alt:
                        raise Exception(
                            j + ' does not have a value for the index ' + str(ind))
                    xi = i.value_alt[ind][:]
                    xj = j.value_alt[ind][:]
                val = val + xi.T * self.quad[i, j] * xj

        return cvx.matrix(val, (1, 1))

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def nnz(self):
        nz = 0
        for ij in self.quad:
            nz += len(self.quad[ij].I)
        return nz

    def __mul__(self, fact):
        if isinstance(fact, AffinExp):
            if fact.isconstant() and fact.size == (1, 1):
                selfcopy = self.copy()
                for ij in selfcopy.quad:
                    selfcopy.quad[ij] = fact.eval()[0] * selfcopy.quad[ij]
                selfcopy.aff = fact * selfcopy.aff
                selfcopy.string = fact.affstring() + '*(' + self.string + ')'
                if not self.LR is None:
                    if self.LR[1] is None and (
                            fact.eval()[0] >= 0):  # Norm squared
                        selfcopy.LR = (np.sqrt(fact.eval()) * self.LR[0], None)
                    elif self.LR[1] is None and (fact.eval()[0] < 0):
                        selfcopy.LR = None
                    else:
                        selfcopy.LR = (fact * self.LR[0], self.LR[1])
                return selfcopy
            else:
                raise Exception('not implemented')
        else:  # constant term
            fact, factString = _retrieve_matrix(fact, (1, 1))
            return self * AffinExp({},
                                   constant=fact[:],
                                   size=fact.size,
                                   string=factString)

    def __div__(self, divisor):  # division (by a scalar)
        if isinstance(divisor, AffinExp):
            if divisor.isconstant():
                divi, diviString = divisor.eval(), divisor.string
            else:
                raise Exception('not implemented')
            if divi.size != (1, 1):
                raise Exception('not implemented')
            if divi[0] == 0:
                raise Exception('Division By Zero')
            divi = divi[0]
            division = self * (1 / divi)
            lstring = self.string
            if ('+' in self.string) or ('-' in self.string):
                lstring = '(' + self.string + ')'
            if ('+' in diviString) or ('-' in diviString):
                diviString = '(' + diviString + ')'
            division.string = lstring + ' / ' + diviString
            return division
        else:  # constant term
            divi, diviString = _retrieve_matrix(divisor, (1, 1))
            return self / \
                AffinExp({}, constant=divi[:], size=(1, 1), string=diviString)

    def __add__(self, term):
        selfcopy = self.copy()
        selfcopy += term
        return selfcopy

    # inplace sum
    def __iadd__(self, term):
        if isinstance(term, QuadExp):
            for ij in self.quad:
                if ij in term.quad:
                    try:
                        self.quad[ij] += term.quad[ij]
                    except TypeError as ex:
                        if str(ex).startswith('incompatible') or str(
                                ex).startswith('invalid'):  # incompatible typecodes
                            self.quad[ij] = self.quad[ij] + term.quad[ij]
                        else:
                            raise
            for ij in term.quad:
                if not (ij in self.quad):
                    self.quad[ij] = term.quad[ij]
            self.aff += term.aff
            self.LR = None
            if term.string not in ['0', '']:
                if term.string[0] == '-':
                    import re
                    if ('+' not in term.string[1:]) and (
                            '-' not in term.string[1:]):
                        self.string += ' ' + term.string
                    elif (term.string[1] == '(') and (
                            re.search('.*\)((\[.*\])|(.T))*$', term.string)):  # a group in a (...)
                        self.string += ' ' + term.string
                    else:
                        self.string += ' + (' + \
                            term.string + ')'
                else:
                    self.string += ' + ' + term.string
            return self
        elif isinstance(term, AffinExp):
            if term.size != (1, 1):
                raise Exception('RHS must be scalar')
            expQE = QuadExp({}, term, term.affstring())
            self += expQE
            return self
        else:
            term, termString = _retrieve_matrix(term, (1, 1))
            expAE = AffinExp(
                factors={},
                constant=term,
                size=term.size,
                string=termString)
            self + expAE
            return self

    def __rmul__(self, fact):
        return self * fact

    def __neg__(self):
        selfneg = (-1) * self
        if self.string[0] == '-':
            import re
            if ('+' not in self.string[1:]) and ('-' not in self.string[1:]):
                selfneg.string = self.string[1:]
            elif (self.string[1] == '(') and (
                    re.search('.*\)((\[.*\])|(.T))*$', self.string)):  # a group in a (...)
                if self.string[-1] == ')':
                    # we remove the parenthesis
                    selfneg.string = self.string[2:-1]
                else:
                    selfneg.string = self.string[1:]  # we keep the parenthesis
            else:
                selfneg.string = '-(' + self.string + ')'
        else:
            if ('+' in self.string) or ('-' in self.string):
                selfneg.string = '-(' + self.string + ')'
            else:
                selfneg.string = '-' + self.string
        return selfneg

    def __sub__(self, term):
        return self + (-term)

    def __rsub__(self, term):
        return term + (-self)

    def __radd__(self, term):
        return self + term

    def __lt__(self, exp):
        if isinstance(exp, QuadExp):
            if ((not self.LR is None) and (self.LR[1] is None)
                    and (not exp.LR is None)):
                if (not exp.LR[1] is None):  # rotated cone
                    return Constraint(
                        'RScone', None, self.LR[0], exp.LR[0], exp.LR[1])
                else:  # simple cone
                    return Constraint('SOcone', None, self.LR[0], exp.LR[0])

            else:
                return Constraint('quad', None, self - exp, 0)
        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('RHS must be scalar')
            exp2 = AffinExp(
                factors={}, constant=cvx.matrix(
                    1., (1, 1)), size=(
                    1, 1), string='1')
            expQE = QuadExp({}, exp, exp.affstring(), LR=(exp, exp2))
            return self < expQE
        else:
            term, termString = _retrieve_matrix(exp, (1, 1))
            expAE = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self < expAE

    def __gt__(self, exp):
        if isinstance(exp, QuadExp):
            if (not exp.LR is None) and (exp.LR[1] is None):  # a squared norm
                return exp < self
            return (-self) < (-exp)
        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('RHS must be scalar')
            if exp.isconstant():
                cst = AffinExp(
                    factors={}, constant=cvx.matrix(
                        np.sqrt(
                            exp.eval()), (1, 1)), size=(
                        1, 1), string=exp.string + '**0.5')

                if not(self.LR is None):
                    return (Norm(cst)**2) < self
                else:
                    return exp < self
            else:
                return (-self) < (-exp)
        else:
            term, termString = _retrieve_matrix(exp, (1, 1))
            expAE = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self > expAE


class GeneralFun(Expression):
    """A class storing a scalar function,
       applied to an affine expression.
       It derives from :class:`Expression<picos.Expression>`.
    """

    def __init__(self, fun, Exp, funstring):
        Expression.__init__(self, self.funstring + '( ' + Exp.string + ')')
        self.fun = fun
        r"""The function ``f`` applied to the affine expression.
                This function must take in argument a
                :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>` ``X``.
                Moreover, the call ``fx,grad,hess=f(X)``
                must return the function value :math:`f(X)` in ``fx``,
                the gradient :math:`\nabla f(X)` in the
                :func:`cvxopt matrix <cvxopt:cvxopt.matrix>` ``grad``,
                and the Hessian :math:`\nabla^2 f(X)` in the
                :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>` ``hess``.
                """
        self.Exp = Exp
        """The affine expression to which the function is applied"""
        self.funstring = funstring
        """a string representation of the function name"""
        #self.string=self.funstring+'( '+self.Exp.affstring()+' )'

    def __repr__(self):
        return '# general function ' + self.string + ' #'

    def __str__(self):
        if self.is_valued():
            return str(self.value[0])
        else:
            return repr(self)

    def eval(self, ind=None):
        val = self.Exp.eval(ind)
        o, g, h = self.fun(val)
        return cvx.matrix(o, (1, 1))

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)


class _ConvexExp(Expression):
    """A parent class for all convex expressions which can be handled in picos"""

    def __init__(self, string, expstring):
        Expression.__init__(self, string)
        self.expstring = expstring

    def __repr__(self):
        return '# ' + self.expstring + ': ' + self.string + ' #'

    def __str__(self):
        if self.is_valued():
            try:
                return str(self.value[0])
            except:
                return str(self.value)
        else:
            return repr(self)


class GeoMeanExp(_ConvexExp):
    """A class storing the geometric mean of a multidimensional expression.
       It derives from :class:`Expression<picos.Expression>`.

       **Overloaded operator**

            :``>``: greater **or equal** than (the rhs must be a scalar affine expression)

    """

    def __init__(self, exp):
        _ConvexExp.__init__(
            self,
            'geomean( ' + exp.string + ')',
            'geometric mean')
        self.exp = exp
        """The affine expression to which the geomean is applied"""

    def eval(self, ind=None):
        val = self.exp.eval(ind)
        dim = self.exp.size[0] * self.exp.size[1]
        return cvx.matrix(np.prod(val)**(1. / dim), (1, 1))

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __gt__(self, exp):
        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('upper bound of a geomean must be scalar')
            if self.exp.size == (1, 1):
                return self.exp > exp
            # construct a list of keys to index new variables with nodes of a binary tree
            # the new variables are added in a temporary problem Ptmp
            from .problem import Problem
            Ptmp = Problem()
            m = self.exp.size[0] * self.exp.size[1]
            lm = [[i] for i in range(m - 1, -1, -1)]
            K = []
            depth = 0
            u = {}
            while len(lm) > 1:
                depth += 1
                nlm = []
                while lm:
                    i1 = lm.pop()[-1]
                    if lm:
                        i2 = lm.pop()[0]
                    else:
                        i2 = 'x'
                    nlm.insert(0, (i2, i1))
                    k = str(depth) + ':' + str(i1) + '-' + str(i2)
                    K.append(k)
                    u[k] = Ptmp.add_variable('u[' + k + ']', 1)
                lm = nlm
            root = K[-1]
            maxd = int(K[-1].split(':')[0])
            Ptmp.remove_variable(u[root].name)
            u[root] = exp

            for k in K:
                i1 = int(k.split('-')[0].split(':')[1])
                i2 = k.split('-')[1]
                if i2 != 'x':
                    i2 = int(i2)
                if k[:2] == '1:':
                    if i2 != 'x':
                        Ptmp.add_constraint(
                            u[k]**2 < self.exp[i1] * self.exp[i2])
                    else:
                        Ptmp.add_constraint(u[k]**2 < self.exp[i1] * exp)
                else:
                    d = int(k.split(':')[0])
                    if i2 == 'x' and d < maxd:
                        k2pot = [ki for ki in K if ki.startswith(
                            str(d - 1) + ':') and int(ki.split(':')[1].split('-')[0]) >= i1]
                        k1 = k2pot[0]
                        if len(k2pot) == 2:
                            k2 = k2pot[1]
                            Ptmp.add_constraint(u[k]**2 < u[k1] * u[k2])
                        else:
                            Ptmp.add_constraint(u[k]**2 < u[k1] * exp)
                    else:
                        k1 = [ki for ki in K if ki.startswith(
                            str(d - 1) + ':' + str(i1))][0]
                        k2 = [ki for ki in K if ki.startswith(
                            str(d - 1) + ':') and ki.endswith('-' + str(i2))][0]
                        Ptmp.add_constraint(u[k]**2 < u[k1] * u[k2])
            return GeoMeanConstraint(
                exp, self.exp, Ptmp, exp.string + '<' + self.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self > exp1


class NormP_Exp(_ConvexExp):
    """A class storing the p-norm of a multidimensional expression.
       It derives from :class:`Expression<picos.Expression>`.
       Use the function :func:`picos.norm() <picos.tools.norm>` to create a instance of this class.
       This class can also be used to store the :math:`L_{p,q}` norm of a matrix.

       Generalized norms are also defined for :math:`p<1`, by using the usual formula
       :math:`\operatorname{norm}(\mathbf{x},p) := \Big(\sum_i x_i^p\Big)^{1/p}`. Note that this function
       is concave (for :math:`p<1`) over the set of vectors with nonnegative coordinates.
       When a constraint of the form :math:`\operatorname{norm}(\mathbf{x},p) > t` with :math:`p\leq 1`
       is entered, PICOS implicitely forces :math:`\mathbf{x}` to be a nonnegative vector.

       **Overloaded operator**

            :``<``: less **or equal** than (the rhs must be a scalar affine expression, AND
                    p must be greater or equal than 1)
            :``>``: greater **or equal** than (the rhs must be a scalar affine expression, AND
                    p must be less or equal than 1)
    """

    def __init__(self, exp, numerator, denominator=1, num2=None, den2=1):
        pstr = str(numerator)
        if denominator > 1:
            pstr += '/' + str(denominator)
        p = float(numerator) / float(denominator)
        if not num2 is None:
            qstr = str(num2)
            if den2 > 1:
                qstr += '/' + str(den2)
            q = float(num2) / float(den2)
            if p >= 1 and q >= 1:
                _ConvexExp.__init__(
                    self,
                    'norm_' + pstr + ',' + qstr + '( ' + exp.string + ')',
                    '(p,q)-norm expression')
            else:
                raise ValueError('(p,q) norm is only implemented for p,q >=1')
        else:
            if p >= 1:
                _ConvexExp.__init__(
                    self,
                    'norm_' + pstr + '( ' + exp.string + ')',
                    'p-norm expression')
            else:
                _ConvexExp.__init__(self,
                                    'norm_' + pstr + '( ' + exp.string + ')',
                                    'generalized p-norm expression')
        self.exp = exp
        """The affine expression to which the p-norm is applied"""

        self.numerator = numerator
        """numerator of p"""

        self.denominator = denominator
        """denominator of p"""

        self.num2 = num2
        """numerator of q"""

        self.den2 = den2
        """denominator of q"""

    def eval(self, ind=None):
        val = self.exp.eval(ind)
        p = float(self.numerator) / float(self.denominator)
        if self.num2 is not None:
            q = float(self.num2) / float(self.den2)
            return np.linalg.norm([np.linalg.norm(list(val[i, :]), q)
                                   for i in range(val.size[0])], p)

        else:
            return np.linalg.norm([vi for vi in val], p)

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __lt__(self, exp):
        if float(self.numerator) / self.denominator < 1:
            raise Exception(
                '<= operator can be used only when the function is convex (p>=1)')
        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('upper bound of a norm must be scalar')
            if self.exp.size == (1, 1):
                return abs(self.exp) < exp
            p = float(self.numerator) / self.denominator
            from .problem import Problem
            Ptmp = Problem()
            m = self.exp.size[0] * self.exp.size[1]

            if self.num2 is not None:  # (p,q)-norm
                q = float(self.num2) / float(self.den2)
                N = self.exp.size[0]
                u = Ptmp.add_variable('v', N)
                for i in range(N):
                    Ptmp.add_constraint(norm(self.exp[i, :], q) <= u[i])
                if p == 1:
                    Ptmp.add_constraint(1 | u <= exp)
                elif p == float('inf'):
                    Ptmp.add_constraint(u <= exp)
                elif p == 2:
                    Ptmp.add_constraint(abs(u) <= exp)
                else:
                    Ptmp.add_constraint(norm(u, p) <= exp)
                return NormPQ_Constraint(
                    exp, self.exp, p, q, Ptmp, self.string + '<' + exp.string)
            if p == 1:
                v = Ptmp.add_variable('v', m)
                Ptmp.add_constraint(self.exp[:] <= v)
                Ptmp.add_constraint(-self.exp[:] <= v)
                Ptmp.add_constraint((1 | v) < exp)
            elif p == float('inf'):
                Ptmp.add_constraint(self.exp <= exp)
                Ptmp.add_constraint(-self.exp <= exp)
            else:
                x = Ptmp.add_variable('x', m)
                v = Ptmp.add_variable('v', m)

                amb = self.numerator - self.denominator
                b = self.denominator
                oneamb = '|1|(' + str(amb) + ',1)'
                oneb = '|1|(' + str(b) + ',1)'
                for i in range(m):
                    if amb > 0:
                        if b == 1:
                            vec = (v[i]) // (exp * oneamb)
                        else:
                            vec = (v[i] * oneb) // (exp * oneamb)
                    else:
                        if b == 1:
                            vec = v[i]
                        else:
                            vec = (v[i] * oneb)
                    Ptmp.add_constraint(abs(self.exp[i]) < x[i])
                    Ptmp.add_constraint(x[i] < geomean(vec))
                Ptmp.add_constraint((1 | v) < exp)

            return NormP_Constraint(
                exp,
                self.exp,
                self.numerator,
                self.denominator,
                Ptmp,
                self.string +
                '<' +
                exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self < exp1

    def __gt__(self, exp):
        p = float(self.numerator) / self.denominator
        if p > 1 or p == 0:
            raise Exception(
                '>= operator can be used only when the function is concave (p<=1, p != 0)')

        if isinstance(exp, AffinExp):
            from .problem import Problem
            if exp.size != (1, 1):
                raise Exception(
                    'lower bound of a generalized p-norm must be scalar')
            Ptmp = Problem()
            m = self.exp.size[0] * self.exp.size[1]
            if p == 1:
                Ptmp.add_constraint(self.exp > 0)
                Ptmp.add_constraint((1 | self.exp) > exp)
                print(
                    "\033[1;31m*** Warning -- generalized norm inequality, expression is forced to be >=0 \033[0m")
            elif p == float('-inf'):
                Ptmp.add_constraint(self.exp >= exp)
                print(
                    "\033[1;31m*** Warning -- generalized norm inequality, norm_-inf(x) is interpreted as min(x), not min(abs(x)) \033[0m")
            elif p >= 0:
                v = Ptmp.add_variable('v', m)

                bma = -(self.numerator - self.denominator)
                a = self.numerator
                onebma = '|1|(' + str(bma) + ',1)'
                onea = '|1|(' + str(a) + ',1)'
                for i in range(m):
                    if a == 1:
                        vec = (self.exp[i]) // (exp * onebma)
                    else:
                        vec = (self.exp[i] * onea) // (exp * onebma)

                    Ptmp.add_constraint(v[i] < geomean(vec))
                Ptmp.add_constraint(exp < (1 | v))
            else:
                v = Ptmp.add_variable('v', m)

                b = abs(self.denominator)
                a = abs(self.numerator)
                oneb = '|1|(' + str(b) + ',1)'
                onea = '|1|(' + str(a) + ',1)'
                for i in range(m):
                    if a == 1 and b == 1:
                        vec = (self.exp[i]) // (v[i])
                    elif a > 1 and b == 1:
                        vec = (self.exp[i] * onea) // (v[i])
                    elif a == 1 and b > 1:
                        vec = (self.exp[i]) // (v[i] * oneb)
                    else:
                        vec = (self.exp[i] * onea) // (v[i] * oneb)

                    Ptmp.add_constraint(exp < geomean(vec))
                Ptmp.add_constraint((1 | v) < exp)
            return NormP_Constraint(
                exp,
                self.exp,
                self.numerator,
                self.denominator,
                Ptmp,
                self.string +
                '>' +
                exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self > exp1


class TracePow_Exp(_ConvexExp):
    """A class storing the pth power of a scalar, or more generally the trace of the power of a symmetric matrix.
       It derives from :class:`Expression<picos.Expression>`.
       Use the function :func:`picos.tracepow() <picos.tools.tracepow>`
       or simply the overloaded ``**`` exponentiation operator
       to create an instance of this class.

       Note that this function is concave for :math:`0<p<1`, and convex for the other values of :math:`p`
       over the set of nonnegative variables ``exp``
       (resp. over the set of positive semidefinite matrices ``exp``), and PICOS implicitely forces
       the constraint ``exp >0`` (resp. ``exp >> 0``) to hold.

       Also, when a coef matrix :math:`M` is specified (for constraints of the
       form :math:`\operatorname{trace}(M X^p) \geq t`),
       the matrix :math:`M` must be positive semidefinite and :math:`p` must be in :math:`(0,1]`.

       **Overloaded operator**

            :``<``: less **or equal** than (the rhs must be a scalar affine expression, AND
                    p must be either greater or equal than 1 or negative)
            :``>``: greater **or equal** than (the rhs must be a scalar affine expression, AND
                    p must be in the range :math:`(0,1]`)


    """

    def __init__(self, exp, numerator, denominator=1, M=None):
        pstr = str(numerator)
        if denominator > 1:
            pstr += '/' + str(denominator)
        p = float(numerator) / float(denominator)
        if M is None:
            if exp.size == (1, 1):
                _ConvexExp.__init__(
                    self,
                    '( ' + exp.string + ')**' + pstr,
                    'pth power expression')
            else:
                _ConvexExp.__init__(self,
                                    'trace( ' + exp.string + ')**' + pstr,
                                    'trace of pth power expression')
        else:
            if exp.size == (1, 1):
                _ConvexExp.__init__(
                    self,
                    M.string + ' *( ' + exp.string + ')**' + pstr,
                    'pth power expression')
            else:
                _ConvexExp.__init__(
                    self,
                    'trace[ ' + M.string + ' *(' + exp.string + ')**' + pstr + ']',
                    'trace of pth power expression')

        if exp.size[0] != exp.size[1]:
            raise ValueError('Matrix must be square')
        self.exp = exp
        """The affine expression to which the p-norm is applied"""

        self.numerator = numerator
        """numerator of p"""

        self.denominator = denominator
        """denominator of p"""

        self.dim = exp.size[0]
        """dimension of ``exp``"""

        self.M = None
        """the coef matrix"""

        if M is not None:  # we assume without checking that M is positive semidefinite
            if p <= 0 or p > 1:
                raise ValueError(
                    'when a coef matrix M is given, p must be between 0 and 1')
            if not M.is_valued():
                raise ValueError('coef matrix M must be valued')
            if not M.size == exp.size:
                raise ValueError(
                    'coef matrix M must have the same size as exp')
            self.M = M

    def eval(self, ind=None):
        val = self.exp.eval(ind)

        if not isinstance(val, cvx.base.matrix):
            val = cvx.matrix(val)
        p = float(self.numerator) / float(self.denominator)
        if self.M is None:
            ev = np.linalg.eigvalsh(np.matrix(val))
            return sum([vi**p for vi in ev])
        else:
            Mval = self.M.eval(ind)
            U, S, V = np.linalg.svd(val)
            Xp = cvx.matrix(U) * cvx.spdiag([s**p for s in S]) * cvx.matrix(V)
            return np.trace(Mval * Xp)

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __lt__(self, exp):
        p = float(self.numerator) / self.denominator
        if (p < 1) and (p > 0):
            raise Exception(
                '<= operator can be used only when the function is convex (p>=1 or p<0)')
        if p == 1:
            return self.exp < exp
        elif p == 2:
            return (self.exp)**2 < exp
        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('upper bound of a tracepow must be scalar')
            from .problem import Problem
            Ptmp = Problem()
            a = self.numerator
            b = self.denominator
            if self.dim == 1:
                idt = new_param('1', 1)
                varcnt = 0
                v = []
            else:
                idt = new_param('I', cvx.spdiag([1.] * self.dim))
                varcnt = 1
                v = [
                    Ptmp.add_variable(
                        'v[0]', (self.dim, self.dim), 'symmetric')]

            if a > b:
                # x2n < tb x2n-a
                pown = int(2**(np.ceil(np.log(a) / np.log(2))))
                if self.dim == 1:
                    lis = [exp] * b + [self.exp] * (pown - a) + [idt] * (a - b)
                else:
                    lis = [v[0]] * b + [self.exp] * \
                        (pown - a) + [idt] * (a - b)
                while len(lis) > 2:
                    newlis = []
                    while lis:
                        v1 = lis.pop()
                        v2 = lis.pop()
                        if v1 is v2:
                            newlis.append(v2)
                        else:
                            if self.dim == 1:
                                v0 = Ptmp.add_variable(
                                    'v[' + str(varcnt) + ']', 1)
                                Ptmp.add_constraint(v0**2 < v1 * v2)
                            else:
                                v0 = Ptmp.add_variable(
                                    'v[' + str(varcnt) + ']', (self.dim, self.dim), 'symmetric')
                                Ptmp.add_constraint(
                                    ((v1 & v0) // (v0 & v2)) >> 0)

                            varcnt += 1
                            newlis.append(v0)
                            v.append(v0)
                    lis = newlis
                if self.dim == 1:
                    Ptmp.add_constraint(self.exp**2 < lis[0] * lis[1])
                else:
                    Ptmp.add_constraint(
                        ((lis[0] & self.exp) // (self.exp & lis[1])) >> 0)
                    Ptmp.add_constraint((idt | v[0]) < exp)
            else:  # p<0
                # 1 < tb xa
                a = abs(a)
                b = abs(b)
                pown = int(2**(np.ceil(np.log(a + b) / np.log(2))))
                if self.dim == 1:
                    lis = [exp] * b + [self.exp] * a + [idt] * (pown - a - b)
                else:
                    lis = [v[0]] * b + [self.exp] * a + [idt] * (pown - a - b)

                while len(lis) > 2:
                    newlis = []
                    while lis:
                        v1 = lis.pop()
                        v2 = lis.pop()
                        if v1 is v2:
                            newlis.append(v2)
                        else:
                            if self.dim == 1:
                                v0 = Ptmp.add_variable(
                                    'v[' + str(varcnt) + ']', 1)
                                Ptmp.add_constraint(v0**2 < v1 * v2)
                            else:
                                v0 = Ptmp.add_variable(
                                    'v[' + str(varcnt) + ']', (self.dim, self.dim), 'symmetric')
                                Ptmp.add_constraint(
                                    ((v1 & v0) // (v0 & v2)) >> 0)

                            varcnt += 1
                            newlis.append(v0)
                            v.append(v0)
                    lis = newlis
                if self.dim == 1:
                    Ptmp.add_constraint(1 < lis[0] * lis[1])
                else:
                    Ptmp.add_constraint(
                        ((lis[0] & idt) // (idt & lis[1])) >> 0)
                    Ptmp.add_constraint((idt | v[0]) < exp)

            return TracePow_Constraint(
                exp,
                self.exp,
                self.numerator,
                self.denominator,
                None,
                Ptmp,
                self.string +
                '<' +
                exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self < exp1

    def __gt__(self, exp):
        p = float(self.numerator) / self.denominator
        if (p > 1) or (p < 0):
            raise Exception(
                '>= operator can be used only when the function is concave (0<p<=1)')
        if p == 1:
            if self.M is None:
                return self.exp > exp
            else:
                return (self.M | self.exp) > exp

        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('lower bound of a tracepow must be scalar')
            from .problem import Problem
            Ptmp = Problem()
            a = self.numerator
            b = self.denominator
            if self.dim == 1:
                idt = new_param('1', 1)
            else:
                idt = new_param('I', cvx.spdiag([1.] * self.dim))

            # we must have 0<a<b
            # t2n < xa t2n-b

            pown = int(2**(np.ceil(np.log(b) / np.log(2))))
            if self.dim == 1:
                idt = new_param('1', 1)
                varcnt = 0
                v = []
                lis = [self.exp] * a + [exp] * (pown - b) + [idt] * (b - a)

            else:
                idt = new_param('I', cvx.spdiag([1.] * self.dim))
                varcnt = 1
                v = [
                    Ptmp.add_variable(
                        'v[0]', (self.dim, self.dim), 'symmetric')]
                lis = [self.exp] * a + [v[0]] * (pown - b) + [idt] * (b - a)

            while len(lis) > 2:
                newlis = []
                while lis:
                    v1 = lis.pop()
                    v2 = lis.pop()
                    if v1 is v2:
                        newlis.append(v2)
                    else:
                        if self.dim == 1:
                            v0 = Ptmp.add_variable('v[' + str(varcnt) + ']', 1)
                            Ptmp.add_constraint(v0**2 < v1 * v2)
                        else:
                            v0 = Ptmp.add_variable(
                                'v[' + str(varcnt) + ']', (self.dim, self.dim), 'symmetric')
                            Ptmp.add_constraint(((v1 & v0) // (v0 & v2)) >> 0)

                        varcnt += 1
                        newlis.append(v0)
                        v.append(v0)
                lis = newlis
            if self.dim == 1:
                if self.M is None:
                    Ptmp.add_constraint(exp**2 < lis[0] * lis[1])
                else:
                    Ptmp.add_constraint(v[0]**2 < lis[0] * lis[1])
                    Ptmp.add_constraint((self.M * v[0]) > exp)
            else:
                Ptmp.add_constraint(((lis[0] & v[0]) // (v[0] & lis[1])) >> 0)
                if self.M is None:
                    Ptmp.add_constraint((idt | v[0]) > exp)
                else:
                    Ptmp.add_constraint((self.M | v[0]) > exp)

            return TracePow_Constraint(
                exp,
                self.exp,
                self.numerator,
                self.denominator,
                self.M,
                Ptmp,
                self.string +
                '>' +
                exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self > exp1


class DetRootN_Exp(_ConvexExp):
    """A class storing the nth root of the determinant of a positive semidefinite matrix.
       It derives from :class:`Expression<picos.Expression>`.
       Use the function :func:`picos.detrootn() <picos.tools.detrootn>`
       to create an instance of this class.
       Note that the matrix :math:`X` is forced to be positive semidefinite
       when a constraint of the form ``t < pic.detrootn(X)`` is added.

       **Overloaded operator**

            :``>``: greater **or equal** than (the rhs must be a scalar affine expression)

    """

    def __init__(self, exp):
        if exp.size[0] != exp.size[1]:
            raise ValueError('Matrix must be square')
        nstr = str(exp.size[0])
        _ConvexExp.__init__(
            self,
            'det( ' + exp.string + ')**1/' + nstr,
            'detrootn expression')

        self.exp = exp
        """The affine expression to which the det-root-n is applied"""

        self.dim = exp.size[0]
        """dimension of ``exp``"""

    def eval(self, ind=None):
        val = self.exp.eval(ind)
        if not isinstance(val, cvx.base.matrix):
            val = cvx.matrix(val)
        return (np.linalg.det(np.matrix(val)))**(1. / self.dim)

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __gt__(self, exp):

        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception('lower bound of a detrootn must be scalar')
            from .problem import Problem
            Ptmp = Problem()
            nr = self.dim * (self.dim + 1) // 2
            l = Ptmp.add_variable('l', (nr, 1))
            L = ltrim1(l, uptri=0)
            dl = diag_vect(L)
            ddL = diag(dl)

            Ptmp.add_constraint((self.exp & L) // (L.T & ddL) >> 0)
            Ptmp.add_constraint(exp < geomean(dl))

            return DetRootN_Constraint(
                exp, self.exp, Ptmp, self.string + '>' + exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self > exp1


class Sum_k_Largest_Exp(_ConvexExp):
    """A class storing the sum of the k largest elements of an
       :class:`AffinExp <picos.AffinExp>`, or the sum
       of its k largest eigenvalues (for a square matrix expression).
       It derives from :class:`Expression<picos.Expression>`.

       Use the function :func:`picos.sum_k_largest() <picos.tools.sum_k_largest>`
       or :func:`picos.sum_k_largest_lambda() <picos.tools.sum_k_largest_lambda>`
       to create an instance of this class.

       Note that the matrix :math:`X` is assumed to be symmetric
       when a constraint of the form ``pic.sum_k_largest_lambda(X,k) < t`` is added.

       **Overloaded operator**

            :``<``: smaller **or equal** than (the rhs must be a scalar affine expression)

    """

    def __init__(self, exp, k, eigenvals=False):
        if eigenvals:
            n = exp.size[0]
            if k == 1:
                expstr = 'lambda_max(' + exp.string + ')'
            elif k == n:
                expstr = 'trace(' + exp.string + ')'
            else:
                expstr = 'sum_' + \
                    str(k) + '_largest_lambda(' + exp.string + ')'
            if exp.size[0] != exp.size[1]:
                raise ValueError('Expression must be square')
        else:
            n = exp.size[0] * exp.size[1]
            if k == 1:
                expstr = 'max(' + exp.string + ')'
            elif k == n:
                expstr = 'sum(' + exp.string + ')'
            else:
                expstr = 'sum_' + str(k) + '_largest(' + exp.string + ')'

        _ConvexExp.__init__(self, expstr, 'sum_k_largest expression')

        self.exp = exp
        """The affine expression to which the sum_k_largest is applied"""

        self.k = k
        """The number of elements to sum"""

        self.eigenvalues = eigenvals
        """whether this is a sum of k largest eigenvalues (for symmetric matrices)"""

    def eval(self, ind=None):
        val = self.exp.eval(ind)
        if not isinstance(val, cvx.base.matrix):
            val = cvx.matrix(val)

        if self.eigenvalues:
            ev = sorted(np.linalg.eigvalsh(val))
        else:
            ev = sorted(val)

        return sum(ev[-self.k:])

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __lt__(self, exp):

        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception(
                    'upper bound of a sum_k_largest must be scalar')
            from .problem import Problem
            Ptmp = Problem()
            if self.eigenvalues:
                n = self.exp.size[0]
                I = new_param('I', cvx.spdiag([1.] * n))
                if self.k == n:
                    return (I | self.exp) < exp
                elif self.k == 1:
                    cons = self.exp << exp * I
                    cons.myconstring = self.string + '<=' + exp.string
                    return cons
                else:
                    s = Ptmp.add_variable('s', 1)
                    Z = Ptmp.add_variable('Z', (n, n), 'symmetric')
                    Ptmp.add_constraint(Z >> 0)
                    Ptmp.add_constraint(self.exp << Z + s * I)
                    Ptmp.add_constraint(exp > (I | Z) + (self.k * s))
            else:
                n = self.exp.size[0] * self.exp.size[1]
                if self.k == 1:
                    cons = self.exp < exp
                    cons.myconstring = self.string + '<=' + exp.string
                    return cons
                elif self.k == n:
                    return (1 | self.exp) < exp
                else:
                    lbda = Ptmp.add_variable('lambda', 1)
                    mu = Ptmp.add_variable('mu', self.exp.size, lower=0)
                    Ptmp.add_constraint(self.exp < lbda + mu)
                    Ptmp.add_constraint(self.k * lbda + (1 | mu) < exp)

            return Sumklargest_Constraint(
                exp,
                self.exp,
                self.k,
                self.eigenvalues,
                True,
                Ptmp,
                self.string +
                '<' +
                exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self < exp1


class Sum_k_Smallest_Exp(_ConvexExp):
    """A class storing the sum of the k smallest elements of an
       :class:`AffinExp <picos.AffinExp>`, or the sum
       of its k smallest eigenvalues (for a square matrix expression).
       It derives from :class:`Expression<picos.Expression>`.

       Use the function :func:`picos.sum_k_smallest() <picos.tools.sum_k_smallest>`
       or :func:`picos.sum_k_smallest_lambda() <picos.tools.sum_k_smallest_lambda>`
       to create an instance of this class.

       Note that the matrix :math:`X` is assumed to be symmetric
       when a constraint of the form ``pic.sum_k_smallest_lambda(X,k) > t`` is added.

       **Overloaded operator**

            :``>``: greater **or equal** than (the rhs must be a scalar affine expression)

    """

    def __init__(self, exp, k, eigenvals=False):
        if eigenvals:
            n = exp.size[0]
            if k == 1:
                expstr = 'lambda_min(' + exp.string + ')'
            elif k == n:
                expstr = 'trace(' + exp.string + ')'
            else:
                expstr = 'sum_' + \
                    str(k) + '_smallest_lambda(' + exp.string + ')'
            if exp.size[0] != exp.size[1]:
                raise ValueError('Expression must be square')
        else:
            n = exp.size[0] * exp.size[1]
            if k == 1:
                expstr = 'min(' + exp.string + ')'
            elif k == n:
                expstr = 'sum(' + exp.string + ')'
            else:
                expstr = 'sum_' + str(k) + '_smallest(' + exp.string + ')'

        _ConvexExp.__init__(self, expstr, 'sum_k_smallest expression')

        self.exp = exp
        """The affine expression to which sum_k_smallest is applied"""

        self.k = k
        """The number of elements to sum"""

        self.eigenvalues = eigenvals
        """whether this is a sum of k smallest eigenvalues (for symmetric matrices)"""

    def eval(self, ind=None):
        val = self.exp.eval(ind)
        if not isinstance(val, cvx.base.matrix):
            val = cvx.matrix(val)

        if self.eigenvalues:
            ev = sorted(np.linalg.eigvalsh(val))
        else:
            ev = sorted(val)

        return sum(ev[:self.k])

    value = property(
        eval,
        Expression.set_value,
        Expression.del_simple_var_value)

    def __gt__(self, exp):

        if isinstance(exp, AffinExp):
            if exp.size != (1, 1):
                raise Exception(
                    'lower bound of a sum_k_smallest must be scalar')
            from .problem import Problem
            Ptmp = Problem()
            if self.eigenvalues:
                n = self.exp.size[0]
                I = new_param('I', cvx.spdiag([1.] * n))
                if self.k == n:
                    return (I | self.exp) < exp
                elif self.k == 1:
                    cons = self.exp >> exp * I
                    cons.myconstring = self.string + '>=' + exp.string
                    return cons
                else:
                    s = Ptmp.add_variable('s', 1)
                    Z = Ptmp.add_variable('Z', (n, n), 'symmetric')
                    Ptmp.add_constraint(Z >> 0)
                    Ptmp.add_constraint(-self.exp << Z + s * I)
                    Ptmp.add_constraint(-exp > (I | Z) + (self.k * s))
            else:
                n = self.exp.size[0] * self.exp.size[1]
                if self.k == 1:
                    cons = self.exp > exp
                    cons.myconstring = self.string + '>=' + exp.string
                    return cons
                elif self.k == n:
                    return (1 | self.exp) > exp
                else:
                    lbda = Ptmp.add_variable('lambda', 1)
                    mu = Ptmp.add_variable('mu', self.exp.size, lower=0)
                    Ptmp.add_constraint(-self.exp < lbda + mu)
                    Ptmp.add_constraint(self.k * lbda + (1 | mu) < -exp)

            return Sumklargest_Constraint(
                exp,
                self.exp,
                self.k,
                self.eigenvalues,
                False,
                Ptmp,
                self.string +
                '>' +
                exp.string)

        else:  # constant
            term, termString = _retrieve_matrix(exp, (1, 1))
            exp1 = AffinExp(
                factors={}, constant=term, size=(
                    1, 1), string=termString)
            return self > exp1


class Variable(AffinExp):
    """This class stores a variable. It
    derives from :class:`AffinExp<picos.AffinExp>`.
    """

    def __init__(self, parent_problem,
                 name,
                 size,
                 Id,
                 startIndex,
                 vtype='continuous',
                 lower=None,
                 upper=None):

        # attributes of the parent class (AffinExp)
        idmat = _svecm1_identity(vtype, size)
        AffinExp.__init__(self, factors={self: idmat},
                          constant=None,
                          size=size,
                          string=name
                          )

        self.name = name
        """The name of the variable (str)"""

        self.parent_problem = parent_problem
        """The Problem instance to which this variable belongs"""

        self.Id = Id
        """An integer index (obsolete)"""
        self._vtype = vtype

        self._startIndex = startIndex

        self._endIndex = None
        """end position in the global vector of all variables"""

        if vtype in ('symmetric',):
            self._endIndex = startIndex + \
                (size[0] * (size[0] + 1)) // 2  # end position +1
        else:
            self._endIndex = startIndex + size[0] * size[1]  # end position +1

        self._value = None

        self.value_alt = {}  # alternative values for solution pools

        # dictionary of (lower,upper) bounds ( +/-infinite if the index is not
        # in the dict)
        self._bnd = _NonWritableDict()

        self._semiDef = False  # True if this is a sym. variable X subject to X>>0

        self._bndtext = ''

        self.passed = []
        """list of solvers which are already aware of this variable"""

        if not(lower is None):
            self.set_lower(lower)

        if not(upper is None):
            self.set_upper(upper)

    def __str__(self):
        if self.is_valued():
            if self.size == (1, 1):
                return str(self.value[0])
            else:
                return str(self.value)
        else:
            return repr(self)

    def __repr__(self):
        return '# variable {0}:({1} x {2}),{3} #'.format(
            self.name, self.size[0], self.size[1], self.vtype)

    def __iadd__(self, term):
        raise NotImplementedError('variable must not be changed inplace. Try to cast the first term of the sum as an AffinExp, e.g. by adding 0 to it.')

    @property
    def bnd(self):
        """
        ``var.bnd[i]`` returns a tuple ``(lo,up)`` of lower and upper bounds for the
        ith element of the variable ``var``. None means +/- infinite.
        if ``var.bnd[i]`` is not defined, then ``var[i]`` is unbounded.
        """
        return self._bnd

    @property
    def startIndex(self):
        """starting position in the global vector of all variables"""
        return self._startIndex

    @property
    def endIndex(self):
        """end position in the global vector of all variables"""
        return self._endIndex

    @property
    def vtype(self):
        """one of the following strings:

             * 'continuous' (continuous variable)
             * 'binary'     (binary 0/1 variable)
             * 'integer'    (integer variable)
             * 'symmetric'  (symmetric matrix variable)
             * 'antisym'    (antisymmetric matrix variable)
             * 'complex'    (complex matrix variable)
             * 'hermitian'  (complex hermitian matrix variable)
             * 'semicont'   (semicontinuous variable [can take the value 0 or any other admissible value])
             * 'semiint'    (semi integer variable [can take the value 0 or any other integer admissible value])
        """
        return self._vtype

    @vtype.setter
    def vtype(self, value):
        if not(
            value in [
                'symmetric',
                'antisym',
                'hermitian',
                'complex',
                'continuous',
                'binary',
                'integer',
                'semicont',
                'semiint']):
            raise ValueError('unknown variable type')
        if self._vtype not in ('symmetric',) and value in ('symmetric',):
            raise Exception(
                'change to symmetric is forbiden because of sym-vectorization')
        if self._vtype in ('symmetric',) and value not in ('symmetric',):
            raise Exception(
                'change from symmetric is forbiden because of sym-vectorization')
        if self._vtype not in ('antisym',) and value in ('antisym',):
            raise Exception(
                'change to antisym is forbiden because of sym-vectorization')
        if self._vtype in ('antisym',) and value not in ('antisym',):
            raise Exception(
                'change from antisym is forbiden because of sym-vectorization')
        self._vtype = value
        if ('[' in self.name and
                ']' in self.name and
                self.name.split('[')[0] in self.parent_problem.listOfVars):
            vlist = self.name.split('[')[0]
            if all(
                    [vi.vtype == value for vi in self.parent_problem.get_variable(vlist)]):
                self.parent_problem.listOfVars[vlist]['vtype'] = value
            else:
                self.parent_problem.listOfVars[vlist]['vtype'] = 'different'

    @property
    def semiDef(self):
        """True if this is a sym. variable X subject to X>>0"""
        return self._semiDef

    @semiDef.setter
    def semiDef(self, value):
        if not(self._semiDef) and (value) and ('mosek' in self.passed):
            print("\033[1;31mWarning: this var has already been passed to mosek, so mosek will not be able to handle it as a bar variable.\033[0m")

        self._semiDef = value

    def set_lower(self, lo):
        """
        sets a lower bound to the variable
        (lo may be scalar or a matrix of the same size as the variable ``self``).
        Entries smaller than -INFINITY = -1e16 are ignored
        """
        lowexp = _retrieve_matrix(lo, self.size)[0]
        if self.vtype in ('symmetric',):
            lowexp = svec(lowexp)
        if self.vtype in ('hermitian', 'complex'):
            raise Exception('lower bound not supported for complex variables')
        for i in range(lowexp.size[0] * lowexp.size[1]):
            li = lowexp[i]
            if li > -INFINITY:
                bil, biu = self.bnd.get(i, (None, None))
                self.bnd._set(i, (li, biu))

        if ('low' not in self._bndtext) and (
                'nonnegative' not in self._bndtext):
            if lowexp:
                self._bndtext += ', bounded below'
            else:
                self._bndtext += ', nonnegative'
        elif ('low' in self._bndtext):
            if not(lowexp):
                self._bndtext.replace('bounded below', 'nonnegative')
            elif ('lower' in self._bndtext):
                self._bndtext.replace('some lower bounds', 'bounded below')
        else:
            if lowexp:
                self._bndtext.replace('nonnegative', 'bounded below')
        for solver in self.passed:
            texteval = 'self.parent_problem.reset_' + solver + '_instance()'
            eval(texteval)

    def set_sparse_lower(self, indices, bnds):
        """
        sets the lower bound bnds[i] to the index indices[i] of the variable.
        For a symmetric matrix variable, bounds on elements in the upper triangle are ignored.

        :param indices: list of indices, given as integers (column major order) or tuples (i,j).
        :type indices: ``list``
        :param bnds: list of lower bounds.
        :type lower: ``list``

        .. Warning:: This function does not modify the existing bounds on elements other
                     than those specified in ``indices``.

        **Example:**

        >>> import picos as pic
        >>> P = pic.Problem()
        >>> X = P.add_variable('X',(3,2),lower = 0)
        >>> X.set_sparse_upper([0,(0,1),1],[1,2,0])
        >>> X.bnd #doctest: +NORMALIZE_WHITESPACE
        {0: (0.0, 1.0),
         1: (0.0, 0.0),
         2: (0.0, None),
         3: (0.0, 2.0),
         4: (0.0, None),
         5: (0.0, None)}
        """
        if self.vtype in ('hermitian', 'complex'):
            raise Exception('lower bound not supported for complex variables')
        s0 = self.size[0]
        vv = []
        ii = []
        jj = []
        for idx, lo in zip(indices, bnds):
            if isinstance(idx, int):
                idx = (idx % s0, idx // s0)
            if self.vtype in ('symmetric',):
                (i, j) = idx
                if i > j:
                    ii.append(i)
                    jj.append(j)
                    vv.append(lo)
                    ii.append(j)
                    jj.append(i)
                    vv.append(lo)
                elif i == j:
                    ii.append(i)
                    jj.append(i)
                    vv.append(lo)
            ii.append(idx[0])
            jj.append(idx[1])
            vv.append(lo)
        spLO = spmatrix(vv, ii, jj, self.size)
        if self.vtype in ('symmetric',):
            spLO = svec(spLO)
        for i, j, v in zip(spLO.I, spLO.J, spLO.V):
            ii = s0 * j + i
            bil, biu = self.bnd.get(ii, (None, None))
            self.bnd._set(ii, (v, biu))

        if ('nonnegative' in self._bndtext):
            self._bndtext.replace('nonnegative', 'bounded below')
        elif ('low' not in self._bndtext):
            self._bndtext += ', some lower bounds'

        for solver in self.passed:
            texteval = 'self.parent_problem.reset_' + solver + '_instance()'
            eval(texteval)

    def set_upper(self, up):
        """
        sets an upper bound to the variable
        (up may be scalar or a matrix of the same size as the variable ``self``).
        Entries larger than INFINITY = 1e16 are ignored
        """
        upexp = _retrieve_matrix(up, self.size)[0]
        if self.vtype in ('symmetric',):
            upexp = svec(upexp)
        if self.vtype in ('hermitian', 'complex'):
            raise Exception('lower bound not supported for complex variables')
        for i in range(upexp.size[0] * upexp.size[1]):
            ui = upexp[i]
            if ui < INFINITY:
                bil, biu = self.bnd.get(i, (None, None))
                self.bnd._set(i, (bil, ui))
        if ('above' not in self._bndtext) and ('upper' not in self._bndtext) and (
                'nonpositive' not in self._bndtext):
            if upexp:
                self._bndtext += ', bounded above'
            else:
                self._bndtext += ', nonpositive'
        elif ('above' in self._bndtext):
            if not(upexp):
                self._bndtext.replace('bounded above', 'nonpositive')
        elif ('upper' in self._bndtext):
            self._bndtext.replace('some upper bounds', 'bounded above')
        else:
            if upexp:
                self._bndtext.replace('nonpositive', 'bounded above')

        for solver in self.passed:
            texteval = 'self.parent_problem.reset_' + solver + '_instance()'
            eval(texteval)

    def set_sparse_upper(self, indices, bnds):
        """
        sets the upper bound bnds[i] to the index indices[i] of the variable.
        For a symmetric matrix variable, bounds on elements in the upper triangle are ignored.

        :param indices: list of indices, given as integers (column major order) or tuples (i,j).
        :type indices: ``list``
        :param bnds: list of upper bounds.
        :type lower: ``list``

        .. Warning:: This function does not modify the existing bounds on elements other
                     than those specified in ``indices``.
        """
        if self.vtype in ('hermitian', 'complex'):
            raise Exception('lower bound not supported for complex variables')
        s0 = self.size[0]
        vv = []
        ii = []
        jj = []
        for idx, up in zip(indices, bnds):
            if isinstance(idx, int):
                idx = (idx % s0, idx // s0)
            if self.vtype in ('symmetric',):
                (i, j) = idx
                if i > j:
                    ii.append(i)
                    jj.append(j)
                    vv.append(up)
                    ii.append(j)
                    jj.append(i)
                    vv.append(up)
                elif i == j:
                    ii.append(i)
                    jj.append(i)
                    vv.append(up)
            else:
                ii.append(idx[0])
                jj.append(idx[1])
                vv.append(up)
        spUP = spmatrix(vv, ii, jj, self.size)
        if self.vtype in ('symmetric',):
            spUP = svec(spUP)
        for i, j, v in zip(spUP.I, spUP.J, spUP.V):
            ii = s0 * j + i
            bil, biu = self.bnd.get(ii, (None, None))
            self.bnd._set(ii, (bil, v))
        if ('nonpositive' in self._bndtext):
            self._bndtext.replace('nonpositive', 'bounded above')
        elif ('above' not in self._bndtext) and ('upper' not in self._bndtext):
            self._bndtext += ', some upper bounds'

        for solver in self.passed:
            texteval = 'self.parent_problem.reset_' + solver + '_instance()'
            eval(texteval)

    def eval(self, ind=None):
        if ind is None:
            if self._value is None:
                raise Exception(self.name + ' is not valued')
            else:
                if self.vtype in ('symmetric',):
                    return cvx.matrix(svecm1(self._value))
                else:
                    return cvx.matrix(self._value)
        else:
            if ind in self.value_alt:
                if self.vtype in ('symmetric',):
                    return cvx.matrix(svecm1(self.value_alt[ind]))
                else:
                    return cvx.matrix(self.value_alt[ind])
            else:
                raise Exception(
                    self.name +
                    ' does not have a value for the index ' +
                    str(ind))

    def set_value(self, value):
        valuemat, valueString = _retrieve_matrix(value, self.size)
        if valuemat.size != self.size:
            raise Exception('should be of size {0}'.format(self.size))
        if self.vtype in ('symmetric',):
            valuemat = svec(valuemat)
        elif self.vtype == 'hermitian':
            v = (valuemat - valuemat.H)[:]
            norm2 = (v.H * v)[0].real
            if norm2 > 1e-6:
                raise ValueError('value is not Hermitian')
            else:
                valuemat = 0.5 * (valuemat + valuemat.H)
        self._value = valuemat

    def del_var_value(self):
        self._value = None

    value = property(eval, set_value, del_var_value, "value of the variable")
    """value of the variable. The value of a variable is
                defined in the following two situations:

                * The user has assigned a value to the variable,
                  by using either the present ``value`` attribute,
                  or the function
                  :func:`set_var_value()<picos.Problem.set_var_value>` of the class
                  :class:`Problem<picos.Problem>`. Note that manually
                  giving a value to a variable can be useful, e.g. to
                  provide a solver with an initial solution (see
                  the option ``hotstart`` documented in
                  :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`)
                * The problem involving the variable has been solved,
                  and the ``value`` attribute stores the optimal value
                  of this variable.
        """

    def __getitem__(self, index):
        """faster implementation of getitem for variable"""
        if self.vtype in ('symmetric',):
            return AffinExp.__getitem__(self, index)

        def indexstr(idx):
            if isinstance(idx, int):
                return str(idx)
            elif isinstance(idx, Expression):
                return idx.string

        def slicestr(sli):
            # single element
            if not (sli.start is None or sli.stop is None):
                sta = sli.start
                sto = sli.stop
                if isinstance(sta, int):
                    sta = new_param(str(sta), sta)
                if isinstance(sto, int):
                    sto = new_param(str(sto), sto)
                if (sto.__index__() == sta.__index__() + 1):
                    return sta.string
            # single element -1 (Expression)
            if (isinstance(sli.start, Expression) and sli.start.__index__()
                    == -1 and sli.stop is None and sli.step is None):
                return sli.start.string
            # single element -1
            if (isinstance(sli.start, int) and sli.start == -1
                    and sli.stop is None and sli.step is None):
                return '-1'
            ss = ''
            if not sli.start is None:
                ss += indexstr(sli.start)
            ss += ':'
            if not sli.stop is None:
                ss += indexstr(sli.stop)
            if not sli.step is None:
                ss += ':'
                ss += indexstr(sli.step)
            return ss

        if isinstance(index, Expression) or isinstance(index, int):
            ind = index.__index__()
            if ind < 0:
                rangeT = [self.size[0] * self.size[1] + ind]
            else:
                rangeT = [ind]
            newsize = (1, 1)
            indstr = indexstr(index)
        elif isinstance(index, slice):
            idx = index.indices(self.size[0] * self.size[1])
            rangeT = range(idx[0], idx[1], idx[2])
            newsize = (len(rangeT), 1)
            indstr = slicestr(index)
        elif isinstance(index, tuple):
            # simple common cases for fast implementation
            if isinstance(
                    index[0],
                    int) and isinstance(
                    index[1],
                    int):  # element
                ind0 = index[0]
                ind1 = index[1]
                if ind0 < 0:
                    ind0 = self.size[0] + ind0
                if ind1 < 0:
                    ind1 = self.size[1] + ind1
                rangeT = [ind1 * self.size[0] + ind0]
                newsize = (1, 1)
                indstr = indexstr(index[0]) + ',' + indexstr(index[1])
            elif isinstance(index[0], int) and index[1] == slice(None, None, None):  # row
                ind0 = index[0]
                if ind0 < 0:
                    ind0 = self.size[0] + ind0
                rangeT = range(ind0, self.size[0] * self.size[1], self.size[0])
                newsize = (1, self.size[1])
                indstr = indexstr(index[0]) + ',:'

            elif isinstance(index[1], int) and index[0] == slice(None, None, None):  # column
                ind1 = index[1]
                if ind1 < 0:
                    ind1 = self.size[1] + ind1
                rangeT = range(ind1 * self.size[0], (ind1 + 1) * self.size[0])
                newsize = (self.size[0], 1)
                indstr = ':,' + indexstr(index[1])
            else:
                if isinstance(
                        index[0],
                        Expression) or isinstance(
                        index[0],
                        int):
                    ind = index[0].__index__()
                    if ind == -1:
                        index = (slice(ind, None, None), index[1])
                    else:
                        index = (slice(ind, ind + 1, None), index[1])
                if isinstance(
                        index[1],
                        Expression) or isinstance(
                        index[1],
                        int):
                    ind = index[1].__index__()
                    if ind == -1:
                        index = (index[0], slice(ind, None, None))
                    else:
                        index = (index[0], slice(ind, ind + 1, None))
                idx0 = index[0].indices(self.size[0])
                idx1 = index[1].indices(self.size[1])
                rangei = range(idx0[0], idx0[1], idx0[2])
                rangej = range(idx1[0], idx1[1], idx1[2])
                rangeT = []
                for j in rangej:
                    rangei_translated = []
                    for vi in rangei:
                        rangei_translated.append(
                            vi + (j * self.size[0]))
                    rangeT.extend(rangei_translated)

                newsize = (len(rangei), len(rangej))
                indstr = slicestr(index[0]) + ',' + slicestr(index[1])

        sz = self.size[0] * self.size[1]
        nsz = len(rangeT)
        newfacs = {}
        II = range(nsz)
        JJ = rangeT
        VV = [1.] * nsz
        newfacs = {self: spmatrix(VV, II, JJ, (nsz, sz))}
        if not self.constant is None:
            newcons = self.constant[rangeT]
        else:
            newcons = None
        newstr = self.string + '[' + indstr + ']'
        # check size
        if newsize[0] == 0 or newsize[1] == 0:
            raise IndexError('slice of zero-dimension')
        return AffinExp(newfacs, newcons, newsize, newstr)


class Set(object):
    """
    Parent class for set objects
    """

    def __init__(self):
        pass


class Ball(Set):
    """
    represents a Ball of Norm p. This object should be created by the function :func:`picos.ball() <picos.tools.ball>` .

    ** Overloaded operators **

      :``>>``: membership of the right hand side in this set.

    """

    def __init__(self, p, radius):
        self.p = p
        self.radius = radius

    def __str__(self):
        if float(self.p) >= 1:
            return '# L_' + str(self.p) + '-Ball of radius ' + \
                str(self.radius) + ' #'
        else:
            return '# generalized outer L_p-Ball {x>=0: ||x||_' + \
                str(self.p) + ' >=' + str(self.radius) + '} #'

    def __repr__(self):
        return str(self)

    def __rshift__(self, exp):
        if isinstance(exp, AffinExp):
            if float(self.p) >= 1:
                cons = (norm(exp, self.p) < self.radius)
            else:
                cons = (norm(exp, self.p) > self.radius)
            return cons
        else:  # constant
            term, termString = _retrieve_matrix(exp, None)
            exp2 = AffinExp(
                factors={},
                constant=term[:],
                size=term.size,
                string=termString)
            return self >> exp2


class Truncated_Simplex(Set):
    """
    represents a simplex, that can be intersected with the ball of radius 1 for the infinity-norm (truncation), and that can be symmetrized with respect to the origin.
    This object should be created by the function :func:`picos.simplex() <picos.tools.simplex>` or  :func:`picos.truncated_simplex() <picos.tools.truncated_simplex>` .

    ** Overloaded operators **

      :``>>``: membership of the right hand side in this set.

    """

    def __init__(self, radius=1, truncated=False, nonneg=True):
        self.radius = radius
        self.truncated = truncated
        self.nonneg = nonneg

    def __str__(self):
        if float(self.radius) == 1 and self.nonneg:
            return '# standard simplex #'
        if self.truncated:
            if self.nonneg:
                return '# truncated simplex {0<=x<=1: sum(x) <= ' + str(
                    self.radius) + '} #'
            else:
                return '# Symmetrized truncated simplex {-1<=x<=1: sum(|x|) <= ' + str(
                    self.radius) + '} #'
        else:
            if self.nonneg:
                return '# simplex {x>=0: sum(x) <= ' + str(self.radius) + '} #'
            else:
                return '# L_1-Ball of radius ' + str(self.radius) + '} #'

    def __repr__(self):
        return str(self)

    def __rshift__(self, exp):

        if isinstance(exp, AffinExp):
            n = exp.size[0] * exp.size[1]
            if self.truncated:
                if self.nonneg:
                    if self.radius <= 1:
                        aff = (-exp[:]) // (1 | exp[:])
                        rhs = cvx.sparse([0] * n + [self.radius])
                        if self.radius == 1:
                            simptext = ' in standard simplex'
                        else:
                            simptext = ' in simplex of radius ' + \
                                str(self.radius)
                    else:
                        aff = (exp[:]) // (-exp[:]) // (1 | exp[:])
                        rhs = cvx.sparse([1] * n + [0] * n + [self.radius])
                        simptext = ' in truncated simplex of radius ' + \
                            str(self.radius)
                    cons = (aff <= rhs)
                    cons.myconstring = exp.string + simptext
                else:
                    from .problem import Problem
                    Ptmp = Problem()
                    v = Ptmp.add_variable('v', n)
                    Ptmp.add_constraint(exp[:] < v)
                    Ptmp.add_constraint(-exp[:] < v)
                    Ptmp.add_constraint((1 | v) < self.radius)
                    if self.radius > 1:
                        Ptmp.add_constraint(v < 1)
                    constring = '||' + exp.string + \
                        '||_{infty;1} <= {1;' + str(self.radius) + '}'
                    cons = Sym_Trunc_Simplex_Constraint(
                        exp, self.radius, Ptmp, constring)
            else:
                if self.nonneg:
                    aff = (-exp[:]) // (1 | exp[:])
                    rhs = cvx.sparse([0] * n + [self.radius])
                    cons = (aff <= rhs)
                    if self.radius == 1:
                        cons.myconstring = exp.string + ' in standard simplex'
                    else:
                        cons.myconstring = exp.string + \
                            ' in simplex of radius ' + str(self.radius)
                else:
                    cons = norm(exp, 1) < self.radius
            return cons
        else:  # constant
            term, termString = _retrieve_matrix(exp, None)
            exp2 = AffinExp(
                factors={},
                constant=term[:],
                size=term.size,
                string=termString)
            return self >> exp2
