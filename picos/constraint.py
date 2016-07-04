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

from __future__ import division

import cvxopt as cvx
import numpy as np
import sys

from .tools import *

__all__ = [
    'Constraint',
    '_Convex_Constraint',
    'Flow_Constraint',
    'GeoMeanConstraint',
    'NormP_Constraint',
    'TracePow_Constraint',
    'DetRootN_Constraint',
    'Sym_Trunc_Simplex_Constraint',
    'NormPQ_Constraint',
    'Sumklargest_Constraint']


class Constraint(object):
    """A class for describing a constraint.
    """

    def __init__(
            self,
            typeOfConstraint,
            Id,
            Exp1,
            Exp2,
            Exp3=None,
            dualVariable=None,
            key=None,
            ):
        from .expression import AffinExp
        self.typeOfConstraint = typeOfConstraint
        u"""A string from the following values,
                   indicating the type of constraint:

                        * ``lin<``, ``lin=`` or ``lin>`` : Linear (in)equality
                          ``Exp1 ≤ Exp2``, ``Exp1 = Exp2`` or ``Exp1 ≥ Exp2``.
                        * ``SOcone`` : Second Order Cone constraint ``||Exp1|| < Exp2``.
                        * ``RScone`` : Rotated Cone constraint
                          ``||Exp1||**2 < Exp2 * Exp3``.
                        * ``lse`` : Geometric Programming constraint ``LogSumExp(Exp1)<0``
                        * ``quad``: scalar quadratic constraint ``Exp1 < 0``.
                        * ``sdp<`` or ``sdp>``: semidefinite constraint
                          ``Exp1 ≼ Exp2`` or ``Exp1 ≽ Exp2``.
                """

        self.Exp1 = Exp1
        """LHS"""
        self.Exp2 = Exp2
        """RHS
                   (ignored for constraints of type ``lse`` and ``quad``, where
                   ``Exp2`` is set to ``0``)
                """
        self.Exp3 = Exp3
        """Second factor of the RHS for ``RScone`` constraints
                   (see :attr:`typeOfConstraint<picos.Constraint.typeOfConstraint>`).
                """
        self.Id = Id
        """An integer identifier"""
        self.dualVariable = dualVariable
        self.semidefVar = None
        """for a constraint of the form X>>0, stores the semidef variable"""
        self.exp1ConeVar = None
        self.exp2ConeVar = None
        self.exp3ConeVar = None
        """for a constraint of the form ||x||<u or ||x||^2<u v, stores x, u and v"""
        self.boundCons = None
        """stores  list of bounds of the form (var,index,lower,upper)"""
        self.key = None
        """A string to give a key name to the constraint"""
        self.myconstring = None  # workaround to redefine string representation
        # workaround to redefine complete constraint (with # ... #) string
        self.myfullconstring = None

        self.passed = []
        """list of solvers to which this constraints was already passed"""

        if typeOfConstraint == 'RScone' and Exp3 is None:
            raise NameError('I need a 3d expression')
        if typeOfConstraint[:3] == 'lin':
            if Exp1.size != Exp2.size:
                raise NameError('incoherent lhs and rhs')
            # are there some bound constrainta ?

        if typeOfConstraint[2:] == 'cone':
            if Exp2.size != (1, 1):
                raise NameError('expression on the rhs should be scalar')
            if not Exp3 is None:
                if Exp3.size != (1, 1):
                    raise NameError(
                        'expression on the rhs should be scalar')

        if typeOfConstraint == 'lse':
            if not (Exp2 == 0 or Exp2.is0()):
                raise NameError('lhs must be 0')
            self.Exp2 = AffinExp(
                factors={}, constant=cvx.matrix(
                    0, (1, 1)), string='0', size=(
                    1, 1))
        if typeOfConstraint == 'quad':
            if not (Exp2 == 0 or Exp2.is0()):
                raise NameError('lhs must be 0')
            self.Exp2 = AffinExp(
                factors={}, constant=cvx.matrix(
                    0, (1, 1)), string='0', size=(
                    1, 1))
        if typeOfConstraint[:3] == 'sdp':
            if Exp1.size != Exp2.size:
                raise NameError('incoherent lhs and rhs')
            if Exp1.size[0] != Exp1.size[1]:
                raise NameError('lhs and rhs should be square')
            # is it a simple constraint of the form X>>0 ?
            fac1 = self.Exp1.factors
            if len(fac1) == 1:
                var = list(fac1.keys())[0]
                mat = fac1[var]
                idty = _svecm1_identity(var.vtype, var.size)
                if (not(self.Exp1.constant) and
                    self.Exp2.is0() and
                    self.typeOfConstraint[3] == '>' and
                    var.vtype in ('symmetric', 'hermitian', 'continuous') and
                    list(mat.I) == list(idty.I) and
                    list(mat.J) == list(idty.J) and
                    list(mat.V) == list(idty.V)
                    ):
                    if var.vtype == 'continuous':
                        raise Exception(
                            "X>>0 with X of vtype continuous. Use vtype='symmetric' instead")
                    else:
                        self.semidefVar = var
            if self.Exp1.is_pure_complex_var():
                if self.Exp2.is0():
                    raise Exception(
                        "X>>0 with X of vtype complex. Use vtype='hermitian' instead")
            fac2 = self.Exp2.factors
            if len(fac2) == 1:
                var = list(fac2.keys())[0]
                mat = fac2[var]
                idty = _svecm1_identity(var.vtype, var.size)
                if (not(self.Exp2.constant) and
                    self.Exp1.is0() and
                    self.typeOfConstraint[3] == '<' and
                    var.vtype in ('symmetric', 'hermitian', 'continuous') and
                    list(mat.I) == list(idty.I) and
                    list(mat.J) == list(idty.J) and
                    list(mat.V) == list(idty.V)
                    ):
                    if var.vtype == 'continuous':
                        raise Exception(
                            "X>>0 with X of vtype continuous. Use vtype='symmetric' instead")
                    else:
                        self.semidefVar = var
            if self.Exp2.is_pure_complex_var():
                if self.Exp1.is0():
                    raise Exception(
                        "X>>0 with X of vtype complex. Use vtype='hermitian' instead")

    def __str__(self):
        if not(self.myfullconstring is None):
            return self.myfullconstring
        if self.typeOfConstraint[:3] == 'lin':
            constr = '# ({0}x{1})-affine constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'SOcone':
            constr = '# ({0}x{1})-SOC constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'RScone':
            constr = '# ({0}x{1})-Rotated SOC constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'lse':
            constr = '# ({0}x{1})-Geometric Programming constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'quad':
            constr = '#Quadratic constraint '
        if self.typeOfConstraint[:3] == 'sdp':
            constr = '# ({0}x{1})-LMI constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if not self.key is None:
            constr += '(' + self.key + ')'
        constr += ': '
        return constr + self.constring() + ' #'

    def __repr__(self):
        if self.typeOfConstraint[:3] == 'lin':
            constr = '# ({0}x{1})-affine constraint: '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'SOcone':
            constr = '# ({0}x{1})-SOC constraint: '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'RScone':
            constr = '# ({0}x{1})-Rotated SOC constraint: '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'lse':
            constr = '# ({0}x{1})-Geometric Programming constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        if self.typeOfConstraint == 'quad':
            constr = '#Quadratic constraint '
        if self.typeOfConstraint[:3] == 'sdp':
            constr = '# ({0}x{1})-LMI constraint '.format(
                self.Exp1.size[0], self.Exp1.size[1])
        return constr + self.constring() + ' #'

    def delete(self):
        """
        deletes the constraint from Problem
        """
        if self.Exp1.factors:
            prb = self.Exp1.factors.keys()[0].parent_problem
        elif self.Exp2.factors:
            prb = self.Exp2.factors.keys()[0].parent_problem
        elif self.Exp3 is not None and self.Exp3.factors:
            prb = self.Exp3.factors.keys()[0].parent_problem
        else:
            return

        cind = prb.constraints.index(self)
        prb.remove_constraint(cind)

    def constring(self):
        if not(self.myconstring is None):
            return self.myconstring
        if self.typeOfConstraint[:3] == 'lin':
            sense = ' ' + self.typeOfConstraint[-1] + ' '
            # if self.Exp2.is0():
            #        return self.Exp1.affstring()+sense+'0'
            # else:
            return self.Exp1.affstring() + sense + self.Exp2.affstring()
        if self.typeOfConstraint == 'SOcone':
            if self.Exp2.is1():
                return '||' + self.Exp1.affstring() + '|| < 1'
            else:
                return '||' + self.Exp1.affstring() + \
                    '|| < ' + self.Exp2.affstring()
        if self.typeOfConstraint == 'RScone':
            # if self.Exp1.size==(1,1):
            #       if self.Exp1.isconstant():
            #               retstr=self.Exp1.affstring() # warning: no square to simplfy
            #       else:
            #               retstr='('+self.Exp1.affstring()+')**2'
            if self.Exp1.size == (1, 1) and self.Exp1.isconstant():
                retstr = self.Exp1.affstring()
                if retstr[-5:] == '**0.5':
                    retstr = retstr[:-5]
                else:
                    retstr += '**2'
            else:
                retstr = '||' + self.Exp1.affstring() + '||^2'
            if (self.Exp2.is1() and self.Exp3.is1()):
                return retstr + ' < 1'
            elif self.Exp2.is1():
                return retstr + ' < ' + self.Exp3.affstring()
            elif self.Exp3.is1():
                return retstr + ' < ' + self.Exp2.affstring()
            else:
                return retstr + ' < ( ' + \
                    self.Exp2.affstring() + ')( ' + self.Exp3.affstring() + ')'
        if self.typeOfConstraint == 'lse':
            return 'LSE[ ' + self.Exp1.affstring() + ' ] < 0'
        if self.typeOfConstraint == 'quad':
            return self.Exp1.string + ' < 0'
        if self.typeOfConstraint[:3] == 'sdp':
            #sense=' '+self.typeOfConstraint[-1]+' '
            if self.typeOfConstraint[-1] == '<':
                sense = ' ≼ '
            else:
                sense = ' ≽ '
            # if self.Exp2.is0():
            #        return self.Exp1.affstring()+sense+'0'
            # elif self.Exp1.is0():
            #        return '0'+sense+self.Exp2.affstring()
            # else:
            return self.Exp1.affstring() + sense + self.Exp2.affstring()

    def keyconstring(self, lgstkey=None):
        constr = ''
        if not self.key is None:
            constr += '(' + self.key + ')'
        if lgstkey is None:
            constr += ':\t'
        else:
            if self.key is None:
                lcur = 0
            else:
                lcur = len(self.key) + 2
            if lgstkey == 0:
                ntabs = 0
            else:
                ntabs = int(np.ceil((2 + lgstkey) / 8.0))
            missingtabs = int(np.ceil(((ntabs * 8) - lcur) / 8.0))
            for i in range(missingtabs):
                constr += '\t'
            if lcur > 0:
                constr += ': '
            else:
                constr += '  '
            constr += self.constring()
        return constr

    def set_dualVar(self, value):
        self.dualVariable = value

    def dual_var(self):
        return self.dualVariable

    def del_dual(self):
        self.dualVariable = None

    dual = property(dual_var, set_dualVar, del_dual)
    """
        Value of the dual variable associated to this constraint

        See the :ref:`note on dual variables <noteduals>` in the tutorial
        for more information.
        """

    def slack_var(self):
        if self.typeOfConstraint == 'lse':
            from .tools import lse
            return -lse(self.Exp1).eval()
        elif self.typeOfConstraint[3] == '<':
            return self.Exp2.eval() - self.Exp1.eval()
        elif self.typeOfConstraint[3] == '>':
            return self.Exp1.eval() - self.Exp2.eval()
        elif self.typeOfConstraint == 'lin=':
            return self.Exp1.eval() - self.Exp2.eval()
        elif self.typeOfConstraint == 'SOcone':
            return self.Exp2.eval() - (abs(self.Exp1)).eval()
        elif self.typeOfConstraint == 'RScone':
            return self.Exp2.eval()[0] * self.Exp3.eval()[0] - \
                (abs(self.Exp1)**2).eval()
        elif self.typeOfConstraint == 'quad':
            return -(self.Exp1.eval())

    def set_slack(self, value):
        raise ValueError('slack is not writable')

    def del_slack(self):
        raise ValueError('slack is not writable')

    slack = property(slack_var, set_slack, del_slack)
    """Value of the slack variable associated to this constraint
           (should be nonnegative/zero if the inequality/equality
           is satisfied: for an inequality of the type ``lhs<rhs``,
           the slack is ``rhs-lhs``, and for ``lhs>rhs``
           the slack is ``lhs-rhs``)
           """


class _Convex_Constraint(Constraint):
    """A parent class for all (nonstandard) convex constraints handled by PICOS"""

    def __init__(self, Ptmp, constring, constypestr):
        self.Ptmp = Ptmp
        self.myconstring = constring
        self.constypestr = constypestr

    def __str__(self):
        return '# ' + self.constypestr + ' : ' + self.constring() + '#'

    def __repr__(self):
        return '# ' + self.constypestr + ' : ' + self.constring() + '#'

    def delete(self):
        parent_problem = self.find_parent_problem()
        if parent_problem is None:
            return
        for cons in self.Ptmp.constraints:
            cind = parent_problem.constraints.index(cons)
            parent_problem.remove_constraint(cind)

    def find_parent_problem(self):
        dummy_prefixes = ('_geo',
                          '_nop',
                          '_ntp',
                          '_ndt',
                          '_nts',
                          '_npq',
                          '_nsk')
        for cons in self.Ptmp.constraints:
            for v in cons.Exp1.factors:
                if v.name[:4] not in dummy_prefixes:
                    return v.parent_problem
            for v in cons.Exp2.factors:
                if v.name[:4] not in dummy_prefixes:
                    return v.parent_problem
            if cons.Exp3 is not None:
                for v in cons.Exp3.factors:
                    if v.name[:4] not in dummy_prefixes:
                        return v.parent_problem
        return None

class Flow_Constraint(_Convex_Constraint):
    """ A temporary object used to pass a flow constraint.
    This class derives from :class:`Constraint <picos.Constraint>`.
    """

    def __init__(self, G, Ptmp, constring):
        self.graph = G
        _Convex_Constraint.__init__(self, Ptmp, constring, 'flow constraint')
        self.prefix = '_flow'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def draw(self):
        from .tools import drawGraph
        drawGraph(self.graph)


class GeoMeanConstraint(_Convex_Constraint):
    """ A temporary object used to pass geometric mean inequalities.
    This class derives from :class:`Constraint <picos.Constraint>`.
    """

    def __init__(self, expaff, expgeo, Ptmp, constring):
        self.expaff = expaff
        self.expgeo = expgeo
        _Convex_Constraint.__init__(
            self, Ptmp, constring, 'geometric mean ineq')
        self.prefix = '_geo'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        return geomean(self.expgeo).value - self.expaff.value

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)


class NormP_Constraint(_Convex_Constraint):
    """ A temporary object used to pass p-norm inequalities.
    This class derives from :class:`Constraint <picos.Constraint>`
    """

    def __init__(self, expaff, expnorm, alpha, beta, Ptmp, constring):
        self.expaff = expaff
        self.expnorm = expnorm
        self.numerator = alpha
        self.denominator = beta
        p = float(alpha) / float(beta)
        if p > 1 or (p == 1 and '<' in constring):
            _Convex_Constraint.__init__(self, Ptmp, constring, 'p-norm ineq')
        else:
            _Convex_Constraint.__init__(
                self, Ptmp, constring, 'generalized p-norm ineq')
        self.prefix = '_nop'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        p = float(self.numerator) / self.denominator
        if p > 1 or (p == 1 and '<' in self.myconstring):
            return self.expaff.value - \
                norm(self.expnorm, self.numerator, self.denominator).value
        else:
            return -(self.expaff.value - norm(self.expnorm,
                                              self.numerator, self.denominator).value)

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)


class NormPQ_Constraint(_Convex_Constraint):
    """ A temporary object used to pass (p,q)-norm inequalities.
    This class derives from :class:`Constraint <picos.Constraint>`
    """

    def __init__(self, expaff, expnorm, p, q, Ptmp, constring):
        self.expaff = expaff
        self.expnorm = expnorm
        self.p = p
        self.q = q
        _Convex_Constraint.__init__(self, Ptmp, constring, 'pq-norm ineq')
        self.prefix = '_npq'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        return self.expaff.value - norm(self.expnorm, (p, q)).value

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)


class TracePow_Constraint(_Convex_Constraint):
    """ A temporary object used to pass (trace of) pth power inequalities
    This class derives from :class:`Constraint <picos.Constraint>`
    """

    def __init__(self, exprhs, explhs, alpha, beta, M, Ptmp, constring):
        self.explhs = explhs
        self.exprhs = exprhs
        self.numerator = alpha
        self.denominator = beta
        self.M = M
        p = float(alpha) / float(beta)
        if explhs.size[0] > 1:
            _Convex_Constraint.__init__(
                self, Ptmp, constring, 'trace of pth power ineq')
        else:
            _Convex_Constraint.__init__(
                self, Ptmp, constring, 'pth power ineq')
        self.prefix = '_ntp'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        p = float(self.numerator) / self.denominator
        slk = self.exprhs.value - \
            tracepow(self.explhs, self.numerator, self.denominator, self.M).value
        if p > 0 and p < 1:
            return -slk
        else:
            return slk

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)


class DetRootN_Constraint(_Convex_Constraint):
    """ A temporary object used to pass nth root of determinant inequalities.
    This class derives from :class:`Constraint <picos.Constraint>`
    """

    def __init__(self, exprhs, expdet, Ptmp, constring):
        self.expdet = expdet
        self.exprhs = exprhs
        _Convex_Constraint.__init__(
            self, Ptmp, constring, 'nth root of det ineq')
        self.prefix = '_ndt'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        return detrootn(self.expdet).value - self.exprhs.value

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)


class Sym_Trunc_Simplex_Constraint(_Convex_Constraint):
    """ A temporary object used to pass symmetrized truncated simplex constraints.
    This class derives from :class:`Constraint <picos.Constraint>`
    """

    def __init__(self, exp, radius, Ptmp, constring):
        self.exp = exp
        self.radius = radius
        _Convex_Constraint.__init__(self, Ptmp, constring,
                                    'symmetrized truncated simplex constraint')
        self.prefix = '_nts'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        return cvx.matrix([1 - norm(self.exp, 'inf').value,
                           self.radius - norm(self.exp, 1).value])

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)


class Sumklargest_Constraint(_Convex_Constraint):
    """ A temporary object used to pass constraints involving "sum of k largest elements".
    This class derives from :class:`Constraint <picos.Constraint>`
    """

    def __init__(self, rhs, exp, k, eigenvalues, islargest, Ptmp, constring):
        self.rhs = rhs
        self.exp = exp
        self.k = k
        self.eigs = eigenvalues  # bool -> sum of eigenvalues of normal elements
        # bool -> False when it is in fact a sum of k smallest elements >= rhs
        self.islargest = islargest
        if islargest:
            _Convex_Constraint.__init__(self, Ptmp, constring,
                                        'sum_k_largest constraint')
        else:
            _Convex_Constraint.__init__(self, Ptmp, constring,
                                        'sum_k_smallest constraint')
        self.prefix = '_nsk'
        """prefix to be added to the names of the temporary variables when add_constraint() is called"""

    def slack_var(self):
        if self.islargest:
            if self.eigs:
                return cvx.matrix(
                    self.rhs.value -
                    sum_k_largest_lambda(
                        self.exp,
                        self.k).value)
            else:
                return cvx.matrix(
                    self.rhs.value -
                    sum_k_largest(
                        self.exp,
                        self.k).value)
        else:
            if self.eigs:
                return cvx.matrix(
                    sum_k_smallest_lambda(
                        self.exp,
                        self.k).value -
                    self.rhs.value)
            else:
                return cvx.matrix(
                    sum_k_smallest(
                        self.exp,
                        self.k).value -
                    self.rhs.value)

    slack = property(slack_var, Constraint.set_slack, Constraint.del_slack)
