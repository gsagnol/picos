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

from .tools import *
from .expression import *
from .constraint import *

__all__ = ['Problem', 'Variable']

global INFINITY
INFINITY = 1e16


class Problem(object):
    """This class represents an optimization problem.
    The constructor creates an empty problem.
    Some options can be provided under the form
    ``key = value``.
    See the list of available options
    in the doc of :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`
    """

    def __init__(self, **options):
        self.objective = ('find', None)  # feasibility problem only
        self.constraints = []
        """list of all constraints"""

        self._deleted_constraints = []
        self.variables = {}
        """dictionary of variables indexed by variable names"""
        self.countVar = 0
        """number of (multidimensional) variables"""
        self.countCons = 0
        """numner of (multidimensional) constraints"""
        self.numberOfVars = 0
        """total number of (scalar) variables"""
        self.numberAffConstraints = 0
        """total number of (scalar) affine constraints"""
        self.numberConeVars = 0
        """number of auxilary variables required to handle the SOC constraints"""
        self.numberConeConstraints = 0
        """number of SOC constraints"""
        self.numberLSEConstraints = 0
        """number of LogSumExp constraints (+1 if the objective is a LogSumExp)"""
        self.numberLSEVars = 0
        """number of vars in LogSumExp expressions"""
        self.numberQuadConstraints = 0
        """number of quadratic constraints (+1 if the objective is quadratic)"""
        self.numberQuadNNZ = 0
        """number of nonzero entries in the matrices defining the quadratic expressions"""
        self.numberSDPConstraints = 0
        """number of SDP constraints"""
        self.numberSDPVars = 0
        """size of the s-vecotrized matrices involved in SDP constraints"""
        self.countGeomean = 0
        """number of geomean (and other nonstandard convex) inequalities"""
        self.cvxoptVars = {'c': None,
                           'A': None, 'b': None,  # equalities
                           'Gl': None, 'hl': None,  # inequalities
                           'Gq': None, 'hq': None,  # quadratic cone
                           'Gs': None, 'hs': None,  # semidefinite cone
                           'F': None, 'g': None,  # GP constraints
                           'quadcons': None}  # other quads

        self.glpk_Instance = None

        self.gurobi_Instance = None
        self.grbvar = []
        self.grb_boundcons = None
        self.grbcons = {}

        self.cplex_Instance = None
        self.cplex_boundcons = None

        self.msk_env = None
        self.msk_task = None
        self.msk_fxd = None
        # msk_scaledcols[i] = a means that msk instance uses a change of
        # variable x[i]' = a*x[i], where x[i] is the original var of the
        # Problem and x[i]' is passed to MOSEK
        self.msk_scaledcols = None
        # a pair (k,j) in msk_scaledcols[i] means that the variable x'[j]
        # passed to mosek belongs to the ith cone, in the kth position.
        self.msk_fxdconevars = None
        #index of active mosek cones (i.e., not deleted)
        self.msk_active_cones = None
        #index of active mosek constraints (not deleted)
        self.msk_active_cons = None

        self.scip_model = None
        self.scip_vars = None
        self.scip_obj = None

        self.sdpa_executable = None
        self.sdpa_dats_filename = None
        self.sdpa_out_filename = None

        self.groupsOfConstraints = {}
        self.listOfVars = {}
        self.consNumbering = []

        self._options = _NonWritableDict()
        if options is None:
            options = {}
        self.set_all_options_to_default()
        self.update_options(**options)

        self.number_solutions = 0

        self.longestkey = 0  # for a nice display of constraints
        self.varNames = []

        self.status = 'unsolved'
        """status returned by the solver. The default when
                   a new problem is created is 'unsolved'.
                """

        self.obj_passed = []
        """list of solver instances where the objective has been passed"""

        self._complex = False  # problem has complex coefs

    ''' TO CHECK are we really entering this function? look at the doc of __del__
    def __del__(self):
        """
        clean-up solver instances that must deleted manually
        """
        del self.msk_env
        del self.msk_task
        del self.cplex_Instance
        del self.gurobi_Instance
        del self.scip_model
    '''

    def __str__(self):
        probstr = '---------------------\n'
        probstr += 'optimization problem  ({0}):\n'.format(self.type)
        probstr += '{0} variables, {1} affine constraints'.format(
            self.numberOfVars, self.numberAffConstraints)

        if self.numberConeVars > 0:
            probstr += ', {0} vars in {1} SO cones'.format(
                self.numberConeVars, self.numberConeConstraints)
        if self.numberLSEConstraints > 0:
            probstr += ', {0} vars in {1} LOG-SUM-EXP'.format(
                self.numberLSEVars, self.numberLSEConstraints)
        if self.numberSDPConstraints > 0:
            probstr += ', {0} vars in {1} SD cones'.format(
                self.numberSDPVars, self.numberSDPConstraints)
        if self.numberQuadConstraints > 0:
            probstr += ', {0} nnz  in {1} quad constraints'.format(
                self.numberQuadNNZ, self.numberQuadConstraints)
        probstr += '\n'

        printedlis = []
        for vkey in self.variables.keys():
            if vkey[
                    :4] in (
                    '_geo',
                    '_nop',
                    '_ntp',
                    '_ndt',
                    '_nts',
                    '_npq',
                    '_nsk'):
                continue
            if '[' in vkey and ']' in vkey:
                lisname = vkey[:vkey.index('[')]
                if lisname not in printedlis:
                    printedlis.append(lisname)
                    var = self.listOfVars[lisname]
                    probstr += '\n' + lisname + ' \t: '
                    probstr += var['type'] + ' of ' + \
                        str(var['numvars']) + ' variables, '
                    if var['size'] == 'different':
                        probstr += 'different sizes'
                    else:
                        probstr += str(var['size'])
                    if var['vtype'] == 'different':
                        probstr += ', different type'
                    else:
                        probstr += ', ' + var['vtype']
                    probstr += var['bnd']
            else:
                var = self.variables[vkey]
                probstr += '\n' + vkey + ' \t: ' + \
                    str(var.size) + ', ' + var.vtype + var._bndtext
        probstr += '\n'
        if self.objective[0] == 'max':
            probstr += '\n\tmaximize ' + self.objective[1].string + '\n'
        elif self.objective[0] == 'min':
            probstr += '\n\tminimize ' + self.objective[1].string + '\n'
        elif self.objective[0] == 'find':
            probstr += '\n\tfind vars\n'
        probstr += 'such that\n'
        if self.countCons == 0:
            probstr += '  []\n'
        k = 0
        while k < self.countCons:
            if k in self.groupsOfConstraints.keys():
                lcur = len(self.groupsOfConstraints[k][2])
                if lcur > 0:
                    lcur += 2
                    probstr += '(' + self.groupsOfConstraints[k][2] + ')'
                if self.longestkey == 0:
                    ntabs = 0
                else:
                    ntabs = int(np.ceil((self.longestkey + 2) / 8.0))
                missingtabs = int(np.ceil(((ntabs * 8) - lcur) / 8.0))
                for i in range(missingtabs):
                    probstr += '\t'
                if lcur > 0:
                    probstr += ': '
                else:
                    probstr += '  '
                probstr += self.groupsOfConstraints[k][1]
                k = self.groupsOfConstraints[k][0] + 1
            else:
                probstr += self.constraints[
                    k].keyconstring(self.longestkey) + '\n'
                k += 1
        probstr += '---------------------'
        return probstr

    """
        ----------------------------------------------------------------
        --                       Utilities                            --
        ----------------------------------------------------------------
        """

    def remove_solver_from_passed(self, solver):
        for cons in self.constraints:
            if solver in cons.passed:
                cons.passed.remove(solver)
        if solver in self.obj_passed:
            self.obj_passed.remove(solver)
        for var in self.variables.values():
            if solver in var.passed:
                var.passed.remove(solver)
            if solver == 'gurobi' and hasattr(var, 'gurobi_endIndex'):
                del var.gurobi_startIndex
                del var.gurobi_endIndex
            if solver == 'cplex' and hasattr(var, 'cplex_endIndex'):
                del var.cplex_startIndex
                del var.cplex_endIndex

    def reset_cvxopt_instance(self, onlyvar=True):
        """reset the variable ``cvxoptVars``, used by the solver cvxopt (and smcp)"""

        self.cvxoptVars = {'c': None,
                           'A': None, 'b': None,  # equalities
                           'Gl': None, 'hl': None,  # inequalities
                           'Gq': None, 'hq': None,  # quadratic cone
                           'Gs': None, 'hs': None,  # semidefinite cone
                           'F': None, 'g': None,  # GP constraints
                           'quadcons': None}  # other quads

        if onlyvar:
            self.remove_solver_from_passed('cvxopt')

    def reset_glpk_instance(self, onlyvar=True):
        """reset the variables used by the solver glpk"""

        if self.glpk_Instance is not None:
            import swiglpk as glpk

            glpk.glp_delete_prob(self.glpk_Instance)
            self.glpk_Instance = None

        if onlyvar:
            self.remove_solver_from_passed('glpk')

    def reset_gurobi_instance(self, onlyvar=True):
        """reset the variables used by the solver gurobi"""

        self.gurobi_Instance = None
        self.grbvar = []
        self.grb_boundcons = None
        self.grbcons = {}

        if onlyvar:
            self.remove_solver_from_passed('gurobi')

    def reset_cplex_instance(self, onlyvar=True):
        """reset the variables used by the solver cplex"""

        self.cplex_Instance = None
        self.cplex_boundcons = None

        if onlyvar:
            self.remove_solver_from_passed('cplex')

    def reset_mosek_instance(self, onlyvar=True):
        """reset the variables used by the solver mosek"""

        self.msk_env = None
        self.msk_task = None
        self.msk_scaledcols = None
        self.msk_fxd = None
        self.msk_fxdconevars = None
        self.msk_active_cones = None
        self.msk_active_cons = None

        if onlyvar:
            self.remove_solver_from_passed('mosek')

    def reset_scip_instance(self, onlyvar=True):
        """reset the variables used by the solver scip"""

        self.scip_model = None
        self.scip_vars = None
        self.scip_obj = None

        if onlyvar:
            self.remove_solver_from_passed('scip')

    def reset_sdpa_instance(self,onlyvar=True):
        self.sdpa_executable = None
        self.sdpa_dats_filename = None
        self.sdpa_out_filename = None

        if onlyvar:
            self.remove_solver_from_passed('sdpa')

    def reset_solver_instances(self):

        self.reset_cvxopt_instance(False)
        self.reset_glpk_instance(False)
        self.reset_gurobi_instance(False)
        self.reset_cplex_instance(False)
        self.reset_mosek_instance(False)
        self.reset_scip_instance(False)
        self.reset_sdpa_instance(False)

        for cons in self.constraints:
            cons.passed = []
        self.obj_passed = []

        for var in self.variables.values():
            var.passed = []
            if hasattr(var, 'gurobi_endIndex'):
                del var.gurobi_startIndex
                del var.gurobi_endIndex
            if hasattr(var, 'cplex_endIndex'):
                del var.cplex_startIndex
                del var.cplex_endIndex

    def remove_all_constraints(self):
        """
        Removes all constraints from the problem
        This function does not remove *hard-coded bounds* on variables;
        use the function :func:`remove_all_variable_bounds() <picos.Problem.remove_all_variable_bounds>`
        to do so.
        """
        self.numberConeConstraints = 0
        self.numberAffConstraints = 0
        self.numberQuadConstraints = 0
        self.numberSDPConstraints = 0
        self.numberLSEConstraints = 0
        self.countGeomean = 0
        self.consNumbering = []
        self.groupsOfConstraints = {}
        self.numberConeVars = 0
        self.numberSDPVars = 0
        self.countCons = 0
        self.constraints = []
        self.numberQuadNNZ = 0
        self.numberLSEVars = 0
        self.countGeomean = 0
        if self.objective[0] is not 'find':
            if self.objective[1] is not None:
                expr = self.objective[1]
                if isinstance(expr, QuadExp):
                    self.numberQuadNNZ = expr.nnz()
                if isinstance(expr, LogSumExp):
                    self.numberLSEVars = expr.Exp.size[0] * expr.Exp.size[1]
        self.reset_solver_instances()

    def remove_all_variable_bounds(self):
        """
        remove all the lower and upper bounds on variables (i.e,,
        *hard-coded bounds* passed in the attribute ``bnd`` of the variables.
        """
        for var in self.variables.values():
            var.bnd._reset()

    def obj_value(self):
        """
        If the problem was already solved, returns the objective value.
        Otherwise, it raises an ``AttributeError``.
        """
        return self.objective[1].eval()[0]

    def get_varName(self, Id):
        return [k for k in self.variables.keys() if self.variables[
            k].Id == Id][0]

    def set_objective(self, typ, expr):
        """
        Defines the objective function of the problem.

        :param typ: can be either ``'max'`` (maximization problem),
                    ``'min'`` (minimization problem),
                    or ``'find'`` (feasibility problem).
        :type typ: str.
        :param expr: an :class:`Expression <picos.Expression>`. The expression to be minimized
                     or maximized. This parameter will be ignored
                     if ``typ=='find'``.
        """
        if typ == 'find':
            self.objective = (typ, None)
            return
        if (isinstance(expr, AffinExp) and expr.size != (1, 1)):
            raise Exception('objective should be scalar')
        if not (isinstance(expr, AffinExp) or isinstance(expr, LogSumExp)
                or isinstance(expr, QuadExp) or isinstance(expr, GeneralFun)):
            raise Exception('unsupported objective')
        if isinstance(self.objective[1], LogSumExp):
            oldexp = self.objective[1]
            self.numberLSEConstraints -= 1
            self.numberLSEVars -= oldexp.Exp.size[0] * oldexp.Exp.size[1]
        if isinstance(self.objective[1], QuadExp):
            oldexp = self.objective[1]
            self.numberQuadConstraints -= 1
            self.numberQuadNNZ -= oldexp.nnz()
        if isinstance(expr, LogSumExp):
            self.numberLSEVars += expr.Exp.size[0] * expr.Exp.size[1]
            self.numberLSEConstraints += 1
        if isinstance(expr, QuadExp):
            self.numberQuadConstraints += 1
            self.numberQuadNNZ += expr.nnz()
        self.objective = (typ, expr)
        self.obj_passed = []  # reset the solvers which know this objective function

    def set_var_value(self, name, value, optimalvar=False):
        """
        Sets the :attr:`value<picos.Variable.value>` attribute of the
        given variable.

        ..
           This can be useful to check
           the value of a complicated :class:`Expression <picos.Expression>`,
           or to use a solver with a *hot start* option.

        :param name: name of the variable to which the value will be given
        :type name: str.
        :param value: The value to be given. The type of
                      ``value`` must be recognized by the function
                      :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`,
                      so that it can be parsed into a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`
                      of the desired size.

        **Example**

        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',2)
        >>> prob.set_var_value('x',[3,4])  #this is in fact equivalent to x.value=[3,4]
        >>> abs(x)**2
        #quadratic expression: ||x||**2 #
        >>> print (abs(x)**2)
        25.0
        """
        ind = None
        if isinstance(name, tuple):  # alternative solution
            ind = name[0]
            name = name[1]

        if not name in self.variables.keys():
            raise Exception('unknown variable name')
        valuemat, valueString = _retrieve_matrix(
            value, self.variables[name].size)
        if valuemat.size != self.variables[name].size:
            raise Exception(
                'should be of size {0}'.format(
                    self.variables[name].size))
        if ind is None:
            # svectorization for symmetric is done by the value property
            self.variables[name].value = valuemat
            if optimalvar:
                self.number_solutions = max(self.number_solutions, 1)
        else:
            if self.variables[name].vtype in ('symmetric',):
                valuemat = svec(valuemat)
            self.variables[name].value_alt[ind] = valuemat
            if optimalvar:
                self.number_solutions = max(self.number_solutions, ind + 1)

    def _makeGandh(self, affExpr):
        """if affExpr is an affine expression,
        this method creates a bloc matrix G to be multiplied by the large
        vectorized vector of all variables,
        and returns the vector h corresponding to the constant term.
        """
        n1 = affExpr.size[0] * affExpr.size[1]
        # matrix G
        I = []
        J = []
        V = []
        for var in affExpr.factors:
            si = var.startIndex
            facvar = affExpr.factors[var]
            if not isinstance(facvar, cvx.base.spmatrix):
                facvar = cvx.sparse(facvar)
            I.extend(facvar.I)
            J.extend([si + j for j in facvar.J])
            V.extend(facvar.V)
        G = spmatrix(V, I, J, (n1, self.numberOfVars))

        # is it really sparse ?
        # if cvx.nnz(G)/float(G.size[0]*G.size[1])>0.5:
        #       G=cvx.matrix(G,tc='d')
        # vector h
        if affExpr.constant is None:
            h = cvx.matrix(0, (n1, 1), tc='d')
        else:
            h = affExpr.constant
        if not isinstance(h, cvx.matrix):
            h = cvx.matrix(h, tc='d')
        if h.typecode != 'd':
            h = cvx.matrix(h, tc='d')
        return G, h

    def set_all_options_to_default(self):
        """set all the options to their default.
        The following options are available, and can be passed
        as pairs of the form ``key=value`` when the :class:`Problem <picos.Problem>` object is created,
        or to the function :func:`solve() <picos.Problem.solve>` :

        * General options common to all solvers:

          * ``verbose = 1`` : Verbosity level.
            `-1` attempts to suppress all output, even errors.
            `0` only outputs warnings and errors.
            `1` generates standard informative output.
            `2` prints all available information for debugging purposes.

          * ``solver = None`` : currently the available solvers are
            ``'cvxopt'``, ``'glpk'``, ``'cplex'``, ``'mosek'``, ``'gurobi'``,
            ``'smcp'``, ``'zibopt'``. The default ``None`` means that you let
            picos select a suitable solver for you.

          * ``tol = 1e-8`` : Relative gap termination tolerance
            for interior-point optimizers (feasibility and complementary slackness).
            *This option is currently ignored by glpk*.

          * ``maxit = None`` : maximum number of iterations
            (for simplex or interior-point optimizers).
            *This option is currently ignored by zibopt*.

          * ``lp_root_method = None`` : algorithm used to solve continuous LP
            problems, including the root relaxation of mixed integer problems.
            The default ``None`` selects automatically an algorithm.
            If set to ``psimplex`` (resp. ``dsimplex``, ``interior``), the solver
            will use a primal simplex (resp. dual simplex, interior-point) algorithm.
            *This option currently works only with cplex, mosek and gurobi. With
            glpk it works for LPs but not for the MIP root relaxation.*

          * ``lp_node_method = None`` : algorithm used to solve subproblems
            at nodes of the branching trees of mixed integer programs.
            The default ``None`` selects automatically an algorithm.
            If set to ``psimplex`` (resp. ``dsimplex``, ``interior``), the solver
            will use a primal simplex (resp. dual simplex, interior-point) algorithm.
            *This option currently works only with cplex, mosek and gurobi*.

          * ``timelimit = None`` : time limit for the solver, in seconds. The default
            ``None`` means no time limit.
            *This option is currently ignored by cvxopt and smcp*.

          * ``treememory = None``  : size of the buffer for the branch and bound tree,
            in Megabytes.
            *This option currently works only with cplex*.

          * ``gaplim = 1e-4`` : For mixed integer problems,
            the solver returns a solution as soon as this value for the gap is reached
            (relative gap between the primal and the dual bound).

          * ``noprimals = False`` : if ``True``, do not copy the optimal variable values in the
            :attr:`value<picos.Variable.value>` attribute of the problem variables.

          * ``noduals = False`` : if ``True``, do not try to retrieve the dual variables.

          * ``nbsol = None`` : maximum number of feasible solution nodes visited
            when solving a mixed integer problem.

          * ``hotstart = False`` : if ``True``, the MIP optimizer tries to start from
            the solution
            specified (even partly) in the :attr:`value<picos.Variable.value>` attribute of the
            problem variables.
            *This option currently works only with cplex, mosek and gurobi*.

          * ``convert_quad_to_socp_if_needed = True`` : Do we convert the convex quadratics to
            second order cone constraints when the solver does not handle them directly ?

          * ``solve_via_dual = None`` : If set to ``True``, the Lagrangian Dual (computed with the
            function :func:`dualize() <picos.Problem.dualize>` ) is passed to the
            solver, instead of the problem itself. In some situations this can yield an
            important speed-up. In particular for Mosek and SOCPs/SDPs whose form is close to the
            standard primal form (as in the :ref:`note on dual variables <noteduals>` of the tutorial),
            since the MOSEK interface is better adapted for problems given in a *dual form*.
            When this option is set to ``None`` (default), PICOS chooses automatically whether the problem
            itself should be passed to the solver, or rather its dual.

          * ``pass_simple_cons_as_bound = False`` : If set to ``True``, linear constraints involving a
            single variable are passed to the solvers as a bound on the variable. This may speed-up
            the solving process (?), but is not safe if you indend to remove this constraint later
            and re-solve the problem.
            *This option currently works only with cplex, mosek and gurobi*.

          * ``return_constraints = False`` : If set to ``True``, the default behaviour of the function
            :func:`add_constraint() <picos.Problem.add_constraint>` is to return
            the created constraint.

        * Specific options available for cvxopt/smcp:

          * ``feastol = None`` : feasibility tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
            If ``feastol`` has the default value ``None``,
            then the value of the option ``tol`` is used.

          * ``abstol = None`` : absolute tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
            If ``abstol`` has the default value ``None``,
            then the value of the option ``tol`` is used.

          * ``reltol = None`` : relative tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
            If ``reltol`` has the default value ``None``,
            then the value of the option ``tol``, multiplied by ``10``, is used.

        * Specific options available for cplex:

          * ``cplex_params = {}`` : a dictionary of
            `cplex parameters <http://pic.dhe.ibm.com/infocenter/cosinfoc/v12r2/index.jsp?topic=%2Filog.odms.cplex.help%2FContent%2FOptimization%2FDocumentation%2FCPLEX%2F_pubskel%2FCPLEX934.html>`_
            to be set before the cplex
            optimizer is called. For example,
            ``cplex_params={'mip.limits.cutpasses' : 5}``
            will limit the number of cutting plane passes when solving the root node
            to ``5``.

          * ``acceptable_gap_at_timelimit = None`` : If the the time limit is reached,
            the optimization process is aborted only if the current gap is less
            than this value. The default value ``None`` means that we
            interrupt the computation regardless of the achieved gap.

          * ``uboundlimit = None`` : tells CPLEX to stop as soon as an upper
            bound smaller than this value is found.

          * ``lboundlimit = None`` : tells CPLEX to stop as soon as a lower
            bound larger than this value is found.

          * ``boundMonitor = True`` : tells CPLEX to store information about
            the evolution of the bounds during the solving process. At the end
            of the computation, a list of triples ``(time,lowerbound,upperbound)``
            will be provided in the field ``bounds_monitor`` of the dictionary
            returned by :func:`solve() <picos.Problem.solve>`.

        * Specific options available for mosek:

          * ``mosek_params = {}`` : a dictionary of
            `mosek parameters <http://docs.mosek.com/6.0/pyapi/node017.html>`_
            to be set before the mosek
            optimizer is called. For example,
            ``mosek_params={'simplex_abs_tol_piv' : 1e-4}``
            sets the absolute pivot tolerance of the
            simplex optimizer to ``1e-4``.

          * ``handleBarVars = True`` : For semidefinite programming,
            Mosek handles the Linear Matrix Inequalities by using a separate
            class of variables, called *bar variables*, representing semidefinite positive matrices.

            If this option is set to ``False``, Mosek adds a new *bar variable* for every LMI, and let the
            elements of the slack variable of the LMIs match the bar variables by adding equality constraints.

            If set to ``True`` (default), PICOS avoid creating useless bar variables for LMIs of the form ``X >> 0``: in this
            case ``X`` will be added in mosek directly as a ``bar variable``. This can avoid
            creating a lot of unnecessary variables for problems whose form is close to the canonical
            dual form (See the :ref:`note on dual variables <noteduals>` in the tutorial).

            See also the option ``solve_via_dual``.

          * ``handleConeVars = True`` : For Second Order Cone Programming, Mosek handles the SOC inequalities
            by *appending a standard cone*. This must be done in a careful way, since
            a single variable is not allowed to belong to several standard cones.

            If this option is set to ``False``, Picos adds a new variable for each coordinate of a vector
            in a second order cone inequality, as well as an equality constraint to match the value of
            this coordinate with the value of the new variable.

            If set to ``True``, additional variables are added only when needed, and simple changes of variables
            are done in order to reduce the number of necessary additional variables.
            This can avoid
            creating a lot of unnecessary variables for problems whose form is close to the canonical
            dual form (See the :ref:`note on dual variables <noteduals>` in the tutorial).
            Consider for example
            the SOC inequality :math:`\Vert [x,x-y,3z,t] \Vert \leq 5t`. Here 2 new variables :math:`u`
            and :math:`v` will be added, with the constraint :math:`x-y=u` and :math:`5t=v`,
            and a change of variable :math:`z'=3z` will be done. Then a standard cone with the variables
            :math:`(v,x,u,z',t)` will be appended (cf. the doc of the mosek interface).

            See also the option ``solve_via_dual``.

        * Specific options available for gurobi:

          * ``gurobi_params = {}`` : a dictionary of
            `gurobi parameters <http://www.gurobi.com/documentation/5.0/reference-manual/node653>`_
            to be set before the gurobi
            optimizer is called. For example,
            ``gurobi_params={'NodeLimit' : 25}``
            limits the number of nodes visited by the MIP optimizer to 25.

        * Specific options available for scip:

          * ``scip_params = {}`` : a dictionary of
            `scip parameters <http://scip.zib.de/doc-2.0.2/html/PARAMETERS.html>`_
            to be set before the scip
            optimizer is called. For example,
            ``scip_params = {'lp/threads' : 4}``
            sets the number of threads to solve the LPs at 4.


        * Specific options available for sdpa:

          * ``sdpa_executable = 'sdpa'`` : The sdpa executable name.

          * ``sdpa_params = {'-pt': 0}`` : dictionary of extra parameters to pass to sdpa.
            Set 'read_solution': 'filename.out' for reading an already existing solution.
        """
        # Additional, hidden option (requires a patch of smcp, to use conlp to
        # interface the feasible starting point solver):
        #
        #* 'smcp_feas'=False [if True, use the feasible start solver with SMCP]
        default_options = {'tol': 1e-8,
                           'feastol': None,
                           'abstol': None,
                           'reltol': None,
                           'maxit': None,
                           'verbose': 1,
                           'solver': None,
                           'step_sqp': 1,  # undocumented
                           'harmonic_steps': 1,  # undocumented
                           'noprimals': False,
                           'noduals': False,
                           'smcp_feas': False,  # undocumented
                           'nbsol': None,
                           'timelimit': None,
                           'acceptable_gap_at_timelimit': None,
                           'treememory': None,
                           'gaplim': 1e-4,
                           'pool_gap': None,  # undocumented
                           'pool_size': None,  # undocumented
                           'lp_root_method': None,
                           'lp_node_method': None,
                           'cplex_params': {},
                           'mosek_params': {},
                           'gurobi_params': {},
                           'scip_params': {},
                           'convert_quad_to_socp_if_needed': True,
                           'hotstart': False,
                           'uboundlimit': None,
                           'lboundlimit': None,
                           'boundMonitor': False,
                           'handleBarVars': True,
                           'handleConeVars': True,
                           'solve_via_dual': None,
                           'pass_simple_cons_as_bound' : False,
                           'return_constraints' : False,
                           'sdpa_executable': 'sdpa',
                           'sdpa_params': {'-pt': 0},
                           }

        self._options = _NonWritableDict(default_options)

    @property
    def options(self):
        return self._options

    def set_option(self, key, val):
        """
        Sets the option **key** to the value **val**.

        :param key: The key of an option
                    (see the list of keys in the doc of
                    :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`).
        :type key: str.
        :param val: New value for the option.
        """
        if key in (
                'handleBarVars',
                'handleConeVars') and val != self.options[key]:
            # because we must pass in make_mosek_instance again.
            self.reset_solver_instances()
        if key not in self.options:
            raise AttributeError('unknown option key :' + str(key))
        self.options._set(key, val)
        if key == 'verbose' and isinstance(val, bool):
            self.options._set('verbose', int(val))

        # trick to force the use of mosek6 during the tests:
        # if val=='mosek':
        #        self.options._set('solver','mosek6')

    def update_options(self, **options):
        """
        update the option dictionary, for each pair of the form
        ``key = value``. For a list of available options and their default values,
        see the doc of :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`.
        """

        for k in options.keys():
            self.set_option(k, options[k])

    def _eliminate_useless_variables(self):
        """
        Removes from the problem the variables that do not
        appear in any constraint or in the objective function.
        """
        foundVars = set([])
        for cons in self.constraints:
            if isinstance(cons.Exp1, AffinExp):
                foundVars.update(cons.Exp1.factors.keys())
                foundVars.update(cons.Exp2.factors.keys())
                if not cons.Exp3 is None:
                    foundVars.update(cons.Exp3.factors.keys())
            elif isinstance(cons.Exp1, QuadExp):
                foundVars.update(cons.Exp1.aff.factors.keys())
                for ij in cons.Exp1.quad:
                    foundVars.update(ij)
            elif isinstance(cons.Exp1, LogSumExp):
                foundVars.update(cons.Exp1.Exp.factors.keys())
        if not self.objective[1] is None:
            obj = self.objective[1]
            if isinstance(obj, AffinExp):
                foundVars.update(obj.factors.keys())
            elif isinstance(obj, QuadExp):
                foundVars.update(obj.aff.factors.keys())
                for ij in obj.quad:
                    foundVars.update(ij)
            elif isinstance(obj, LogSumExp):
                foundVars.update(obj.Exp.factors.keys())

        vars2del = []
        for vname, v in six.iteritems(self.variables):
            if v not in foundVars:
                vars2del.append(vname)

        for vname in sorted(vars2del):
            self.remove_variable(vname)
            if self.options['verbose'] > 1:
                print(
                    'variable ' +
                    vname +
                    ' was useless and has been removed')

    """
        ----------------------------------------------------------------
        --                TOOLS TO CREATE AN INSTANCE                 --
        ----------------------------------------------------------------
        """

    def add_variable(
            self,
            name,
            size=1,
            vtype='continuous',
            lower=None,
            upper=None):
        """
        adds a variable in the problem,
        and returns the corresponding instance of the :class:`Variable <picos.Variable>`.

        For example,

        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',3)
        >>> x
        # variable x:(3 x 1),continuous #

        :param name: The name of the variable.
        :type name: str.
        :param size: The size of the variable.

                     Can be either:

                        * an ``int`` *n* , in which case the variable is a **vector of dimension n**

                        * or a ``tuple`` *(n,m)*, and the variable is a **n x m-matrix**.

        :type size: int or tuple.
        :param vtype: variable :attr:`type <picos.Variable.vtype>`.
                      Can be:

                        * ``'continuous'`` (default),

                        * ``'binary'``: 0/1 variable

                        * ``'integer'``: integer valued variable

                        * ``'symmetric'``: symmetric matrix

                        * ``'antisym'``: antisymmetric matrix

                        * ``'complex'``: complex matrix variable

                        * ``'hermitian'``: complex hermitian matrix

                        * ``'semicont'``: 0 or continuous variable satisfying its bounds (supported by CPLEX and GUROBI only)

                        * ``'semiint'``: 0 or integer variable satisfying its bounds (supported by CPLEX and GUROBI only)

        :type vtype: str.
        :param lower: a lower bound for the variable. Can be either a vector/matrix of the
                      same size as the variable, or a scalar (in which case all elements
                      of the variable have the same lower bound).
        :type lower: Any type recognized by the function
                      :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`.
        :param upper: an upper bound for the variable. Can be either a vector/matrix of the
                      same size as the variable, or a scalar (in which case all elements
                      of the variable have the same upper bound).
        :type upper: Any type recognized by the function
                      :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`.

        :returns: An instance of the class :class:`Variable <picos.Variable>`.
        """

        # TODOC tutorial examples with bounds and sparse bounds

        if name in self.variables:
            raise Exception('this variable already exists')
        if isinstance(size, six.integer_types):
            size = (int(size), 1)
        else:
            size = tuple(int(x) for x in size)
        if len(size) == 1:
            size = (int(size[0]), 1)

        lisname = None
        if '[' in name and ']' in name:  # list or dict of variables
            lisname = name[:name.index('[')]
            ind = name[name.index('[') + 1:name.index(']')]
            if lisname in self.listOfVars:
                oldn = self.listOfVars[lisname]['numvars']
                self.listOfVars[lisname]['numvars'] += 1
                if size != self.listOfVars[lisname]['size']:
                    self.listOfVars[lisname]['size'] = 'different'
                if vtype != self.listOfVars[lisname]['vtype']:
                    self.listOfVars[lisname]['vtype'] = 'different'
                if self.listOfVars[lisname][
                        'type'] == 'list' and ind != str(oldn):
                    self.listOfVars[lisname]['type'] = 'dict'
            else:
                self.listOfVars[lisname] = {
                    'numvars': 1, 'size': size, 'vtype': vtype}
                if ind == '0':
                    self.listOfVars[lisname]['type'] = 'list'
                else:
                    self.listOfVars[lisname]['type'] = 'dict'

        countvar = self.countVar
        numbervar = self.numberOfVars

        if vtype in ('symmetric', 'hermitian'):
            if size[0] != size[1]:
                raise ValueError('symmetric variables must be square')
            s0 = size[0]
            self.numberOfVars += s0 * (s0 + 1) // 2
        elif vtype == 'antisym':
            if size[0] != size[1]:
                raise ValueError('antisymmetric variables must be square')
            s0 = size[0]
            if s0 <= 1:
                raise ValueError('dimension too small')
            if not(lower is None and upper is None):
                raise ValueError(
                    'hard lower / upper bounds not supported for antisym variables. Add a constraint instead.')
            idmat = _svecm1_identity(vtype, (s0, s0))
            Xv = self.add_variable(name + '_utri',
                                   (s0 * (s0 - 1)) // 2,
                                   vtype='continuous',
                                   lower=None,
                                   upper=None)
            exp = idmat * Xv
            exp.string = name
            exp._size = (s0, s0)
            return exp

        elif vtype == 'complex':
            l = None
            lr, li, ur, ui = None, None, None, None
            if l:
                l = _retrieve_matrix(l, size)[0]
                if l.typecode == 'z':
                    lr = l.real()
                    li = l.imag()
                else:
                    lr = l
                    li = None
            u = None
            if u:
                u = _retrieve_matrix(u, size)[0]
                if u.typecode == 'z':
                    ur = u.real()
                    ui = u.imag()
                else:
                    ur = u
                    ui = None
            Zr = self.add_variable(
                name + '_RE',
                size,
                vtype='continuous',
                lower=lr,
                upper=ur)
            Zi = self.add_variable(
                name + '_IM',
                size,
                vtype='continuous',
                lower=li,
                upper=ui)
            exp = Zr + 1j * Zi
            exp.string = name
            return exp
        else:
            self.numberOfVars += size[0] * size[1]
        self.varNames.append(name)
        self.countVar += 1

        # svec operation
        idmat = _svecm1_identity(vtype, size)

        self.variables[name] = Variable(self,
                                        name,
                                        size,
                                        countvar,
                                        numbervar,
                                        vtype=vtype,
                                        lower=lower,
                                        upper=upper)
        if lisname is not None:
            if 'bnd' in self.listOfVars[lisname]:
                bndtext = self.listOfVars[lisname]['bnd']
                thisbnd = self.variables[name]._bndtext
                if bndtext != thisbnd:
                    self.listOfVars[lisname]['bnd'] = ', some bounds'
            else:
                self.listOfVars[lisname]['bnd'] = self.variables[name]._bndtext

        return self.variables[name]

    def remove_variable(self, name):
        """
        Removes the variable ``name`` from the problem.
        :param name: name of the variable to remove.
        :type name: str.

        .. Warning:: This method does not check if some constraint still involves the variable
                     to be removed.
        """
        if '[' in name and ']' in name:  # list or dict of variables
            lisname = name[:name.index('[')]
            if lisname in self.listOfVars:
                varattr = self.listOfVars[lisname]
                varattr['numvars'] -= 1
                if varattr['numvars'] == 0:
                    del self.listOfVars[lisname]  # empty list of vars
        if name not in self.variables.keys():
            raise Exception(
                'variable does not exist. Maybe you tried to remove some item x[i] of the variable x ?')
        self.countVar -= 1
        var = self.variables[name]
        sz = var.size
        self.numberOfVars -= sz[0] * sz[1]
        self.varNames.remove(name)
        del self.variables[name]
        self._recomputeStartEndIndices()
        self.reset_solver_instances()

    def _recomputeStartEndIndices(self):
        ind = 0
        for nam in self.varNames:
            var = self.variables[nam]
            var._startIndex = ind
            if var.vtype in ('symmetric',):
                ind += int((var.size[0] * (var.size[0] + 1)) // 2)
            else:
                ind += var.size[0] * var.size[1]
            var._endIndex = ind

    def _remove_temporary_variables(self):
        """
        Remove the variables __tmp...
        created by the solvers to cast the problem as socp
        """
        offset = 0
        todel = []
        for nam in self.varNames:
            var = self.variables[nam]
            if '__tmp' in nam or '__noconstant' in nam:
                self.countVar -= 1
                sz = self.variables[nam].size
                offset += sz[0] * sz[1]
                self.numberOfVars -= sz[0] * sz[1]
                # self.varNames.remove(nam)
                todel.append(nam)
                del self.variables[nam]
            else:
                var._startIndex -= offset
                var._endIndex -= offset

        for nam in todel:
            self.varNames.remove(nam)

        if '__tmprhs' in self.listOfVars:
            del self.listOfVars['__tmprhs']
        if '__tmplhs' in self.listOfVars:
            del self.listOfVars['__tmplhs']

    def copy(self):
        """creates a copy of the problem."""
        import copy
        cop = Problem()
        cvars = {}
        for (iv, v) in sorted([(v.startIndex, v)
                               for v in self.variables.values()]):
            cvars[v.name] = cop.add_variable(v.name, v.size, v.vtype)
        for c in self.constraints:
            """old version doesnt handle conevars and bounded vars
            c2=copy.deepcopy(c)
            c2.Exp1=_copy_exp_to_new_vars(c2.Exp1,cvars)
            c2.Exp2=_copy_exp_to_new_vars(c2.Exp2,cvars)
            c2.Exp3=_copy_exp_to_new_vars(c2.Exp3,cvars)
            if c.semidefVar:
                    c2.semidefVar = cvars[c.semidefVar.name]
            """
            E1 = _copy_exp_to_new_vars(c.Exp1, cvars)
            E2 = _copy_exp_to_new_vars(c.Exp2, cvars)
            E3 = _copy_exp_to_new_vars(c.Exp3, cvars)
            c2 = Constraint(c.typeOfConstraint, None, E1, E2, E3)
            cop.add_constraint(c2, c.key)
        obj = _copy_exp_to_new_vars(self.objective[1], cvars)
        cop.set_objective(self.objective[0], obj)

        cop.consNumbering = copy.deepcopy(self.consNumbering)
        cop.groupsOfConstraints = copy.deepcopy(self.groupsOfConstraints)
        cop._options = _NonWritableDict(self.options)

        return cop

    def add_constraint(self, cons, key=None, ret=False):
        """Adds a constraint in the problem.

        :param cons: The constraint to be added.
        :type cons: :class:`Constraint <picos.Constraint>`
        :param key: Optional parameter to describe the constraint with a key string.
        :type key: str.
        :param ret: Do you want the added constraint to be returned ?
                    This can be a useful handle to extract the optimal dual variable of this constraint
                    or to delete the constraint with delete().
                    Note: The constraint is always returned if the option
                    ``return_constraints`` is set to ``True``.
        :type ret: bool.
        """
        # SPECIAL CASE OF A NONSTANDARD CONVEX CONSTRAINT
        if isinstance(cons, _Convex_Constraint):
            for ui, vui in six.iteritems(cons.Ptmp.variables):
                uiname = cons.prefix + str(self.countGeomean) + '_' + ui
                self.add_variable(uiname, vui.size, vui.vtype)
                si = self.variables[uiname].startIndex
                ei = self.variables[uiname].endIndex
                self.variables[uiname] = vui
                self.variables[uiname]._startIndex = si
                self.variables[uiname]._endIndex = ei
                self.variables[uiname].name = uiname

            indcons = self.countCons
            self.add_list_of_constraints(cons.Ptmp.constraints, key=key)
            goc = self.groupsOfConstraints[indcons]
            goc[1] = cons.constring() + '\n'
            self.countGeomean += 1
            if self.options['return_constraints'] or ret:
                return cons
            else:
                return

        cons.key = key
        if not key is None:
            self.longestkey = max(self.longestkey, len(key))
        self.constraints.append(cons)
        self.consNumbering.append(self.countCons)
        self.countCons += 1
        # is there any complex coef ?
        found = False
        for exp in (cons.Exp1, cons.Exp2, cons.Exp3):
            if exp is None:
                continue
            try:
                dct_facts = exp.factors
            except AttributeError as ex:
                if 'QuadExp' in str(
                        ex):  # quadratic expression (should) not contain complex coefs
                    continue
                else:
                    raise

            for x, fac in six.iteritems(exp.factors):
                if fac.typecode == 'z':
                    if fac.imag():
                        self._complex = True
                        found = True
                        break
                    else:
                        exp.factors[x] = fac.real()
            if not(exp.constant is None) and exp.constant.typecode == 'z':
                if exp.constant.imag():
                    self._complex = True
                    found = True
                else:
                    exp.constant = exp.constant.real()

            if found:
                break

        if cons.typeOfConstraint[:3] == 'lin':
            self.numberAffConstraints += (
                cons.Exp1.size[0] * cons.Exp1.size[1])

        elif cons.typeOfConstraint[2:] == 'cone':
            self.numberConeVars += (cons.Exp1.size[0] * cons.Exp1.size[1]) + 1
            self.numberConeConstraints += 1
            if cons.typeOfConstraint[:2] == 'RS':
                self.numberConeVars += 1

        elif cons.typeOfConstraint == 'lse':
            self.numberLSEVars += (cons.Exp1.size[0] * cons.Exp1.size[1])
            self.numberLSEConstraints += 1
        elif cons.typeOfConstraint == 'quad':
            self.numberQuadConstraints += 1
            self.numberQuadNNZ += cons.Exp1.nnz()
        elif cons.typeOfConstraint[:3] == 'sdp':
            self.numberSDPConstraints += 1
            self.numberSDPVars += (cons.Exp1.size[0]
                                   * (cons.Exp1.size[0] + 1)) // 2
            # is it a simple constraint of the form X>>0 ?
            if cons.semidefVar:
                cons.semidefVar.semiDef = True
        if self.options['return_constraints'] or ret:
            return cons

    def add_list_of_constraints(
            self,
            lst,
            it=None,
            indices=None,
            key=None,
            ret=False):
        u"""adds a list of constraints in the problem.
        This fonction can be used with python list comprehensions
        (see the example below).

        :param lst: list of :class:`Constraint<picos.Constraint>`.
        :param it: Description of the letters which should
                   be used to replace the dummy indices.
                   The function tries to find a template
                   for the string representations of the
                   constraints in the list. If several indices change in the
                   list, their letters should be given as a
                   list of strings, in their order of appearance in the
                   resulting string. For example, if three indices
                   change in the constraints, and you want them to be named
                   ``'i'``, ``'j'`` and ``'k'``, set ``it = ['i','j','k']``.
                   You can also group two indices which always appear together,
                   e.g. if ``'i'`` always appear next to ``'j'`` you
                   could set ``it = [('ij',2),'k']``. Here, the number 2
                   indicates that ``'ij'`` replaces 2 indices.
                   If ``it`` is set to ``None``, or if the function is not
                   able to find a template,
                   the string of the first constraint will be used for
                   the string representation of the list of constraints.
        :type it: None or str or list.
        :param indices: a string to denote the set where the indices belong to.
        :type indices: str.
        :param key: Optional parameter to describe the list of constraints with a key string.
        :type key: str.
        :param ret: Do you want the added list of constraints to be returned ?
                    This can be useful to access the duals of these constraints.
                    Note: The constraint is always returned if the option
                    ``return_constraints`` is set to ``True``.
        :type ret: bool.

        **Example:**

        >>> import picos as pic
        >>> import cvxopt as cvx
        >>> prob=pic.Problem()
        >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(5)]
        >>> x #doctest: +NORMALIZE_WHITESPACE
        [# variable x[0]:(2 x 1),continuous #,
         # variable x[1]:(2 x 1),continuous #,
         # variable x[2]:(2 x 1),continuous #,
         # variable x[3]:(2 x 1),continuous #,
         # variable x[4]:(2 x 1),continuous #]
        >>> y=prob.add_variable('y',5)
        >>> IJ=[(1,2),(2,0),(4,2)]
        >>> w={}
        >>> for ij in IJ:
        ...         w[ij]=prob.add_variable('w[{0}]'.format(ij),3)
        ...
        >>> u=pic.new_param('u',cvx.matrix([2,5]))
        >>> prob.add_list_of_constraints(
        ... [u.T*x[i]<y[i] for i in range(5)],
        ... 'i',
        ... '[5]')
        >>>
        >>> prob.add_list_of_constraints(
        ... [abs(w[i,j])<y[j] for (i,j) in IJ],
        ... [('ij',2)],
        ... 'IJ')
        >>>
        >>> prob.add_list_of_constraints(
        ... [y[t] > y[t+1] for t in range(4)],
        ... 't',
        ... '[4]')
        >>>
        >>> print prob #doctest: +NORMALIZE_WHITESPACE
        ---------------------
        optimization problem (SOCP):
        24 variables, 9 affine constraints, 12 vars in 3 SO cones
        <BLANKLINE>
        x   : list of 5 variables, (2, 1), continuous
        w   : dict of 3 variables, (3, 1), continuous
        y   : (5, 1), continuous
        <BLANKLINE>
            find vars
        such that
          u.T*x[i] < y[i] for all i in [5]
          ||w[ij]|| < y[ij__1] for all ij in IJ
          y[t] > y[t+1] for all t in [4]
        ---------------------

        """
        if not(lst):
            return

        firstCons = self.countCons
        thisconsnums = []
        for ks in lst:
            self.add_constraint(ks)
            cstnum = self.consNumbering.pop()
            thisconsnums.append(cstnum)

        self.consNumbering.append(thisconsnums)

        lastCons = self.countCons - 1
        if key is None:
            key = ''
        else:
            self.longestkey = max(self.longestkey, len(key))
        if it is None:
            strlis = '[' + str(len(lst)) + \
                ' constraints (first: ' + lst[0].constring() + ')]\n'
        else:
            strlis = ' for all '
            if len(it) > 1:
                strlis += '('
            for x in it:
                if isinstance(x, tuple):
                    strlis += x[0]
                else:
                    strlis += x
                strlis += ','
            strlis = strlis[:-1]  # remvove the last comma
            if len(it) > 1:
                strlis += ')'
            if not indices is None:
                strlis += ' in ' + indices
            if isinstance(
                    it,
                    tuple) and len(it) == 2 and isinstance(
                    it[1],
                    int):
                it = (it,)
            if isinstance(it, list):
                it = tuple(it)
            if not isinstance(it, tuple):
                it = (it,)
            lstr = [l.constring()
                    for l in lst if '(first:' not in l.constring()]
            try:
                indstr = putIndices(lstr, it)
                strlis = indstr + strlis + '\n'
            except Exception as ex:
                strlis = '[' + str(len(lst)) + \
                    ' constraints (first: ' + lst[0].constring() + ')]\n'
        self.groupsOfConstraints[firstCons] = [lastCons, strlis, key]
        # remove unwanted subgroup of constraints (which are added when we add
        # list of abstract constraints
        goctodel = []
        for goc in self.groupsOfConstraints:
            if goc > firstCons and goc <= lastCons:
                goctodel.append(goc)
        for goc in goctodel:
            del self.groupsOfConstraints[goc]
        if self.options['return_constraints'] or ret:
            return lst

    def get_valued_variable(self, name):
        """
        Returns the value of the variable (as an :func:`cvxopt matrix <cvxopt:cvxopt.matrix>`)
        with the given ``name``.
        If ``name`` refers to a list (resp. dict) of variables,
        named with the template ``name[index]`` (resp. ``name[key]``),
        then the function returns the list (resp. dict)
        of these variables.

        :param name: name of the variable, or of a list/dict of variables.
        :type name: str.

        .. Warning:: If the problem has not been solved,
                     or if the variable is not valued,
                     this function will raise an Exception.
        """
        exp = self.get_variable(name)
        if isinstance(exp, list):
            for i in range(len(exp)):
                exp[i] = exp[i].eval()
        elif isinstance(exp, dict):
            for i in exp:
                exp[i] = exp[i].eval()
        else:
            exp = exp.eval()
        return exp

    def get_variable(self, name):
        """
        Returns the variable (as a :class:`Variable <picos.Variable>`)
        with the given ``name``.
        If ``name`` refers to a list (resp. dict) of variables,
        named with the template ``name[index]`` (resp. ``name[key]``),
        then the function returns the list (resp. dict)
        of these variables.

        :param name: name of the variable, or of a list/dict of variables.
        :type name: str.
        """
        var = name
        if var in self.listOfVars.keys():
            if self.listOfVars[var]['type'] == 'dict':
                rvar = {}
            else:
                rvar = [0] * self.listOfVars[var]['numvars']
            seenKeys = []
            for ind in [
                vname[
                    len(var) +
                    1:-
                    1] for vname in self.variables.keys() if (
                    vname[
                        :len(var)] == var and vname[
                    len(var)] == '[')]:
                if ind.isdigit():
                    key = int(ind)
                    if key not in seenKeys:
                        seenKeys.append(key)
                    else:
                        key = ind
                elif ',' in ind:
                    isplit = ind.split(',')
                    if isplit[0].startswith('('):
                        isplit[0] = isplit[0][1:]
                    if isplit[-1].endswith(')'):
                        isplit[-1] = isplit[-1][:-1]
                    if all([i.isdigit() for i in isplit]):
                        key = tuple([int(i) for i in isplit])
                        if key not in seenKeys:
                            seenKeys.append(key)
                        else:
                            key = ind
                    else:
                        key = ind
                else:
                    try:
                        key = float(ind)
                    except ValueError:
                        key = ind
                rvar[key] = self.variables[var + '[' + ind + ']']
            return rvar
        elif var + '_IM' in self.variables and var + '_RE' in self.variables:  # complex
            exp = self.variables[var + '_RE'] + \
                1j * self.variables[var + '_IM']
            exp.string = var
            return exp
        elif var not in self.variables and var + '_utri' in self.variables:  # antisym
            vu = self.variables[var + '_utri']
            n = int((1 + (1 + vu.size[0] * 8)**0.5) / 2.)
            idasym = _svecm1_identity('antisym', (n, n))
            exp = idasym * vu
            exp.string = var
            exp._size = (n, n)
            return exp
        elif var in self.variables:
            return self.variables[var]
        else:
            raise Exception('no such variable')

    def get_constraint(self, ind):
        u"""
        returns a constraint of the problem.

        :param ind: There are two ways to index a constraint.

                       * if ``ind`` is an *int* :math:`n`, then the nth constraint (starting from 0)
                         will be returned, where all the constraints are counted
                         in the order where they were passed to the problem.

                       * if ``ind`` is a *tuple* :math:`(k,i)`, then the ith constraint
                         from the kth group of constraints is returned
                         (starting from 0). By
                         *group of constraints*, it is meant a single constraint
                         or a list of constraints added together with the
                         function :func:`add_list_of_constraints() <picos.Problem.add_list_of_constraints>`.

                       * if ``ind`` is a tuple of length 1 :math:`(k,)`,
                         then the list of constraints of the kth group is returned.

        :type ind: int or tuple.

        **Example:**

        >>> import picos as pic
        >>> import cvxopt as cvx
        >>> prob=pic.Problem()
        >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(5)]
        >>> y=prob.add_variable('y',5)
        >>> prob.add_list_of_constraints(
        ... [(1|x[i])<y[i] for i in range(5)],
        ... 'i',
        ... '[5]')
        >>> prob.add_constraint(y>0)
        >>> print prob #doctest: +NORMALIZE_WHITESPACE
        ---------------------
        optimization problem (LP):
        15 variables, 10 affine constraints
        <BLANKLINE>
        x   : list of 5 variables, (2, 1), continuous
        y   : (5, 1), continuous
        <BLANKLINE>
            find vars
        such that
          〈 |1| | x[i] 〉 < y[i] for all i in [5]
          y > |0|
        ---------------------
        >>> prob.get_constraint(1)                              #2d constraint (numbered from 0)
        # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #
        >>> prob.get_constraint((0,3))                          #4th consraint from the 1st group
        # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #
        >>> prob.get_constraint((1,))                           #unique constraint of the 2d 'group'
        # (5x1)-affine constraint: y > |0| #
        >>> prob.get_constraint((0,))                           #list of constraints of the 1st group #doctest: +NORMALIZE_WHITESPACE
        [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
         # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #,
         # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
         # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
         # (1x1)-affine constraint: 〈 |1| | x[4] 〉 < y[4] #]
        >>> prob.get_constraint(5)                              #6th constraint
        # (5x1)-affine constraint: y > |0| #

        """
        indtuple = ind
        if isinstance(indtuple, int):
            return self.constraints[indtuple]
        lsind = self.consNumbering
        if not(isinstance(indtuple, tuple) or isinstance(indtuple, list)) or (
                len(indtuple) == 0):
            raise Exception('ind must be an int or a nonempty tuple')

        for k in indtuple:
            if not isinstance(lsind, list):
                if k == 0:
                    break
                else:
                    raise Exception('too many indices')
            if k >= len(lsind):
                raise Exception('index is too large')
            lsind = lsind[k]

        if isinstance(lsind, list):
                # flatten for the case where it is still a list of list
            return [self.constraints[i] for i in _flatten(lsind)]
        return self.constraints[lsind]

    def remove_constraint(self, ind):
        """
        Deletes a constraint or a list of constraints of the problem.

        :param ind: The indexing of constraints works as in the
                    function :func:`get_constraint() <picos.Problem.get_constraint>`:

                        * if ``ind`` is an integer :math:`n`, the nth constraint
                          (numbered from 0) is deleted

                        * if ``ind`` is a *tuple* :math:`(k,i)`, then the ith constraint
                          from the kth group of constraints is deleted
                          (starting from 0). By
                          *group of constraints*, it is meant a single constraint
                          or a list of constraints added together with the
                          function :func:`add_list_of_constraints() <picos.Problem.add_list_of_constraints>`.

                        * if ``ind`` is a tuple of length 1 :math:`(k,)`,
                          then the whole kth group of constraints is deleted.

        :type ind: int or tuple.

        **Example:**

        >>> import picos as pic
        >>> import cvxopt as cvx
        >>> prob=pic.Problem()
        >>> x=[prob.add_variable('x[{0}]'.format(i),2) for i in range(4)]
        >>> y=prob.add_variable('y',4)
        >>> prob.add_list_of_constraints(
        ... [(1|x[i])<y[i] for i in range(4)], 'i', '[5]')
        >>> prob.add_constraint(y>0)
        >>> prob.add_list_of_constraints(
        ... [x[i]<2 for i in range(3)], 'i', '[3]')
        >>> prob.add_constraint(x[3]<1)
        >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
        [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
         # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #,
         # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
         # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
         # (4x1)-affine constraint: y > |0| #,
         # (2x1)-affine constraint: x[0] < |2.0| #,
         # (2x1)-affine constraint: x[1] < |2.0| #,
         # (2x1)-affine constraint: x[2] < |2.0| #,
         # (2x1)-affine constraint: x[3] < |1| #]
        >>> prob.remove_constraint(1)                           #2d constraint (numbered from 0) deleted
        >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
        [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
         # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
         # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
         # (4x1)-affine constraint: y > |0| #,
         # (2x1)-affine constraint: x[0] < |2.0| #,
         # (2x1)-affine constraint: x[1] < |2.0| #,
         # (2x1)-affine constraint: x[2] < |2.0| #,
         # (2x1)-affine constraint: x[3] < |1| #]
        >>> prob.remove_constraint((1,))                        #2d 'group' of constraint deleted, i.e. the single constraint y>|0|
        >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
        [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
         # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
         # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
         # (2x1)-affine constraint: x[0] < |2.0| #,
         # (2x1)-affine constraint: x[1] < |2.0| #,
         # (2x1)-affine constraint: x[2] < |2.0| #,
         # (2x1)-affine constraint: x[3] < |1| #]
        >>> prob.remove_constraint((2,))                        #3d 'group' of constraint deleted, (originally the 4th group, i.e. x[3]<|1|)
        >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
        [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
         # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
         # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
         # (2x1)-affine constraint: x[0] < |2.0| #,
         # (2x1)-affine constraint: x[1] < |2.0| #,
         # (2x1)-affine constraint: x[2] < |2.0| #]
        >>> prob.remove_constraint((1,1))                       #2d constraint of the 2d group (originally the 3rd group), i.e. x[1]<|2|
        >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
        [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
         # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
         # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
         # (2x1)-affine constraint: x[0] < |2.0| #,
         # (2x1)-affine constraint: x[2] < |2.0| #]


        """
        # TODO    *examples with list of geomeans

        if isinstance(ind, int):  # constraint given with its "raw index"
            cons = self.constraints[ind]
            if cons.typeOfConstraint[:3] == 'lin':
                self.numberAffConstraints -= (
                    cons.Exp1.size[0] * cons.Exp1.size[1])
            elif cons.typeOfConstraint[2:] == 'cone':
                self.numberConeVars -= (
                    (cons.Exp1.size[0] * cons.Exp1.size[1]) + 1)
                self.numberConeConstraints -= 1
                if cons.typeOfConstraint[:2] == 'RS':
                    self.numberConeVars -= 1
            elif cons.typeOfConstraint == 'lse':
                self.numberLSEVars -= (cons.Exp1.size[0] * cons.Exp1.size[1])
                self.numberLSEConstraints -= 1
            elif cons.typeOfConstraint == 'quad':
                self.numberQuadConstraints -= 1
                self.numberQuadNNZ -= cons.Exp1.nnz()
            elif cons.typeOfConstraint[:3] == 'sdp':
                self.numberSDPConstraints -= 1
                self.numberSDPVars -= (cons.Exp1.size[0]
                                       * (cons.Exp1.size[0] + 1)) // 2
                if cons.semidefVar:
                    cons.semidefVar.semiDef = False

            cons.original_index = ind
            not_passed_yet = [solver for solver in ('mosek','cplex','gurobi','cvxopt','glpk','scip','sdpa')
                              if solver not in cons.passed]
            cons.passed = not_passed_yet
            #deleted constraint is considered as 'passed', i.e. it can be ignored, if it
            #was not yet part of the solver instance

            self._deleted_constraints.append(cons)
            del self.constraints[ind]
            self.countCons -= 1
            if ind in self.consNumbering:  # single added constraint
                self.consNumbering.remove(ind)
                start = ind
                self.consNumbering = offset_in_lil(self.consNumbering, 1, ind)
            else:  # a constraint within a group of constraints
                for i, l in enumerate(self.consNumbering):
                    if ind in _flatten([l]):
                        l0 = l[0]
                        while isinstance(l0, list):
                            l0 = l0[0]
                        start = l0
                        _remove_in_lil(self.consNumbering, ind)

                self.consNumbering = offset_in_lil(self.consNumbering, 1, ind)
                goc = self.groupsOfConstraints[start]
                self.groupsOfConstraints[start] = [goc[0] - 1,
                                                   goc[1][:-1] + '{-1cons}\n',
                                                   goc[2]]
                if goc[0] == start:
                    del self.groupsOfConstraints[start]
            # offset in subsequent goc
            for stidx in self.groupsOfConstraints:
                if stidx > start:
                    goc = self.groupsOfConstraints[stidx]
                    del self.groupsOfConstraints[stidx]
                    goc[0] = goc[0] - 1
                    self.groupsOfConstraints[stidx - 1] = goc

            print
            return

        indtuple = ind
        lsind = self.consNumbering
        for k in indtuple:
            if not isinstance(lsind, list):
                if k == 0:
                    break
                else:
                    raise Exception('too many indices')
            if k >= len(lsind):
                raise Exception('index is too large')
            lsind = lsind[k]
        # now, lsind must be the index or list of indices to remove
        if isinstance(lsind, list):  # a list of constraints
            # we flatten lsind for the case where it is still a list of lists
            lsind_top = lsind
            lsind = list(_flatten(lsind))

            for ind in reversed(lsind):
                cons = self.constraints[ind]
                if cons.typeOfConstraint[:3] == 'lin':
                    self.numberAffConstraints -= (
                        cons.Exp1.size[0] * cons.Exp1.size[1])
                elif cons.typeOfConstraint[2:] == 'cone':
                    self.numberConeVars -= (
                        (cons.Exp1.size[0] * cons.Exp1.size[1]) + 1)
                    self.numberConeConstraints -= 1
                    if cons.typeOfConstraint[:2] == 'RS':
                        self.numberConeVars -= 1
                elif cons.typeOfConstraint == 'lse':
                    self.numberLSEVars -= (cons.Exp1.size[0]
                                           * cons.Exp1.size[1])
                    self.numberLSEConstraints -= 1
                elif cons.typeOfConstraint == 'quad':
                    self.numberQuadConstraints -= 1
                    self.numberQuadNNZ -= cons.Exp1.nnz()
                elif cons.typeOfConstraint[:3] == 'sdp':
                    self.numberSDPConstraints -= 1
                    self.numberSDPVars -= (cons.Exp1.size[0]
                                           * (cons.Exp1.size[0] + 1)) // 2

                    if cons.semidefVar:
                        cons.semidefVar.semiDef = False
                del self.constraints[ind]
            self.countCons -= len(lsind)
            _remove_in_lil(self.consNumbering, lsind_top)
            self.consNumbering = offset_in_lil(
                self.consNumbering, len(lsind), lsind[0])
            # update this group of constraints
            for start, goc in six.iteritems(self.groupsOfConstraints):
                if lsind[0] >= start and lsind[0] <= goc[0]:
                    break

            self.groupsOfConstraints[start] = [
                goc[0] -
                len(lsind),
                goc[1][
                    :-
                    1] +
                '{-%dcons}\n' %
                len(lsind),
                goc[2]]
            if self.groupsOfConstraints[start][0] < start:
                del self.groupsOfConstraints[start]
            # offset in subsequent goc
            oldkeys = self.groupsOfConstraints.keys()
            for stidx in oldkeys:
                if stidx > start:
                    goc = self.groupsOfConstraints[stidx]
                    del self.groupsOfConstraints[stidx]
                    goc[0] = goc[0] - len(lsind)
                    self.groupsOfConstraints[stidx - len(lsind)] = goc
        elif isinstance(lsind, int):
            self.remove_constraint(lsind)

        self._eliminate_useless_variables()

    def _eval_all(self):
        """
        Returns the big vector with all variable values,
        in the order induced by sorted(self.variables.keys()).
        """
        xx = cvx.matrix([], (0, 1))
        for v in sorted(self.variables.keys()):
            xx = cvx.matrix([xx, self.variables[v].value[:]])
        return xx

    def check_current_value_feasibility(self, tol=1e-5):
        """
        returns ``True`` if the
        current value of the variabless
        is a feasible solution, up to the
        tolerance ``tol``. If ``tol`` is set to ``None``,
        the option parameter ``options['tol']`` is used instead.
        The integer feasibility is checked with a tolerance of 1e-3.
        """
        if tol is None:
            if not(self.options['feastol'] is None):
                tol = self.options['feastol']
            else:
                tol = self.options['tol']
        for cs in self.constraints:
            sl = cs.slack
            if not(isinstance(sl, cvx.matrix) or isinstance(sl, cvx.spmatrix)):
                sl = cvx.matrix(sl)
            if cs.typeOfConstraint.startswith('sdp'):
                # check symmetry
                if min(sl - sl.T) < -tol:
                    return (False, -min(sl - sl.T))
                if min(sl.T - sl) < -tol:
                    return (False, -min(sl.T - sl))
                # check positive semidefiniteness
                if isinstance(sl, cvx.spmatrix):
                    sl = cvx.matrix(sl)
                sl = np.array(sl)
                eg = np.linalg.eigvalsh(sl)
                if min(eg) < -tol:
                    return (False, -min(eg))
            else:
                if min(sl) < -tol:
                    return (False, -min(sl))
        # integer feasibility
        if not(self.is_continuous()):
            for vnam, v in six.iteritems(self.variables):
                if v.vtype in ('binary', 'integer'):
                    sl = v.value
                    dsl = [min(s - int(s), int(s) + 1 - s) for s in sl]
                    if max(dsl) > 1e-3:
                        return (False, max(dsl))

        # so OK, it's feasible
        return (True, None)

    """
        ----------------------------------------------------------------
        --                BUILD THE VARIABLES FOR A SOLVER            --
        ----------------------------------------------------------------
        """

    # GUROBI
    def _make_gurobi_instance(self):
        """
        defines the variables gurobi_Instance and grbvar
        """

        '''
        #this is not needed anymore, because we handle deleted constraints dynamically

        if any([('gurobi' not in cs.passed) for cs in self._deleted_constraints]):
            for cs in self._deleted_constraints:
                if 'gurobi' not in cs.passed:
                    cs.passed.append('gurobi')
            self.reset_gurobi_instance(True)
        '''

        try:
            import gurobipy as grb
        except:
            raise ImportError('gurobipy not found')

        grb_type = {'continuous': grb.GRB.CONTINUOUS,
                    'binary': grb.GRB.BINARY,
                    'integer': grb.GRB.INTEGER,
                    'semicont': grb.GRB.SEMICONT,
                    'semiint': grb.GRB.SEMIINT,
                    'symmetric': grb.GRB.CONTINUOUS}

        if (self.gurobi_Instance is None):
            m = grb.Model()
            boundcons = []
            grbcons = {}
        else:
            m = self.gurobi_Instance
            boundcons = self.grb_boundcons
            grbcons = self.grbcons

        if self.objective[0] == 'max':
            m.ModelSense = grb.GRB.MAXIMIZE
        else:
            m.ModelSense = grb.GRB.MINIMIZE

        self.options._set('solver', 'gurobi')

        grb_infty = 1e20

        # create new variable and quad constraints to handle socp
        tmplhs = []
        tmprhs = []
        icone = 0

        NUMVAR_OLD = m.numVars  # total number of vars before
        NUMVAR0_OLD = int(_bsum([(var.endIndex - var.startIndex)  # old number of vars without cone vars
                                 for var in self.variables.values()
                                 if ('gurobi' in var.passed)]))
        # number of conevars already there.
        OFFSET_CV = NUMVAR_OLD - NUMVAR0_OLD
        NUMVAR0_NEW = int(_bsum([(var.endIndex - var.startIndex)  # new vars without new cone vars
                                 for var in self.variables.values()
                                 if not('gurobi' in var.passed)]))

        newcons = {}
        newvars = []
        posvars = []

        if self.numberConeConstraints > 0:
            for constrKey, constr in enumerate(self.constraints):
                if 'gurobi' in constr.passed:
                    continue
                if constr.typeOfConstraint[2:] == 'cone':
                    if icone == 0:  # first conic constraint
                        if '__noconstant__' in self.variables:
                            noconstant = self.get_variable('__noconstant__')
                        else:
                            noconstant = self.add_variable(
                                '__noconstant__', 1)
                            newvars.append(('__noconstant__', 1))
                            posvars.append(noconstant.startIndex)
                        # no variable shift -> same noconstant var as before
                if constr.typeOfConstraint == 'SOcone':
                    if '__tmplhs[{0}]__'.format(constrKey) in self.variables:
                        # remove_variable should never called (we let it for
                        # security)
                        # constrKey replaced the icone+offset_cone of previous
                        # version
                        self.remove_variable(
                            '__tmplhs[{0}]__'.format(constrKey))
                    if '__tmprhs[{0}]__'.format(constrKey) in self.variables:
                        self.remove_variable(
                            '__tmprhs[{0}]__'.format(constrKey))
                    tmplhs.append(self.add_variable(
                        '__tmplhs[{0}]__'.format(constrKey),
                        constr.Exp1.size))
                    tmprhs.append(self.add_variable(
                        '__tmprhs[{0}]__'.format(constrKey),
                        1))
                    newvars.append(('__tmplhs[{0}]__'.format(constrKey),
                                    constr.Exp1.size[0] * constr.Exp1.size[1]))
                    newvars.append(('__tmprhs[{0}]__'.format(constrKey),
                                    1))
                    # v_cons is 0/1/-1 to avoid constants in cone (problem with
                    # duals)
                    v_cons = cvx.matrix([np.sign(constr.Exp1.constant[i])
                                         if constr.Exp1[i].isconstant() else 0
                                         for i in range(constr.Exp1.size[0] * constr.Exp1.size[1])],
                                        constr.Exp1.size)
                    # lhs and rhs of the cone constraint
                    newcons['tmp_lhs_{0}'.format(constrKey)] = (
                        constr.Exp1 + v_cons * noconstant == tmplhs[-1])
                    newcons['tmp_rhs_{0}'.format(constrKey)] = (
                        constr.Exp2 - noconstant == tmprhs[-1])
                    # conic constraints
                    posvars.append(tmprhs[-1].startIndex)
                    newcons['tmp_conequad_{0}'.format(constrKey)] = (
                        -tmprhs[-1]**2 + (tmplhs[-1] | tmplhs[-1]) < 0)

                    if constr.Id is None:
                        constr.Id = {}
                    constr.Id.setdefault('gurobi',[])
                    for i in range(constr.Exp1.size[0] * constr.Exp1.size[1]):
                        constr.Id['gurobi'].append('lintmp_lhs_{0}_{1}'.format(constrKey,i))
                    for i in range(constr.Exp2.size[0] * constr.Exp2.size[1]):
                        constr.Id['gurobi'].append('lintmp_rhs_{0}_{1}'.format(constrKey,i))
                    constr.Id['gurobi'].append('tmp_conequad_{0}'.format(constrKey))
                    cs = newcons['tmp_conequad_{0}'.format(constrKey)]
                    cs.myconstring = 'tmp_conequad_{0}'.format(constrKey)

                    icone += 1

                if constr.typeOfConstraint == 'RScone':
                    if '__tmplhs[{0}]__'.format(constrKey) in self.variables:
                        self.remove_variable(
                            '__tmplhs[{0}]__'.format(constrKey))
                    if '__tmprhs[{0}]__'.format(constrKey) in self.variables:
                        self.remove_variable(
                            '__tmprhs[{0}]__'.format(constrKey))
                    tmplhs.append(self.add_variable(
                        '__tmplhs[{0}]__'.format(constrKey),
                        (constr.Exp1.size[0] * constr.Exp1.size[1]) + 1
                    ))
                    tmprhs.append(self.add_variable(
                        '__tmprhs[{0}]__'.format(constrKey),
                        1))
                    newvars.append(
                        ('__tmplhs[{0}]__'.format(constrKey),
                         (constr.Exp1.size[0] * constr.Exp1.size[1]) + 1))
                    newvars.append(('__tmprhs[{0}]__'.format(constrKey),
                                    1))
                    # v_cons is 0/1/-1 to avoid constants in cone (problem with
                    # duals)
                    expcat = ((2 * constr.Exp1[:]) //
                              (constr.Exp2 - constr.Exp3))
                    v_cons = cvx.matrix([np.sign(expcat.constant[i])
                                         if expcat[i].isconstant() else 0
                                         for i in range(expcat.size[0] * expcat.size[1])],
                                        expcat.size)

                    # lhs and rhs of the cone constraint
                    newcons['tmp_lhs_{0}'.format(constrKey)] = (
                        (2 * constr.Exp1[:] // (constr.Exp2 - constr.Exp3)) + v_cons * noconstant == tmplhs[-1])
                    newcons['tmp_rhs_{0}'.format(constrKey)] = (
                        constr.Exp2 + constr.Exp3 - noconstant == tmprhs[-1])
                    # conic constraints
                    posvars.append(tmprhs[-1].startIndex)
                    newcons['tmp_conequad_{0}'.format(constrKey)] = (
                        -tmprhs[-1]**2 + (tmplhs[-1] | tmplhs[-1]) < 0)

                    if constr.Id is None:
                        constr.Id = {}
                    constr.Id.setdefault('gurobi',[])
                    for i in range(constr.Exp1.size[0] * constr.Exp1.size[1] + 1):
                        constr.Id['gurobi'].append('lintmp_lhs_{0}_{1}'.format(constrKey,i))
                    constr.Id['gurobi'].append('lintmp_rhs_{0}_{1}'.format(constrKey,0))
                    constr.Id['gurobi'].append('tmp_conequad_{0}'.format(constrKey))
                    cs = newcons['tmp_conequad_{0}'.format(constrKey)]
                    cs.myconstring = 'tmp_conequad_{0}'.format(constrKey)

                    icone += 1

        NUMVAR_NEW = int(_bsum([(var.endIndex - var.startIndex)  # new vars including cone vars
                                for var in self.variables.values()
                                if not('gurobi' in var.passed)]))

        # total number of variables (including extra vars for cones)
        NUMVAR = NUMVAR_OLD + NUMVAR_NEW

        # variables

        if (self.options['verbose'] > 1) and NUMVAR_NEW > 0:
            limitbar = NUMVAR_NEW
            prog = ProgressBar(0, limitbar, None, mode='fixed')
            oldprog = str(prog)
            print('Creating variables...')
            print()

        x = []  # list of new vars

        # TODO pb if bounds of old variables changed
        if NUMVAR_NEW:

            ub = dict((j, grb.GRB.INFINITY) for j in range(NUMVAR_OLD, NUMVAR))
            lb = dict((j, -grb.GRB.INFINITY)
                      for j in range(NUMVAR_OLD, NUMVAR))

            for kvar, variable in [(kvar, variable) for (kvar, variable)
                                   in six.iteritems(self.variables)
                                   if 'gurobi' not in variable.passed]:

                variable.gurobi_startIndex = variable.startIndex + OFFSET_CV
                variable.gurobi_endIndex = variable.endIndex + OFFSET_CV
                sj = variable.gurobi_startIndex
                ej = variable.gurobi_endIndex

                for ind, (lo, up) in six.iteritems(variable.bnd):
                    if not(lo is None):
                        lb[sj + ind] = lo
                    if not(up is None):
                        ub[sj + ind] = up

            for i in posvars: #bounds for dummy cone variables
                lb[i]=0.

            vartopass = sorted([(variable.gurobi_startIndex, variable) for (kvar, variable)
                                in six.iteritems(self.variables)
                                if 'gurobi' not in variable.passed])

            for (vcsi, variable) in vartopass:
                variable.passed.append('gurobi')
                sj = variable.gurobi_startIndex
                tp = variable.vtype
                varsize = variable.endIndex - variable.startIndex

                for k in range(varsize):
                    name = variable.name + '_' + str(k)
                    x.append(m.addVar(obj=0,
                                      name=name,
                                      vtype=grb_type[tp],
                                      lb=lb[sj + k],
                                      ub=ub[sj + k]))

                    if self.options['verbose'] > 1:
                        #<--display progress
                        prog.increment_amount()
                        if oldprog != str(prog):
                            print(prog, "\r", end="")
                            sys.stdout.flush()
                            oldprog = str(prog)
                        #-->

            if self.options['verbose'] > 1:
                prog.update_amount(limitbar)
                print(prog, "\r", end="")
                print()

        m.update()
        # parse all vars for hotstart
        if self.options['hotstart']:
            for kvar, variable in six.iteritems(self.variables):
                if variable.is_valued():
                    vstart = variable.value
                    varsize = variable.endIndex - variable.startIndex
                    for k in range(varsize):
                        name = kvar + '_' + str(k)
                        xj = m.getVarByName(name)
                        xj.Start = vstart[k]
        m.update()

        # parse all variable for the objective (only if not obj_passed)
        if 'gurobi' not in self.obj_passed:
            self.obj_passed.append('gurobi')
            if self.objective[1] is None:
                objective = {}
            elif isinstance(self.objective[1], QuadExp):
                objective = self.objective[1].aff.factors
            elif isinstance(self.objective[1], AffinExp):
                objective = self.objective[1].factors

            m.setObjective(0)
            m.update()

            for variable, vect in six.iteritems(objective):
                varsize = variable.endIndex - variable.startIndex
                for (k, v) in zip(vect.J, vect.V):
                    name = variable.name + '_' + str(k)
                    xj = m.getVarByName(name)
                    xj.obj = v

            m.update()

            # quad part of the objective
            if isinstance(self.objective[1], QuadExp):
                lpart = m.getObjective()
                qd = self.objective[1].quad
                qind1, qind2, qval = [], [], []
                for i, j in qd:
                    fact = qd[i, j]
                    namei = i.name
                    namej = j.name
                    si = i.startIndex
                    sj = j.startIndex
                    if (j, i) in qd:  # quad stores x'*A1*y + y'*A2*x
                        if si < sj:
                            fact += qd[j, i].T
                        elif si > sj:
                            fact = cvx.sparse([0])
                        elif si == sj:
                            pass
                    qind1.extend([namei + '_' + str(k) for k in fact.I])
                    qind2.extend([namej + '_' + str(k) for k in fact.J])
                    qval.extend(fact.V)
                q_exp = grb.quicksum([f * m.getVarByName(n1) * m.getVarByName(n2)
                                      for (f, n1, n2) in zip(qval, qind1, qind2)])
                m.setObjective(q_exp + lpart)
                m.update()

        # constraints

        NUMCON_NEW = int(_bsum([(cs.Exp1.size[0] * cs.Exp1.size[1])
                                for cs in self.constraints
                                if (cs.typeOfConstraint.startswith('lin'))
                                and not('gurobi' in cs.passed)] +
                               [1 for cs in self.constraints
                                if (cs.typeOfConstraint == 'quad')
                                and not('gurobi' in cs.passed)]
                               )
                         )

        # progress bar
        if self.options['verbose'] > 0:
            print()
            print('adding constraints...')
            print()
        if self.options['verbose'] > 1:
            limitbar = NUMCON_NEW
            prog = ProgressBar(0, limitbar, None, mode='fixed')
            oldprog = str(prog)

        # join all constraints
        def join_iter(it1, it2):
            for i in it1:
                yield i
            for i in it2:
                yield i

        allcons = join_iter(enumerate(self.constraints),
                            six.iteritems(newcons))

        irow = 0
        qind = m.NumQConstrs

        for constrKey, constr in allcons:

            if 'gurobi' in constr.passed:
                continue
            else:
                constr.passed.append('gurobi')

            # init of boundcons[key]
            if isinstance(constrKey,int):
                boundcons.append([])

            if constr.typeOfConstraint[:3] == 'lin':

                # parse the (i,j,v) triple
                ijv = []
                for var, fact in six.iteritems(
                        (constr.Exp1 - constr.Exp2).factors):
                    if not isinstance(fact, cvx.base.spmatrix):
                        fact = cvx.sparse(fact)
                    ijv.extend(zip(fact.I,
                                   [var.name + '_' + str(j) for j in fact.J],
                                   fact.V))
                ijvs = sorted(ijv)

                itojv = {}
                lasti = -1
                for (i, j, v) in ijvs:
                    if i == lasti:
                        itojv[i].append((j, v))
                    else:
                        lasti = i
                        itojv[i] = [(j, v)]

                # constant term
                szcons = constr.Exp1.size[0] * constr.Exp1.size[1]
                rhstmp = cvx.matrix(0., (szcons, 1))
                constant1 = constr.Exp1.constant  # None or a 1*1 matrix
                constant2 = constr.Exp2.constant
                if not constant1 is None:
                    rhstmp = rhstmp - constant1
                if not constant2 is None:
                    rhstmp = rhstmp + constant2

                # constraint of the form 0*x==a
                if len(itojv) != szcons:
                    zero_rows = (set(range(szcons)) - set(itojv.keys()))
                    for i in zero_rows:
                        if ((constr.typeOfConstraint[:4] == 'lin<' and rhstmp[i] < 0) or
                                (constr.typeOfConstraint[:4] == 'lin>' and rhstmp[i] > 0) or
                                (constr.typeOfConstraint[:4] == 'lin=' and rhstmp[i] != 0)):
                            raise Exception(
                                'you try to add a constraint of the form 0 * x == 1')

                    constr.zero_rows = list(zero_rows)

                for i, jv in six.iteritems(itojv):
                    r = rhstmp[i]
                    if len(jv) == 1 and self.options['pass_simple_cons_as_bound']:
                        # BOUND
                        name, v = jv[0]
                        xj = m.getVarByName(name)
                        b = r / float(v)
                        if v > 0:
                            if constr.typeOfConstraint[:4] in ['lin<', 'lin=']:
                                if b < xj.ub:
                                    xj.ub = b
                            if constr.typeOfConstraint[:4] in ['lin>', 'lin=']:
                                if b > xj.lb:
                                    xj.lb = b
                        else:  # v<0
                            if constr.typeOfConstraint[:4] in ['lin<', 'lin=']:
                                if b > xj.lb:
                                    xj.lb = b
                            if constr.typeOfConstraint[:4] in ['lin>', 'lin=']:
                                if b < xj.ub:
                                    xj.ub = b
                        if constr.typeOfConstraint[3] == '=':
                            b = '='
                        if isinstance(constrKey,int):
                            boundcons[constrKey].append((i, name, b, v))
                    else:
                        LEXP = grb.LinExpr(
                            [v for j, v in jv],
                            [m.getVarByName(name) for name, v in jv])
                        name = 'lin' + str(constrKey) + '_' + str(i)
                        if constr.typeOfConstraint[:4] == 'lin<':
                            grbcons[name] = m.addConstr(LEXP <= r, name=name)
                        elif constr.typeOfConstraint[:4] == 'lin>':
                            grbcons[name] = m.addConstr(LEXP >= r, name=name)
                        elif constr.typeOfConstraint[:4] == 'lin=':
                            grbcons[name] = m.addConstr(LEXP == r, name=name)
                        if constr.Id is None:
                            constr.Id = {}
                        constr.Id.setdefault('gurobi',[])
                        constr.Id['gurobi'].append(name)

                        irow += 1

                    if self.options['verbose'] > 1:
                        #<--display progress
                        prog.increment_amount()
                        if oldprog != str(prog):
                            print(prog, "\r", end="")
                            sys.stdout.flush()
                            oldprog = str(prog)
                        #-->

            elif constr.typeOfConstraint == 'quad':
                # quad part
                qind1, qind2, qval = [], [], []
                qd = constr.Exp1.quad
                q_exp = 0.
                for i, j in qd:
                    fact = qd[i, j]
                    namei = i.name
                    namej = j.name
                    si = i.startIndex
                    sj = j.startIndex
                    if (j, i) in qd:  # quad stores x'*A1*y + y'*A2*x
                        if si < sj:
                            fact += qd[j, i].T
                        elif si > sj:
                            fact = cvx.sparse([0])
                        elif si == sj:
                            pass
                    qind1.extend([namei + '_' + str(k) for k in fact.I])
                    qind2.extend([namej + '_' + str(k) for k in fact.J])
                    qval.extend(fact.V)
                q_exp = grb.quicksum([f * m.getVarByName(n1) * m.getVarByName(n2)
                                      for (f, n1, n2) in zip(qval, qind1, qind2)])
                # lin part
                lind, lval = [], []
                af = constr.Exp1.aff.factors
                for var in af:
                    name = var.name
                    lind.extend([name + '_' + str(k) for k in af[var].J])
                    lval.extend(af[var].V)
                l_exp = grb.LinExpr(
                    lval,
                    [m.getVarByName(name) for name in lind])
                # constant
                qcs = 0.
                if not(constr.Exp1.aff.constant is None):
                    qcs = - constr.Exp1.aff.constant[0]

                #name
                if constr.myconstring is not None and 'tmp_conequad' in constr.myconstring:
                    qname = constr.myconstring
                    qind+=1
                else:
                    qname = 'q'+str(qind)
                    qind+=1

                grbcons[qname] = m.addQConstr(q_exp + l_exp <= qcs, qname)

                if self.options['verbose'] > 1:
                    #<--display progress
                    prog.increment_amount()
                    if oldprog != str(prog):
                        print(prog, "\r", end="")
                        sys.stdout.flush()
                        oldprog = str(prog)
                    #-->

            elif constr.typeOfConstraint[2:] == 'cone':
                pass
                # will be handled in the newcons dictionary

            else:
                raise Exception(
                    'type of constraint not handled (yet ?) for gurobi:{0}'.format(
                        constr.typeOfConstraint))

        if self.options['verbose'] > 1:
            prog.update_amount(limitbar)
            print(prog, "\r", end="")
            print()

        m.update()

        #constraints deletion
        print_message_not_printed_yet = True
        warning_message_not_printed_yet = True
        todel_from_boundcons = []
        for cs in self._deleted_constraints:
            if 'gurobi' in cs.passed:
                continue
            else:
                cs.passed.append('gurobi')

            if print_message_not_printed_yet and self.options['verbose'] > 0:
                print()
                print('Removing constraints from Gurobi instance...')
                print_message_not_printed_yet = False

            sgn = cs.typeOfConstraint[3]
            todel_from_boundcons.append(cs.original_index)
            if (self.options['verbose'] > 0 and
                self.options['pass_simple_cons_as_bound'] and
                warning_message_not_printed_yet and
                self.grb_boundcons[cs.original_index]
                ):

                print("\033[1;31m*** You have been removing a constraint that can be "+
                      "(partly) interpreted as a variable bound. This is not safe when "
                      "the option ``pass_simple_cons_as_bound`` is set to True\033[0m")

                warning_message_not_printed_yet = False

            for i,name,b,v in self.grb_boundcons[cs.original_index]:
                xj = m.getVarByName(name)
                if sgn == '=':
                    xj.lb = -grb_infty
                    xj.ub = grb_infty
                elif (sgn == '<' and v>0) or (sgn == '>' and v<0):
                    xj.ub = grb_infty
                elif (sgn == '<' and v>0) or (sgn == '>' and v<0):
                    xj.lb = -grb_infty

            for i in cs.Id['gurobi']:
                m.remove(self.grbcons[i])
                if 'cone' in cs.typeOfConstraint and not(i.startswith('tmp_conequad')):
                    ind,jj = [int(kk) for kk in i.split('hs_',1)[1].split('_')]
                    varname = '__tmp'+i.split('hs_')[0][-1]+'hs[{0}]___{1}'.format(ind,jj)
                    m.remove(m.getVarByName(varname))

        for i in todel_from_boundcons:
            del boundcons[i]

        m.update()

        self.gurobi_Instance = m
        self.grbvar.extend(x)
        self.grb_boundcons = boundcons
        self.grbcons = grbcons

        if 'noconstant' in newcons or len(tmplhs) > 0:
            self._remove_temporary_variables()

        if self.options['verbose'] > 0:
            print('Gurobi instance created')
            print()

    def is_continuous(self):
        """ Returns ``True`` if there are only continuous variables"""
        for kvar in self.variables.keys():
            if self.variables[kvar].vtype not in [
                    'continuous', 'symmetric', 'hermitian', 'complex']:
                return False
        return True

    def is_complex(self):
        tps = [x.vtype for x in self.variables.values()]
        if 'hermitian' in tps or 'complex' in tps:
            return True
        else:
            return self._complex

    def _make_cplex_instance(self):
        """
        Defines the variables cplex_Instance and cplexvar,
        used by the cplex solver.
        """
        try:
            import cplex
        except:
            raise ImportError('cplex library not found')

        import itertools

        if (self.cplex_Instance is None):
            c = cplex.Cplex()
            boundcons = []

        else:
            c = self.cplex_Instance
            boundcons = self.cplex_boundcons #stores index of constraints interpreted as a bound

        sense_opt = self.objective[0]
        if sense_opt == 'max':
            c.objective.set_sense(c.objective.sense.maximize)
        elif sense_opt == 'min':
            c.objective.set_sense(c.objective.sense.minimize)

        self.set_option('solver', 'cplex')

        cplex_type = {'continuous': c.variables.type.continuous,
                      'binary': c.variables.type.binary,
                      'integer': c.variables.type.integer,
                      'semicont': c.variables.type.semi_continuous,
                      'semiint': c.variables.type.semi_integer,
                      'symmetric': c.variables.type.continuous}

        # create new variables and quad constraints to handle socp
        tmplhs = []
        tmprhs = []
        icone = 0

        NUMVAR_OLD = c.variables.get_num()  # total number of vars before
        NUMVAR0_OLD = int(_bsum([(var.endIndex - var.startIndex)  # old number of vars without cone vars
                                 for var in self.variables.values()
                                 if ('cplex' in var.passed)]))
        # number of conevars already there.
        OFFSET_CV = NUMVAR_OLD - NUMVAR0_OLD
        NUMVAR0_NEW = int(_bsum([(var.endIndex - var.startIndex)  # new vars without new cone vars
                                 for var in self.variables.values()
                                 if not('cplex' in var.passed)]))

        newcons = {}
        newvars = []

        posvars = []

        if self.numberConeConstraints > 0:
            for constrKey, constr in enumerate(self.constraints):
                if 'cplex' in constr.passed:
                    continue
                if constr.typeOfConstraint[2:] == 'cone':
                    if icone == 0:  # first conic constraint
                        if '__noconstant__' in self.variables:
                            noconstant = self.get_variable('__noconstant__')
                        else:
                            noconstant = self.add_variable(
                                '__noconstant__', 1)
                            newvars.append(('__noconstant__', 1))
                            posvars.append(noconstant.startIndex)

                        # no variable shift -> same noconstant var as before
                if constr.typeOfConstraint == 'SOcone':
                    if '__tmplhs[{0}]__'.format(constrKey) in self.variables:
                        # remove_variable should never called (we let it for
                        # security)
                        self.remove_variable(
                            '__tmplhs[{0}]__'.format(constrKey))
                    if '__tmprhs[{0}]__'.format(constrKey) in self.variables:
                        self.remove_variable(
                            '__tmprhs[{0}]__'.format(constrKey))
                    tmplhs.append(self.add_variable(
                        '__tmplhs[{0}]__'.format(constrKey),
                        constr.Exp1.size))
                    tmprhs.append(self.add_variable(
                        '__tmprhs[{0}]__'.format(constrKey),
                        1))
                    newvars.append(('__tmplhs[{0}]__'.format(constrKey),
                                    constr.Exp1.size[0] * constr.Exp1.size[1]))
                    newvars.append(('__tmprhs[{0}]__'.format(constrKey),
                                    1))
                    # v_cons is 0/1/-1 to avoid constants in cone (problem with
                    # duals)
                    v_cons = cvx.matrix(
                        [
                            np.sign(
                                constr.Exp1.constant[i]) if (
                                constr.Exp1.constant is not None) and constr.Exp1[i].isconstant() else 0 for i in range(
                                constr.Exp1.size[0] *
                                constr.Exp1.size[1])],
                        constr.Exp1.size)
                    # lhs and rhs of the cone constraint
                    newcons['tmp_lhs_{0}'.format(constrKey)] = (
                        constr.Exp1 + v_cons * noconstant == tmplhs[-1])
                    newcons['tmp_rhs_{0}'.format(constrKey)] = (
                        constr.Exp2 - noconstant == tmprhs[-1])
                    posvars.append(tmprhs[-1].startIndex)
                    # conic constraints
                    newcons['tmp_conequad_{0}'.format(constrKey)] = (
                        -tmprhs[-1]**2 + (tmplhs[-1] | tmplhs[-1]) < 0)

                    if constr.Id is None:
                        constr.Id = {}
                    constr.Id.setdefault('cplex',[])
                    for i in range(constr.Exp1.size[0] * constr.Exp1.size[1]):
                        constr.Id['cplex'].append('lintmp_lhs_{0}_{1}'.format(constrKey,i))
                    for i in range(constr.Exp2.size[0] * constr.Exp2.size[1]):
                        constr.Id['cplex'].append('lintmp_rhs_{0}_{1}'.format(constrKey,i))
                    constr.Id['cplex'].append('tmp_conequad_{0}'.format(constrKey))
                    cs = newcons['tmp_conequad_{0}'.format(constrKey)]
                    cs.myconstring = 'tmp_conequad_{0}'.format(constrKey)

                    icone += 1

                if constr.typeOfConstraint == 'RScone':
                    if '__tmplhs[{0}]__'.format(constrKey) in self.variables:
                        self.remove_variable(
                            '__tmplhs[{0}]__'.format(constrKey))
                    if '__tmprhs[{0}]__'.format(constrKey) in self.variables:
                        self.remove_variable(
                            '__tmprhs[{0}]__'.format(constrKey))
                    tmplhs.append(self.add_variable(
                        '__tmplhs[{0}]__'.format(constrKey),
                        (constr.Exp1.size[0] * constr.Exp1.size[1]) + 1
                    ))
                    tmprhs.append(self.add_variable(
                        '__tmprhs[{0}]__'.format(constrKey),
                        1))
                    newvars.append(
                        ('__tmplhs[{0}]__'.format(constrKey),
                         (constr.Exp1.size[0] * constr.Exp1.size[1]) + 1))
                    newvars.append(('__tmprhs[{0}]__'.format(constrKey),
                                    1))
                    # v_cons is 0/1/-1 to avoid constants in cone (problem with
                    # duals)
                    expcat = ((2 * constr.Exp1[:]) //
                              (constr.Exp2 - constr.Exp3))
                    v_cons = cvx.matrix([np.sign(expcat.constant[i])
                                         if (expcat.constant is not None) and expcat[i].isconstant() else 0
                                         for i in range(expcat.size[0] * expcat.size[1])],
                                        expcat.size)

                    # lhs and rhs of the cone constraint
                    newcons['tmp_lhs_{0}'.format(constrKey)] = (
                        (2 * constr.Exp1[:] // (constr.Exp2 - constr.Exp3)) + v_cons * noconstant == tmplhs[-1])
                    newcons['tmp_rhs_{0}'.format(constrKey)] = (
                        constr.Exp2 + constr.Exp3 - noconstant == tmprhs[-1])
                    # conic constraints
                    posvars.append(tmprhs[-1].startIndex)
                    newcons['tmp_conequad_{0}'.format(constrKey)] = (
                        -tmprhs[-1]**2 + (tmplhs[-1] | tmplhs[-1]) < 0)

                    if constr.Id is None:
                        constr.Id = {}
                    constr.Id.setdefault('cplex',[])
                    for i in range(constr.Exp1.size[0] * constr.Exp1.size[1] + 1):
                        constr.Id['cplex'].append('lintmp_lhs_{0}_{1}'.format(constrKey,i))

                    constr.Id['cplex'].append('lintmp_rhs_{0}_{1}'.format(constrKey,0))
                    constr.Id['cplex'].append('tmp_conequad_{0}'.format(constrKey))
                    cs = newcons['tmp_conequad_{0}'.format(constrKey)]
                    cs.myconstring = 'tmp_conequad_{0}'.format(constrKey)

                    icone += 1

        NUMVAR_NEW = int(_bsum([(var.endIndex - var.startIndex)  # new vars including cone vars
                                for var in self.variables.values()
                                if not('cplex' in var.passed)]))

        # total number of variables (including extra vars for cones)
        NUMVAR = NUMVAR_OLD + NUMVAR_NEW

        # variables

        if (self.options['verbose'] > 1) and NUMVAR_NEW > 0:
            limitbar = NUMVAR_NEW
            prog = ProgressBar(0, limitbar, None, mode='fixed')
            oldprog = str(prog)
            print('Creating variables...')
            print()

        colnames = []
        types = []

        lb = {}
        ub = {}

        if NUMVAR_NEW:

            # specify bounds later, in constraints
            ub = dict((j, cplex.infinity) for j in range(NUMVAR_OLD, NUMVAR))
            lb = dict((j, -cplex.infinity) for j in range(NUMVAR_OLD, NUMVAR))

            for kvar, variable in [(kvar, variable) for (kvar, variable)
                                   in six.iteritems(self.variables)
                                   if 'cplex' not in variable.passed]:

                variable.cplex_startIndex = variable.startIndex + OFFSET_CV
                variable.cplex_endIndex = variable.endIndex + OFFSET_CV
                sj = variable.cplex_startIndex
                ej = variable.cplex_endIndex

                for ind, (lo, up) in six.iteritems(variable.bnd):
                    if not(lo is None):
                        lb[sj + ind] = lo
                    if not(up is None):
                        ub[sj + ind] = up

            vartopass = sorted([(variable.cplex_startIndex, variable) for (kvar, variable)
                                in six.iteritems(self.variables)
                                if 'cplex' not in variable.passed])

            for (vcsi, variable) in vartopass:
                variable.passed.append('cplex')
                varsize = variable.endIndex - variable.startIndex

                for k in range(varsize):
                    colnames.append(variable.name + '_' + str(k))
                    types.append(cplex_type[variable.vtype])

                    if self.options['verbose'] > 1:
                        #<--display progress
                        prog.increment_amount()
                        if oldprog != str(prog):
                            print(prog, "\r", end="")
                            sys.stdout.flush()
                            oldprog = str(prog)
                        #-->

            if self.options['verbose'] > 1:
                prog.update_amount(limitbar)
                print(prog, "\r", end="")
                print()

        # parse all vars for hotstart
        mipstart_ind = []
        mipstart_vals = []
        if self.options['hotstart']:
            for kvar, variable in six.iteritems(self.variables):
                sj = variable.cplex_startIndex
                ej = variable.cplex_endIndex
                if variable.is_valued():
                    mipstart_ind.extend(range(sj, ej))
                    mipstart_vals.extend(variable.value)

        # parse all variable for the obective (only if not obj_passed)
        newobjcoefs = []
        quad_terms = []
        if 'cplex' not in self.obj_passed:
            self.obj_passed.append('cplex')

            if self.objective[1] is None:
                objective = {}
            elif isinstance(self.objective[1], QuadExp):
                objective = self.objective[1].aff.factors
            elif isinstance(self.objective[1], AffinExp):
                objective = self.objective[1].factors

            for variable, vect in six.iteritems(objective):
                sj = variable.cplex_startIndex
                newobjcoefs.extend(zip(vect.J + sj, vect.V))

            if isinstance(self.objective[1], QuadExp):
                qd = self.objective[1].quad
                for i, j in qd:
                    fact = qd[i, j]
                    si = i.cplex_startIndex
                    sj = j.cplex_startIndex
                    if (j, i) in qd:  # quad stores x'*A1*y + y'*A2*x
                        if si < sj:
                            fact += qd[j, i].T
                        elif si > sj:
                            fact = cvx.sparse([0])
                        elif si == sj:
                            pass
                    quad_terms += zip(fact.I + si, fact.J + sj, 2 * fact.V)

        # constraints

        NUMCON_NEW = int(_bsum([(cs.Exp1.size[0] * cs.Exp1.size[1])
                                for cs in self.constraints
                                if (cs.typeOfConstraint.startswith('lin'))
                                and not('cplex' in cs.passed)] +
                               [1 for cs in self.constraints
                                if (cs.typeOfConstraint == 'quad')
                                and not('cplex' in cs.passed)]
                               )
                         )

        # progress bar
        if self.options['verbose'] > 0:
            print()
            print('adding constraints...')
            print()
        if self.options['verbose'] > 1:
            limitbar = NUMCON_NEW
            prog = ProgressBar(0, limitbar, None, mode='fixed')
            oldprog = str(prog)

        rows = []
        cols = []
        vals = []
        rhs = []
        rownames = []
        senses = ''

        ql = []
        qq = []
        qc = []
        qnames = []
        qind = c.quadratic_constraints.get_num()

        # join all constraints
        def join_iter(it1, it2):
            for i in it1:
                yield i
            for i in it2:
                yield i

        allcons = join_iter(enumerate(self.constraints),
                            six.iteritems(newcons))

        irow = 0

        for constrKey, constr in allcons:

            if 'cplex' in constr.passed:
                continue
            else:
                constr.passed.append('cplex')

            # init of boundcons[key]
            if isinstance(constrKey,int):
                boundcons.append([])

            if constr.typeOfConstraint[:3] == 'lin':

                # parse the (i,j,v) triple
                ijv = []
                for var, fact in six.iteritems(
                        (constr.Exp1 - constr.Exp2).factors):
                    if not isinstance(fact, cvx.base.spmatrix):
                        fact = cvx.sparse(fact)
                    sj = var.cplex_startIndex
                    ijv.extend(zip(fact.I, fact.J + sj, fact.V))
                ijvs = sorted(ijv)

                itojv = {}
                lasti = -1
                for (i, j, v) in ijvs:
                    if i == lasti:
                        itojv[i].append((j, v))
                    else:
                        lasti = i
                        itojv[i] = [(j, v)]

                # constant term
                szcons = constr.Exp1.size[0] * constr.Exp1.size[1]
                rhstmp = cvx.matrix(0., (szcons, 1))
                constant1 = constr.Exp1.constant  # None or a 1*1 matrix
                constant2 = constr.Exp2.constant
                if not constant1 is None:
                    rhstmp = rhstmp - constant1
                if not constant2 is None:
                    rhstmp = rhstmp + constant2

                # constraint of the form 0*x==a
                if len(itojv) != szcons:
                    zero_rows = (set(range(szcons)) - set(itojv.keys()))
                    for i in zero_rows:
                        if ((constr.typeOfConstraint[:4] == 'lin<' and rhstmp[i] < 0) or
                                (constr.typeOfConstraint[:4] == 'lin>' and rhstmp[i] > 0) or
                                (constr.typeOfConstraint[:4] == 'lin=' and rhstmp[i] != 0)):
                            raise Exception(
                                'you try to add a constraint of the form 0 * x == 1')
                    constr.zero_rows = list(zero_rows)

                for i, jv in six.iteritems(itojv):
                    r = rhstmp[i]
                    if len(jv) == 1 and self.options['pass_simple_cons_as_bound']:
                        # BOUND
                        j, v = jv[0]
                        b = r / float(v)
                        if j < NUMVAR_OLD:
                            clj = c.variables.get_lower_bounds(j)
                            cuj = c.variables.get_upper_bounds(j)
                        else:
                            clj = lb[j]
                            cuj = ub[j]
                        if v > 0:
                            if constr.typeOfConstraint[:4] in ['lin<', 'lin=']:
                                if b < cuj:
                                    ub[j] = b
                            if constr.typeOfConstraint[:4] in ['lin>', 'lin=']:
                                if b > clj:
                                    lb[j] = b
                        else:  # v<0
                            if constr.typeOfConstraint[:4] in ['lin<', 'lin=']:
                                if b > clj:
                                    lb[j] = b
                            if constr.typeOfConstraint[:4] in ['lin>', 'lin=']:
                                if b < cuj:
                                    ub[j] = b
                        if constr.typeOfConstraint[3] == '=':
                            b = '='
                        if isinstance(constrKey,int):#a regular constraint, not a tmp constraint for cones
                            boundcons[constrKey].append((i, j, b, v))
                    else:
                        if constr.typeOfConstraint[:4] == 'lin<':
                            senses += "L"  # lower
                        elif constr.typeOfConstraint[:4] == 'lin>':
                            senses += "G"  # greater
                        elif constr.typeOfConstraint[:4] == 'lin=':
                            senses += "E"  # equal

                        rows.extend([irow] * len(jv))
                        cols.extend([j for j, v in jv])
                        vals.extend([v for j, v in jv])
                        rhs.append(r)
                        irow += 1
                        rownames.append('lin' + str(constrKey) + '_' + str(i))
                        if constr.Id is None:
                            constr.Id = {}
                        constr.Id.setdefault('cplex',[])
                        constr.Id['cplex'].append('lin' + str(constrKey) + '_' + str(i))

                    if self.options['verbose'] > 1:
                        #<--display progress
                        prog.increment_amount()
                        if oldprog != str(prog):
                            print(prog, "\r", end="")
                            sys.stdout.flush()
                            oldprog = str(prog)
                        #-->

            elif constr.typeOfConstraint == 'quad':
                # quad part
                qind1, qind2, qval = [], [], []
                qd = constr.Exp1.quad
                for i, j in qd:
                    fact = qd[i, j]
                    si = i.cplex_startIndex
                    sj = j.cplex_startIndex
                    if (j, i) in qd:  # quad stores x'*A1*y + y'*A2*x
                        if si < sj:
                            fact += qd[j, i].T
                        elif si > sj:
                            fact = cvx.sparse([0])
                        elif si == sj:
                            pass
                    qind1.extend(fact.I + si)
                    qind2.extend(fact.J + sj)
                    qval.extend(fact.V)
                q_exp = cplex.SparseTriple(ind1=qind1,
                                           ind2=qind2,
                                           val=qval)
                # lin part
                lind, lval = [], []
                af = constr.Exp1.aff.factors
                for var in af:
                    sj = var.cplex_startIndex
                    lind.extend(af[var].J + sj)
                    lval.extend(af[var].V)
                l_exp = cplex.SparsePair(ind=lind, val=lval)

                # constant
                qcs = 0.
                if not(constr.Exp1.aff.constant is None):
                    qcs = - constr.Exp1.aff.constant[0]

                ql += [l_exp]
                qq += [q_exp]
                qc += [qcs]
                if constr.myconstring is not None and 'tmp_conequad' in constr.myconstring:
                    qnames.append(constr.myconstring)
                    qind+=1
                else:
                    qnames.append('q'+str(qind))
                    qind+=1

                if self.options['verbose'] > 1:
                    #<--display progress
                    prog.increment_amount()
                    if oldprog != str(prog):
                        print(prog, "\r", end="")
                        sys.stdout.flush()
                        oldprog = str(prog)
                    #-->

            elif constr.typeOfConstraint[2:] == 'cone':
                pass
                # will be handled in the newcons dictionary

            else:
                raise Exception(
                    'type of constraint not handled (yet ?) for cplex:{0}'.format(
                        constr.typeOfConstraint))

        if self.options['verbose'] > 1:
            prog.update_amount(limitbar)
            print(prog, "\r", end="")
            print()

        print_message_not_printed_yet = True
        warning_message_not_printed_yet = True
        todel_from_boundcons = []
        for cs in self._deleted_constraints:
            if 'cplex' in cs.passed:
                continue
            else:
                cs.passed.append('cplex')

            if print_message_not_printed_yet and self.options['verbose'] > 0:
                print()
                print('Removing constraints from Cplex instance...')
                print_message_not_printed_yet = False

            sgn = cs.typeOfConstraint[3]
            todel_from_boundcons.append(cs.original_index)
            if (self.options['verbose'] > 0 and
                self.options['pass_simple_cons_as_bound'] and
                warning_message_not_printed_yet and
                self.cplex_boundcons[cs.original_index]
                ):

                print("\033[1;31m*** You have been removing a constraint that can be "+
                      "(partly) interpreted as a variable bound. This is not safe when "
                      "the option ``pass_simple_cons_as_bound`` is set to True\033[0m")
                warning_message_not_printed_yet = False


            for i,j,b,v in self.cplex_boundcons[cs.original_index]:
                if sgn == '=':
                    lb[j] = -cplex.infinity
                    ub[j] = cplex.infinity
                elif (sgn == '<' and v>0) or (sgn == '>' and v<0):
                    ub[j] = cplex.infinity
                elif (sgn == '<' and v>0) or (sgn == '>' and v<0):
                    lb[j] = -cplex.infinity

            for i in cs.Id['cplex']:
                if cs.typeOfConstraint == 'quad':
                    c.quadratic_constraints.delete(i)
                elif cs.typeOfConstraint.startswith('lin'):
                    c.linear_constraints.delete(i)
                else:
                    if i.startswith('tmp_conequad'):
                        c.quadratic_constraints.delete(i)
                    else:
                        c.linear_constraints.delete(i)
                        ind,jj = [int(kk) for kk in i.split('hs_',1)[1].split('_')]
                        varname = '__tmp'+i.split('hs_')[0][-1]+'hs[{0}]___{1}'.format(ind,jj)
                        c.variables.delete(varname)

        for i in todel_from_boundcons:
            del boundcons[i]

        if self.options['verbose'] > 0:
            print()
            print('Passing to cplex...')

        c.variables.add(names=colnames, types=types)

        #nonnegative constraints for dummy variables in cone:
        for i in posvars:
            lb[i]=0.

        if lb:
            c.variables.set_lower_bounds(six.iteritems(lb))
        if ub:
            c.variables.set_upper_bounds(six.iteritems(ub))
        if newobjcoefs:
            c.objective.set_linear(newobjcoefs)

        if len(quad_terms) > 0:
            c.objective.set_quadratic_coefficients(quad_terms)

        offset = c.linear_constraints.get_num()
        rows = [r + offset for r in rows]
        c.linear_constraints.add(rhs=rhs, senses=senses, names=rownames)

        if len(rows) > 0:
            c.linear_constraints.set_coefficients(zip(rows, cols, vals))
        for lp, qp, qcs,nam in zip(ql, qq, qc,qnames):
            c.quadratic_constraints.add(lin_expr=lp,
                                        quad_expr=qp,
                                        rhs=qcs,
                                        sense="L",
                                        name = nam)

        if self.options['hotstart'] and len(mipstart_ind) > 0:
            c.MIP_starts.add(cplex.SparsePair(
                ind=mipstart_ind, val=mipstart_vals),
                c.MIP_starts.effort_level.repair)

        tp = self.type
        if tp == 'LP':
            c.set_problem_type(c.problem_type.LP)
        elif tp == 'MIP':
            c.set_problem_type(c.problem_type.MILP)
        elif tp in ('QCQP', 'SOCP', 'Mixed (SOCP+quad)'):
            c.set_problem_type(c.problem_type.QCP)
        elif tp in ('MIQCP', 'MISOCP', 'Mixed (MISOCP+quad)'):
            c.set_problem_type(c.problem_type.MIQCP)
        elif tp == 'QP':
            c.set_problem_type(c.problem_type.QP)
        elif tp == 'MIQP':
            c.set_problem_type(c.problem_type.MIQP)
        else:
            raise Exception('unhandled type of problem')

        self.cplex_Instance = c
        self.cplex_boundcons = boundcons

        if 'noconstant' in newcons or len(tmplhs) > 0:
            self._remove_temporary_variables()

        if self.options['verbose'] > 0:
            print('CPLEX INSTANCE created')

    def _make_cvxopt_instance(self, aff_part_of_quad=True, cone_as_quad=False,
                              new_cvxopt_cons_only=False,
                              new_scip_cons_only=False,
                              reset=True,
                              hard_coded_bounds=False):
        """
        defines the variables in self.cvxoptVars, used by the cvxopt solver
        new_scip_cons_only: if True, consider only cons where 'scip' not in passed
        new_cvxopt_cons_only: if True, consider only cons where 'cvxopt' not in passed
        reset: if True, reset the cvxoptVars at the beginning.
        """
        if any([('cvxopt' not in cs.passed) for cs in self._deleted_constraints]):
            for cs in self._deleted_constraints:
                if 'cvxopt' not in cs.passed:
                    cs.passed.append('cvxopt')
            self.reset_cvxopt_instance(True)

        ss = self.numberOfVars
        # initial values
        if self.cvxoptVars['A'] is None:
            reset = True

        if reset:
            self.cvxoptVars['A'] = spmatrix([], [], [], (0, ss), tc='d')
            self.cvxoptVars['b'] = cvx.matrix([], (0, 1), tc='d')
            self.cvxoptVars['Gl'] = spmatrix([], [], [], (0, ss), tc='d')
            self.cvxoptVars['hl'] = cvx.matrix([], (0, 1), tc='d')
            self.cvxoptVars['Gq'] = []
            self.cvxoptVars['hq'] = []
            self.cvxoptVars['Gs'] = []
            self.cvxoptVars['hs'] = []
            self.cvxoptVars['quadcons'] = []
        elif ss > self.cvxoptVars['A'].size[1]:
            nv = ss - self.cvxoptVars['A'].size[1]
            self.cvxoptVars['A'] = cvx.sparse([[self.cvxoptVars['A']], [spmatrix(
                [], [], [], (self.cvxoptVars['A'].size[0], nv), tc='d')]])
            self.cvxoptVars['Gl'] = cvx.sparse([[self.cvxoptVars['Gl']], [spmatrix(
                [], [], [], (self.cvxoptVars['Gl'].size[0], nv), tc='d')]])
            for i, Gqi in enumerate(self.cvxoptVars['Gq']):
                self.cvxoptVars['Gq'][i] = cvx.sparse(
                    [[Gqi], [spmatrix([], [], [], (Gqi.size[0], nv), tc='d')]])
            for i, Gsi in enumerate(self.cvxoptVars['Gs']):
                self.cvxoptVars['Gs'][i] = cvx.sparse(
                    [[Gsi], [spmatrix([], [], [], (Gsi.size[0], nv), tc='d')]])

        # objective
        if not((new_scip_cons_only and 'scip' in self.obj_passed) or
               (new_cvxopt_cons_only and 'cvxopt' in self.obj_passed)):
            if isinstance(self.objective[1], QuadExp):
                self.cvxoptVars['quadcons'].append(('_obj', -1))
                objexp = self.objective[1].aff
            elif isinstance(self.objective[1], LogSumExp):
                objexp = self.objective[1].Exp
            else:
                objexp = self.objective[1]
            if self.numberLSEConstraints == 0:
                if self.objective[0] == 'find':
                    self.cvxoptVars['c'] = cvx.matrix(0, (ss, 1), tc='d')
                elif self.objective[0] == 'min':
                    (c, constantInObjective) = self._makeGandh(objexp)
                    self.cvxoptVars['c'] = cvx.matrix(c, tc='d').T
                elif self.objective[0] == 'max':
                    (c, constantInObjective) = self._makeGandh(objexp)
                    self.cvxoptVars['c'] = -cvx.matrix(c, tc='d').T
            else:
                if self.objective[0] == 'find':
                    self.cvxoptVars['F'] = cvx.matrix(0, (1, ss), tc='d')
                    self.cvxoptVars['K'] = [0]
                else:
                    (F, g) = self._makeGandh(objexp)
                    self.cvxoptVars['K'] = [F.size[0]]
                    if self.objective[0] == 'min':
                        self.cvxoptVars['F'] = cvx.matrix(F, tc='d')
                        self.cvxoptVars['g'] = cvx.matrix(g, tc='d')
                    elif self.objective[0] == 'max':
                        self.cvxoptVars['F'] = -cvx.matrix(F, tc='d')
                        self.cvxoptVars['g'] = -cvx.matrix(g, tc='d')

            if not(aff_part_of_quad) and isinstance(
                    self.objective[1], QuadExp):
                self.cvxoptVars['c'] = cvx.matrix(0, (ss, 1), tc='d')
            if new_scip_cons_only:
                self.obj_passed.append('scip')
            if new_cvxopt_cons_only:
                self.obj_passed.append('cvxopt')
        elif self.cvxoptVars['c'].size[0] < ss:
            nv = ss - self.cvxoptVars['c'].size[0]
            self.cvxoptVars['c'] = cvx.matrix(cvx.sparse(
                [self.cvxoptVars['c'], spmatrix([], [], [], (nv, 1), tc='d')]))

        if new_cvxopt_cons_only:
            for var in self.variables.values():
                if 'cvxopt' not in var.passed:
                    var.passed.append('cvxopt')

        if self.options['verbose'] > 1:
            limitbar = self.numberAffConstraints + self.numberConeConstraints + \
                self.numberQuadConstraints + self.numberLSEConstraints + self.numberSDPConstraints
            prog = ProgressBar(0, limitbar, None, mode='fixed')
            oldprog = str(prog)

        # constraints
        for k, consk in enumerate(self.constraints):
            if self.options['verbose'] > 1:
                #<--display progress
                prog.increment_amount()
                if oldprog != str(prog):
                    print(prog, "\r", end="")
                    sys.stdout.flush()
                    oldprog = str(prog)
                #-->
            if new_scip_cons_only:
                if 'scip' in consk.passed:
                    continue
                else:
                    consk.passed.append('scip')
            if new_cvxopt_cons_only:
                if 'cvxopt' in consk.passed:
                    continue
                else:
                    consk.passed.append('cvxopt')
            # linear constraints
            if consk.typeOfConstraint[:3] == 'lin':
                sense = consk.typeOfConstraint[3]
                (G_lhs, h_lhs) = self._makeGandh(consk.Exp1)
                (G_rhs, h_rhs) = self._makeGandh(consk.Exp2)
                if sense == '=':
                    self.cvxoptVars['A'] = cvx.sparse(
                        [self.cvxoptVars['A'], G_lhs - G_rhs])
                    self.cvxoptVars['b'] = cvx.matrix(
                        [self.cvxoptVars['b'], h_rhs - h_lhs])
                elif sense == '<':
                    self.cvxoptVars['Gl'] = cvx.sparse(
                        [self.cvxoptVars['Gl'], G_lhs - G_rhs])
                    self.cvxoptVars['hl'] = cvx.matrix(
                        [self.cvxoptVars['hl'], h_rhs - h_lhs])
                elif sense == '>':
                    self.cvxoptVars['Gl'] = cvx.sparse(
                        [self.cvxoptVars['Gl'], G_rhs - G_lhs])
                    self.cvxoptVars['hl'] = cvx.matrix(
                        [self.cvxoptVars['hl'], h_lhs - h_rhs])
                else:
                    raise NameError('unexpected case')
            elif consk.typeOfConstraint == 'SOcone':
                if not(cone_as_quad):
                    (A, b) = self._makeGandh(consk.Exp1)
                    (c, d) = self._makeGandh(consk.Exp2)
                    self.cvxoptVars['Gq'].append(cvx.sparse([-c, -A]))
                    self.cvxoptVars['hq'].append(cvx.matrix([d, b]))
                else:
                    self.cvxoptVars['quadcons'].append(
                        (k, self.cvxoptVars['Gl'].size[0]))
                    if aff_part_of_quad:
                        raise Exception('cone_as_quad + aff_part_of_quad')
            elif consk.typeOfConstraint == 'RScone':
                if not(cone_as_quad):
                    (A, b) = self._makeGandh(consk.Exp1)
                    (c1, d1) = self._makeGandh(consk.Exp2)
                    (c2, d2) = self._makeGandh(consk.Exp3)
                    self.cvxoptVars['Gq'].append(
                        cvx.sparse([-c1 - c2, -2 * A, c2 - c1]))
                    self.cvxoptVars['hq'].append(
                        cvx.matrix([d1 + d2, 2 * b, d1 - d2]))
                else:
                    self.cvxoptVars['quadcons'].append(
                        (k, self.cvxoptVars['Gl'].size[0]))
                    if aff_part_of_quad:
                        raise Exception('cone_as_quad + aff_part_of_quad')
            elif consk.typeOfConstraint == 'lse':
                (F, g) = self._makeGandh(consk.Exp1)
                self.cvxoptVars['F'] = cvx.sparse([self.cvxoptVars['F'], F])
                self.cvxoptVars['g'] = cvx.matrix([self.cvxoptVars['g'], g])
                self.cvxoptVars['K'].append(F.size[0])
            elif consk.typeOfConstraint == 'quad':
                self.cvxoptVars['quadcons'].append(
                    (k, self.cvxoptVars['Gl'].size[0]))
                if aff_part_of_quad:
                    # quadratic part handled later
                    (G_lhs, h_lhs) = self._makeGandh(consk.Exp1.aff)
                    self.cvxoptVars['Gl'] = cvx.sparse(
                        [self.cvxoptVars['Gl'], G_lhs])
                    self.cvxoptVars['hl'] = cvx.matrix(
                        [self.cvxoptVars['hl'], -h_lhs])
            elif consk.typeOfConstraint[:3] == 'sdp':
                sense = consk.typeOfConstraint[3]
                (G_lhs, h_lhs) = self._makeGandh(consk.Exp1)
                (G_rhs, h_rhs) = self._makeGandh(consk.Exp2)
                if sense == '<':
                    self.cvxoptVars['Gs'].append(G_lhs - G_rhs)
                    self.cvxoptVars['hs'].append(h_rhs - h_lhs)
                elif sense == '>':
                    self.cvxoptVars['Gs'].append(G_rhs - G_lhs)
                    self.cvxoptVars['hs'].append(h_lhs - h_rhs)
                else:
                    raise NameError('unexpected case')

            else:
                raise NameError('unexpected case')

        # hard-coded bounds
        if hard_coded_bounds:
            for (var, variable) in six.iteritems(self.variables):
                for ind, (lo, up) in six.iteritems(variable.bnd):
                    if not(lo is None):
                        (G_lhs, h_lhs) = self._makeGandh(variable[ind])
                        self.cvxoptVars['Gl'] = cvx.sparse(
                            [self.cvxoptVars['Gl'], -G_lhs])
                        self.cvxoptVars['hl'] = cvx.matrix(
                            [self.cvxoptVars['hl'], -lo])
                    if not(up is None):
                        (G_lhs, h_lhs) = self._makeGandh(variable[ind])
                        self.cvxoptVars['Gl'] = cvx.sparse(
                            [self.cvxoptVars['Gl'], G_lhs])
                        self.cvxoptVars['hl'] = cvx.matrix(
                            [self.cvxoptVars['hl'], up])

        # reshape hs matrices as square matrices
        # for m in self.cvxoptVars['hs']:
        #        n=int(np.sqrt(len(m)))
        #        m.size=(n,n)

        if self.options['verbose'] > 1:
            prog.update_amount(limitbar)
            print(prog, "\r", end="")
            sys.stdout.flush()
            print()

    @staticmethod
    def _picos2glpk_variable_index(globalVariableIndex):
        return globalVariableIndex + 1

    @staticmethod
    def _glpk2picos_variable_index(globalVariableIndex):
        return globalVariableIndex - 1

    def _make_glpk_instance(self):
        import swiglpk as glpk

        if self.options['verbose'] > 0:
            print("Building a GLPK problem instance.")
            glpk.glp_term_out(glpk.GLP_ON)
        else:
            glpk.glp_term_out(glpk.GLP_OFF)

        # TODO: Allow updates to instance, if GLPK supports this.
        if self.glpk_Instance is not None:
            self.reset_glpk_instance()

        self.glpk_Instance = glpk.glp_create_prob();

        # An alias to the problem instance.
        p = self.glpk_Instance

        # Set the objective.
        if self.objective[0] in ("find", "min"):
            glpk.glp_set_obj_dir(p, glpk.GLP_MIN)
        elif self.objective[0] is "max":
            glpk.glp_set_obj_dir(p, glpk.GLP_MAX)
        else:
            raise NotImplementedError("Objective '{0} not supported by GLPK."
                .format(self.objective[0]))

        # Set objective function shift
        if self.objective[1] is not None and self.objective[1].constant is not None:
            if not isinstance(self.objective[1], AffinExp):
                raise NotImplementedError("Non-linear objective function not "
                    "supported by GLPK.")

            if self.objective[1].constant.size != (1,1):
                raise NotImplementedError("Non-scalar objective function not "
                    "supported by GLPK.")

            glpk.glp_set_obj_coef(p, 0, self.objective[1].constant[0])

        # Add variables.
        # Multideminsional variables are split into multiple scalar variables
        # represented as matrix columns within GLPK.
        for varName in self.varNames:
            var = self.variables[varName]

            # Add a column for every scalar variable.
            numCols = var.size[0] * var.size[1]
            glpk.glp_add_cols(p, numCols)

            for localIndex, picosIndex in enumerate(range(var.startIndex, var.endIndex)):
                glpkIndex = self._picos2glpk_variable_index(picosIndex)

                # Assign a name to the scalar variable.
                scalarName = varName
                if numCols > 1:
                    x = localIndex // var.size[0]
                    y = localIndex % var.size[0]
                    scalarName += "_{:d}_{:d}".format(x + 1, y + 1)
                glpk.glp_set_col_name(p, glpkIndex, scalarName)

                # Assign bounds to the scalar variable.
                lower, upper = var.bnd.get(localIndex, (None, None))
                if lower is not None and upper is not None:
                    if lower == upper:
                        glpk.glp_set_col_bnds(p, glpkIndex, glpk.GLP_FX, lower, upper)
                    else:
                        glpk.glp_set_col_bnds(p, glpkIndex, glpk.GLP_DB, lower, upper)
                elif lower is not None and upper is None:
                    glpk.glp_set_col_bnds(p, glpkIndex, glpk.GLP_LO, lower, 0)
                elif lower is None and upper is not None:
                    glpk.glp_set_col_bnds(p, glpkIndex, glpk.GLP_UP, 0, upper)
                else:
                    glpk.glp_set_col_bnds(p, glpkIndex, glpk.GLP_FR, 0, 0)

                # Assign a type to the scalar variable.
                if var.vtype == "continuous":
                    glpk.glp_set_col_kind(p, glpkIndex, glpk.GLP_CV)
                elif var.vtype == "integer":
                    glpk.glp_set_col_kind(p, glpkIndex, glpk.GLP_IV)
                elif var.vtype == "binary":
                    glpk.glp_set_col_kind(p, glpkIndex, glpk.GLP_BV)
                else:
                    raise NotImplementedError("Variable type '{0}' not "
                        "supported by GLPK.".format(var.vtype()))

                # Set objective function coefficient of the scalar variable.
                if self.objective[1] is not None and var in self.objective[1].factors:
                    glpk.glp_set_obj_coef(p, glpkIndex, self.objective[1].factors[var][localIndex])

        # Add constraints.
        # Multideminsional constraints are split into multiple scalar
        # constraints represented as matrix rows within GLPK.
        rowOffset = 1
        for constraintNum, constraint in enumerate(self.constraints):
            if constraint.typeOfConstraint not in ("lin<", "lin=", "lin>"):
                raise NotImplementedError("Non-linear constraints not supported"
                    " by GLPK.")

            # Add a row for every scalar constraint.
            # Internally, GLPK uses an auxiliary variable for every such row,
            # bounded by the right hand side of the scalar constraint in a
            # canonical form.
            numRows = constraint.Exp1.size[0] * constraint.Exp1.size[1]
            glpk.glp_add_rows(p, numRows)

            # Transform constraint into canonical form understood by GLPK.
            LHS = constraint.Exp1 - constraint.Exp2
            if LHS.constant:
                RHS = AffinExp(size=LHS.size, constant=-LHS.constant)
            else:
                # TODO: Give every AffinExp without an explicit constant a
                #       constant of zero?
                RHS = AffinExp(size=LHS.size) + 0
            LHS += RHS

            if (self.options['verbose'] > 1):
                print("Handling PICOS Constraint: ", constraint)

            # Split multidimensional constraints into multiple scalar constraints.
            for localConstraintIndex in range(numRows):
                # Determine GLPK's row index of the scalar constraint.
                glpkConstraintIndex = rowOffset + localConstraintIndex

                # Extract the scalar constraint for the current row.
                lhs = LHS[localConstraintIndex]
                rhs = RHS.constant[localConstraintIndex]

                # Give the auxiliary variable associated with the current row a name.
                if constraint.key:
                    name = constraint.key
                else:
                    name = "rhs_{:d}".format(constraintNum)
                if numRows > 1:
                    x = localConstraintIndex // constraint.Exp1.size[0]
                    y = localConstraintIndex % constraint.Exp1.size[0]
                    name += "_{:d}_{:d}".format(x + 1, y + 1)
                glpk.glp_set_row_name(p, glpkConstraintIndex, name)

                # Assign bounds to the auxiliary variable.
                if constraint.typeOfConstraint == "lin=":
                    glpk.glp_set_row_bnds(p, glpkConstraintIndex, glpk.GLP_FX, rhs, rhs)
                elif constraint.typeOfConstraint == "lin<":
                    glpk.glp_set_row_bnds(p, glpkConstraintIndex, glpk.GLP_UP, 0, rhs)
                elif constraint.typeOfConstraint == "lin>":
                    glpk.glp_set_row_bnds(p, glpkConstraintIndex, glpk.GLP_LO, rhs, 0)

                # Set coefficients for current row.
                # Note that GLPK requires a glpk.intArray containing column
                # indices and a glpk.doubleArray of same size containing the
                # coefficients for the listed column index. The first element
                # of both arrays (with index 0) is skipped by GLPK.
                setColumns = []
                setCoefficients = []
                for var, coefficients in lhs.factors.items():
                    for localVariableIndex in range(var.endIndex - var.startIndex):
                        glpkVariableIndex = self._picos2glpk_variable_index(
                            var.startIndex + localVariableIndex)
                        setColumns.append(glpkVariableIndex)
                        setCoefficients.append(coefficients[localVariableIndex])
                numSetColumns = len(setColumns)
                setColumns_glpk = glpk.intArray(numSetColumns + 1)
                setCoefficients_glpk = glpk.doubleArray(numSetColumns + 1)
                for i in range(numSetColumns):
                    setColumns_glpk[i + 1] = setColumns[i]
                    setCoefficients_glpk[i + 1] = setCoefficients[i]
                if (self.options['verbose'] > 1):
                    print("Adding GLPK Constraint: Variables:", setColumns,
                        "Coefficients:", setCoefficients, "RHS:", rhs)
                glpk.glp_set_mat_row(p, glpkConstraintIndex, numSetColumns,
                    setColumns_glpk, setCoefficients_glpk)

            rowOffset += numRows

        if self.options['verbose'] > 0:
            print("GLPK problem instance built.")

    #-----------
    # mosek tool
    #-----------

    # Define a stream printer to grab output from MOSEK
    def _streamprinter(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    # separate a linear constraint between 'plain' vars and matrix 'bar' variables
    # J and V denote the sparse indices/values of the constraints for the
    # whole (s-)vectorized vector
    def _separate_linear_cons(self, J, V, idx_sdp_vars):
        # sparse values of the constraint for 'plain' variables
        jj = []
        vv = []
        # sparse values of the constraint for the next svec bar variable
        js = []
        vs = []
        mats = []
        offset = 0
        if idx_sdp_vars:
            idxsdpvars = [ti for ti in idx_sdp_vars]
            nextsdp = idxsdpvars.pop()
        else:
            return J, V, []
        for (j, v) in zip(J, V):
            if j < nextsdp[0]:
                jj.append(j - offset)
                vv.append(v)
            elif j < nextsdp[1]:
                js.append(j - nextsdp[0])
                vs.append(v)
            else:
                while j >= nextsdp[1]:
                    mats.append(
                        svecm1(
                            spmatrix(
                                vs,
                                js,
                                [0] *
                                len(js),
                                (nextsdp[1] -
                                 nextsdp[0],
                                    1)),
                            triu=True).T)
                    js = []
                    vs = []
                    offset += (nextsdp[1] - nextsdp[0])
                    try:
                        nextsdp = idxsdpvars.pop()
                    except IndexError:
                        nextsdp = (float('inf'), float('inf'))
                if j < nextsdp[0]:
                    jj.append(j - offset)
                    vv.append(v)
                elif j < nextsdp[1]:
                    js.append(j - nextsdp[0])
                    vs.append(v)
        while len(mats) < len(idx_sdp_vars):
            mats.append(
                svecm1(
                    spmatrix(
                        vs,
                        js,
                        [0] *
                        len(js),
                        (nextsdp[1] -
                         nextsdp[0],
                            1)),
                    triu=True).T)
            js = []
            vs = []
            nextsdp = (0, 1)  # doesnt matter, it will be an empt matrix anyway
        return jj, vv, mats

    def _make_mosek_instance(self):
        """
        defines the variables msk_env and msk_task used by the solver mosek.
        """


        #thi is not needed anymore, because we are handling deleted constraints dynamically in mosek
        '''
        if any([('mosek' not in cs.passed) for cs in self._deleted_constraints]):
            for cs in self._deleted_constraints:
                if 'mosek' not in cs.passed:
                    cs.passed.append('mosek')
            self.reset_mosek_instance(True)
        '''

        if self.options['verbose'] > 0:
            print('build mosek instance')
        #import mosek
        # force to use version 6.0 of mosek.
        if self.options['solver'] == 'mosek6':
            try:
                import mosek as mosek
                version7 = not(hasattr(mosek, 'cputype'))
                if version7:
                    raise ImportError(
                        "I could''t find mosek 6.0; the package named mosek is the v7.0")
            except:
                raise ImportError('mosek library not found')
        # try to load mosek7, else use the default mosek package (which can be
        # any version)
        else:
            try:
                import mosek7 as mosek
            except ImportError:
                try:
                    import mosek as mosek
                except:
                    raise ImportError('mosek library not found')

        # True if this is the version 7 of MOSEK
        version7 = not(hasattr(mosek, 'cputype'))

        if self.msk_env and self.msk_task:
            env = self.msk_env
            task = self.msk_task
        else:
            # Make a MOSEK environment
            env = mosek.Env()
            # Attach a printer to the environment
            if self.options['verbose'] >= 1:
                env.set_Stream(mosek.streamtype.log, self._streamprinter)
            # Create a task
            task = env.Task(0, 0)
            # Attach a printer to the task
            if self.options['verbose'] >= 1:
                task.set_Stream(mosek.streamtype.log, self._streamprinter)

        # Give MOSEK an estimate of the size of the input data.
        # This is done to increase the speed of inputting data.
        reset_hbv_True = False
        NUMVAR_OLD = task.getnumvar()
        if self.options['handleBarVars']:
            NUMVAR0_OLD = int(_bsum([(var.endIndex - var.startIndex)
                                     for var in self.variables.values()
                                     if not(var.semiDef)
                                     and ('mosek' in var.passed)]))
            NUMVAR_NEW = int(_bsum([(var.endIndex - var.startIndex)
                                    for var in self.variables.values()
                                    if not(var.semiDef)
                                    and not('mosek' in var.passed)]))

            indices = [(v.startIndex, v.endIndex, v)
                       for v in self.variables.values()]
            indices = sorted(indices)
            idxsdpvars = [(si, ei)
                          for (si, ei, v) in indices[::-1] if v.semiDef]
            indsdpvar = [i for i, cons in
                         enumerate([cs for cs in self.constraints if cs.typeOfConstraint.startswith('sdp')])
                         if cons.semidefVar]
            if not(idxsdpvars):
                reset_hbv_True = True
                self.options._set('handleBarVars', False)

        else:
            NUMVAR0_OLD = int(_bsum([(var.endIndex - var.startIndex)
                                     for var in self.variables.values()
                                     if ('mosek' in var.passed)]))
            NUMVAR_NEW = int(_bsum([(var.endIndex - var.startIndex)
                                    for var in self.variables.values()
                                    if not('mosek' in var.passed)]))

        # total number of variables (including extra vars for cones, but not
        # the bar vars)
        NUMVAR = NUMVAR_OLD + NUMVAR_NEW
        # number of "plain" vars (without "bar" matrix vars and additional vars
        # for so cones)
        NUMVAR0 = NUMVAR0_OLD + NUMVAR_NEW

        NUMCON_OLD = task.getnumcon()
        NUMCON_NEW = int(_bsum([(cs.Exp1.size[0] * cs.Exp1.size[1])
                                for cs in self.constraints
                                if (cs.typeOfConstraint.startswith('lin'))
                                and not('mosek' in cs.passed)] +
                               [1 for cs in self.constraints
                                if (cs.typeOfConstraint == 'quad')
                                and not('mosek' in cs.passed)]
                               )
                         )

        NUMCON = NUMCON_OLD + NUMCON_NEW

        NUMCONES = task.getnumcone()


        NUMSDP = self.numberSDPConstraints
        if NUMSDP > 0:
            #indices = [(v.startIndex,v.endIndex-v.startIndex) for v in self.variables.values()]
            #indices = sorted(indices)
            #BARVARDIM = [int((8*sz-1)**0.5/2.) for (_,sz) in indices]
            BARVARDIM_OLD = [cs.Exp1.size[0]
                             for cs in self.constraints
                             if cs.typeOfConstraint.startswith('sdp')
                             and ('mosek' in cs.passed)]
            BARVARDIM_NEW = [cs.Exp1.size[0]
                             for cs in self.constraints
                             if cs.typeOfConstraint.startswith('sdp')
                             and not('mosek' in cs.passed)]
            BARVARDIM = BARVARDIM_OLD + BARVARDIM_NEW
        else:
            BARVARDIM_OLD = []
            BARVARDIM_NEW = []
            BARVARDIM = []

        if (NUMSDP and not(version7)) or self.numberLSEConstraints:
            raise Exception(
                'SDP or GP constraints are not interfaced. For SDP, try mosek 7.0')

        #-------------#
        #   new vars  #
        #-------------#
        if version7:
            # Append 'NUMVAR_NEW' variables.
            # The variables will initially be fixed at zero (x=0).
            task.appendvars(NUMVAR_NEW)
            task.appendbarvars(BARVARDIM_NEW)
        else:
            task.append(mosek.accmode.var, NUMVAR_NEW)

        #-------------------------------------------------------------#
        # shift the old cone vars to make some place for the new vars #
        #-------------------------------------------------------------#

        if NUMVAR_NEW:
            # shift in the linear constraints
            if (NUMVAR_OLD > NUMVAR0_OLD):
                for j in range(NUMVAR0_OLD, NUMVAR_OLD):
                    sj = [0] * NUMCON_OLD
                    vj = [0] * NUMCON_OLD
                    if version7:
                        nzj = task.getacol(j, sj, vj)
                        # remove the old column
                        task.putacol(j, sj[:nzj], [0.] * nzj)
                        # rewrites it, shifted to the right
                        task.putacol(j + NUMVAR_NEW, sj[:nzj], vj[:nzj])
                    else:
                        nzj = task.getavec(mosek.accmode.var, j, sj, vj)
                        task.putavec(
                            mosek.accmode.var, j, sj[
                                :nzj], [0.] * nzj)
                        task.putavec(mosek.accmode.var, j +
                                     NUMVAR_NEW, sj[:nzj], vj[:nzj])

            # shift in the conic constraints
            nc = task.getnumcone()
            if nc:
                sub = None# [0] * NUMVAR_OLD
            for icone in range(nc):
                (ctype, cpar, sz) = task.getcone(icone, sub)
                shiftsub = [(s + NUMVAR_NEW if s >= NUMVAR0_OLD else s)
                            for s in sub[:sz]]
                task.putcone(icone, ctype, cpar, shiftsub)

        # WE DO NOT SHIFT QUADSCOEFS, BOUNDS OR OBJCOEFS SINCE THERE MUST NOT
        # BE ANY

        #-------------#
        #   new cons  #
        #-------------#
        if version7:
            # Append 'NUMCON_NEW' empty constraints.
            # The constraints will initially have no bounds.
            task.appendcons(NUMCON_NEW)
        else:
            task.append(mosek.accmode.con, NUMCON_NEW)

        if self.numberQuadNNZ:
            # 1.5 factor because the mosek doc
            task.putmaxnumqnz(int(1.5 * self.numberQuadNNZ))
            # claims it might be good to allocate more space than needed
        #--------------#
        # obj and vars #
        #--------------#

        # find integer variables, put 0-1 bounds on binaries
        ints = []
        for k, var in six.iteritems(self.variables):
            if var.vtype == 'binary':
                for ind, i in enumerate(range(var.startIndex, var.endIndex)):
                    ints.append(i)
                    (clb, cub) = var.bnd.get(ind, (-INFINITY, INFINITY))
                    lb = max(0., clb)
                    ub = min(1., cub)
                    var.bnd._set(ind, (lb, ub))

            elif self.variables[k].vtype == 'integer':
                for i in range(
                        self.variables[k].startIndex,
                        self.variables[k].endIndex):
                    ints.append(i)

            elif self.variables[k].vtype not in ['continuous', 'symmetric']:
                raise Exception('vtype not handled (yet) with mosek')
        if self.options['handleBarVars']:
            ints, _, mats = self._separate_linear_cons(
                ints, [0.] * len(ints), idxsdpvars)
            if any([bool(mat) for mat in mats]):
                raise Exception(
                    'semidef vars with integer elements are not supported')

        # supress all integers
        for j in range(NUMVAR):
            task.putvartype(j, mosek.variabletype.type_cont)

        # specifies integer variables
        for i in ints:
            task.putvartype(i, mosek.variabletype.type_int)

        # objective
        newobj = False
        if 'mosek' not in self.obj_passed:
            newobj = True
            self.obj_passed.append('mosek')
            if self.objective[1]:
                obj = self.objective[1]
                subI = []
                subJ = []
                subV = []
                if isinstance(obj, QuadExp):
                    for i, j in obj.quad:
                        si, ei = i.startIndex, i.endIndex
                        sj, ej = j.startIndex, j.endIndex
                        Qij = obj.quad[i, j]
                        if not isinstance(Qij, cvx.spmatrix):
                            Qij = cvx.sparse(Qij)
                        if si == sj:  # put A+A' on the diag
                            sI = list((Qij + Qij.T).I + si)
                            sJ = list((Qij + Qij.T).J + sj)
                            sV = list((Qij + Qij.T).V)
                            for u in range(len(sI) - 1, -1, -1):
                                # remove when j>i
                                if sJ[u] > sI[u]:
                                    del sI[u]
                                    del sJ[u]
                                    del sV[u]
                        elif si >= ej:  # add A in the lower triang
                            sI = list(Qij.I + si)
                            sJ = list(Qij.J + sj)
                            sV = list(Qij.V)
                        else:  # add B' in the lower triang
                            sI = list((Qij.T).I + sj)
                            sJ = list((Qij.T).J + si)
                            sV = list((Qij.T).V)
                        subI.extend(sI)
                        subJ.extend(sJ)
                        subV.extend(sV)
                    obj = obj.aff

                JV = []
                for var in obj.factors:
                    mat = obj.factors[var]
                    for j, v in zip(mat.J, mat.V):
                        JV.append((var.startIndex + j, v))
                JV = sorted(JV)
                J = [ji for (ji, _) in JV]
                V = [vi for (_, vi) in JV]

                if self.options['handleBarVars']:
                    J, V, mats = self._separate_linear_cons(J, V, idxsdpvars)

                    for imat, mat in enumerate(mats):
                        if mat:
                            matij = task.appendsparsesymmat(
                                mat.size[0],
                                mat.I, mat.J, mat.V)
                            task.putbarcj(indsdpvar[imat], [matij], [1.0])
                    if subI:
                        subI, subV, mat2 = self._separate_linear_cons(
                            subI, subV, idxsdpvars)
                        subJ, _, mat3 = self._separate_linear_cons(
                            subJ, [0.] * len(subJ), idxsdpvars)
                        if (any([bool(mat) for mat in mat2]) or
                                any([bool(mat) for mat in mat3])):
                            raise Exception(
                                'quads with sdp bar vars are not supported')
                for j, v in zip(J, V):
                    task.putcj(j, v)
                if subI:
                    task.putqobj(subI, subJ, subV)

        # store bound on vars (will be added in the instance at the end)
        vbnds = {}
        for varname in self.varNames:
            var = self.variables[varname]
            if 'mosek' not in var.passed:
                var.passed.append('mosek')
            else:  # retrieve current bounds
                sz = var.endIndex - var.startIndex
                si = var.startIndex

                if self.options['handleBarVars']:
                    if var.semiDef:
                        continue  # this is a bar var so it has no bounds in the mosek instance
                    si, _, _ = self._separate_linear_cons(
                        [si], [0], idxsdpvars)
                    si = si[0]
                bk, bl, bu = [0.] * sz, [0.] * sz, [0.] * sz
                task.getboundslice(mosek.accmode.var, si, si + sz, bk, bl, bu)
                for ind, (ky, l, u) in enumerate(zip(bk, bl, bu)):
                    if ky is mosek.boundkey.lo:
                        vbnds[var.startIndex + ind] = (l, None)
                    elif ky is mosek.boundkey.up:
                        vbnds[var.startIndex + ind] = (None, u)
                    elif ky is mosek.boundkey.fr:
                        pass
                    else:  # fx or ra
                        vbnds[var.startIndex + ind] = (l, u)

            for ind, (lo, up) in six.iteritems(var.bnd):
                (clo, cup) = vbnds.get(var.startIndex + ind, (None, None))
                if lo is None:
                    lo = -INFINITY
                if up is None:
                    up = INFINITY
                if clo is None:
                    clo = -INFINITY
                if cup is None:
                    cup = INFINITY
                nlo = max(clo, lo)
                nup = min(cup, up)
                if nlo <= -INFINITY:
                    nlo = None
                if nup >= INFINITY:
                    nup = None
                vbnds[var.startIndex + ind] = (nlo, nup)

        for j in range(NUMVAR):
            # make the variables free
            task.putbound(mosek.accmode.var, j, mosek.boundkey.fr, 0., 0.)

        if not(self.is_continuous()) and self.options['hotstart']:
            # Set status of all variables to unknown
            if not version7:
                task.makesolutionstatusunknown(mosek.soltype.itg)
            jj = []
            sv = []
            for kvar, variable in six.iteritems(self.variables):
                if variable.is_valued():
                    for i, v in enumerate(variable.value):
                        jj.append(variable.startIndex + i)
                        sv.append(v)

            if self.options['handleBarVars']:
                jj, sv, mats = self._separate_linear_cons(jj, sv, idxsdpvars)
                if any([bool(mat) for mat in mats]):
                    raise Exception('semidef vars hotstart is not supported')

            for j, v in zip(jj, sv):
                task.putsolutioni(
                    mosek.accmode.var,
                    j,
                    mosek.soltype.itg,
                    mosek.stakey.supbas,
                    v,
                    0.0, 0.0, 0.0)

        fxdvars = self.msk_fxd
        if fxdvars is None:
            fxdvars = {}

        icone = NUMVAR

        active_cones = self.msk_active_cones
        if not active_cones:
            active_cones = []
            idx_cone = 0
        else:
            idx_cone = active_cones[-1] + 1

        #iaff = NUMCON_OLD
        active_cons = self.msk_active_cons
        if not active_cons:
            active_cons = []
            iaff = 0
            idx_aff = 0
        else:
            iaff = len(active_cons)
            idx_aff = active_cons[-1] + 1


        tridex = {}
        isdp = len(BARVARDIM_OLD)
        scaled_cols = self.msk_scaledcols
        if scaled_cols is None:
            scaled_cols = {}
        new_scaled_cols = []
        fxdconevars = self.msk_fxdconevars
        if fxdconevars is None:
            fxdconevars = []
        allconevars = [t[1]
                       for list_tuples in fxdconevars for t in list_tuples]
        allconevars.extend(range(NUMVAR0, NUMVAR))

        #-------------#
        # CONSTRAINTS #
        #-------------#

        for idcons, cons in enumerate(self.constraints):
            if 'mosek' in cons.passed:
                continue
            else:
                cons.passed.append('mosek')
            if cons.typeOfConstraint.startswith('lin'):
                fxdvars[idcons] = []
                # parse the (i,j,v) triple
                ijv = []
                for var, fact in six.iteritems(
                        (cons.Exp1 - cons.Exp2).factors):
                    if not isinstance(fact, cvx.base.spmatrix):
                        fact = cvx.sparse(fact)
                    sj = var.startIndex
                    ijv.extend(zip(fact.I, fact.J + sj, fact.V))
                ijvs = sorted(ijv)

                itojv = {}
                lasti = -1
                for (i, j, v) in ijvs:
                    if i == lasti:
                        itojv[i].append((j, v))
                    else:
                        lasti = i
                        itojv[i] = [(j, v)]

                # constant term
                szcons = cons.Exp1.size[0] * cons.Exp1.size[1]
                rhstmp = cvx.matrix(0., (szcons, 1))
                constant1 = cons.Exp1.constant  # None or a 1*1 matrix
                constant2 = cons.Exp2.constant
                if not constant1 is None:
                    rhstmp = rhstmp - constant1
                if not constant2 is None:
                    rhstmp = rhstmp + constant2

                for i in range(szcons):
                    jv = itojv.get(i, [])
                    J0 = [jvk[0] for jvk in jv]
                    V = [jvk[1] for jvk in jv]

                    is_fixed_var = (len(J0) == 1) and self.options['pass_simple_cons_as_bound']
                    if len(J0) == 1:
                        j0 = J0[0]
                        v0 = V[0]
                    if self.options['handleBarVars']:
                        J, V, mats = self._separate_linear_cons(
                            J0, V, idxsdpvars)
                    else:
                        J = J0
                    if len(J0) == 1 and len(J) == 0:
                        is_fixed_var = False  # this is a bound constraint on a bar var, handle as normal cons
                        if (v0 > 0 and cons.typeOfConstraint == 'lin<') or (
                                v0 < 0 and cons.typeOfConstraint == 'lin>'):
                            lo = None
                            up = rhstmp[i] / v0
                        else:
                            lo = rhstmp[i] / v0
                            up = None
                        if j0 in vbnds:
                            # we handle the cons here, so do not add it at the end
                            bdj0 = vbnds[j0]
                            if (bdj0[0] == lo) and (lo is not None):
                                if bdj0[1] is None:
                                    del vbnds[j0]
                                else:
                                    vbnds[j0] = (None, bdj0[1])
                            elif (bdj0[1] == up) and (up is not None):
                                if bdj0[0] is None:
                                    del vbnds[j0]
                                else:
                                    vbnds[j0] = (bdj0[0], None)

                    if is_fixed_var:
                        bdj0 = vbnds.get(j0, (-INFINITY, INFINITY))
                        if cons.typeOfConstraint == 'lin=':
                            fx = rhstmp[i] / v0
                            if (bdj0[0] is None or fx >= bdj0[0]) and (
                                    bdj0[1] is None or fx <= bdj0[1]):
                                vbnds[j0] = (fx, fx)
                            else:
                                raise Exception(
                                    'an equality constraint is not feasible: xx_{0} = {1}'.format(
                                        j0, fx))

                        elif (v0 > 0 and cons.typeOfConstraint == 'lin<') or (v0 < 0 and cons.typeOfConstraint == 'lin>'):
                            up = rhstmp[i] / v0
                            if bdj0[1] is None or up < bdj0[1]:
                                vbnds[j0] = (bdj0[0], up)
                        else:
                            lo = rhstmp[i] / v0
                            if bdj0[0] is None or lo > bdj0[0]:
                                vbnds[j0] = (lo, bdj0[1])

                        if cons.typeOfConstraint == 'lin>':
                            # and constraint handled as a bound
                            fxdvars[idcons].append((i, J[0], -V[0]))
                        else:
                            fxdvars[idcons].append((i, J[0], V[0]))

                        if cons.Id is None:
                            cons.Id = {}
                        cons.Id.setdefault('mosek',[])
                        cons.Id['mosek'].append(('var',J[0]))

                        NUMCON -= 1
                        # remove one unnecessary constraint at the end
                        if version7:
                            task.removecons([NUMCON])
                        else:
                            task.remove(mosek.accmode.con, [NUMCON])

                    else:
                        b = rhstmp[i]
                        if version7:
                            try:
                                task.putarow(iaff, J, V)
                            except:
                                import pdb;pdb.set_trace()
                        else:
                            task.putaijlist([iaff] * len(J), J, V)
                        if self.options['handleBarVars']:
                            for imat, mat in enumerate(mats):
                                if mat:
                                    matij = task.appendsparsesymmat(
                                        mat.size[0],
                                        mat.I, mat.J, mat.V)
                                    task.putbaraij(
                                        iaff, indsdpvar[imat], [matij], [1.0])

                        if cons.typeOfConstraint[3] == '=':
                            task.putbound(
                                mosek.accmode.con, iaff, mosek.boundkey.fx, b, b)
                        elif cons.typeOfConstraint[3] == '<':
                            task.putbound(
                                mosek.accmode.con, iaff, mosek.boundkey.up, -INFINITY, b)
                        elif cons.typeOfConstraint[3] == '>':
                            task.putbound(
                                mosek.accmode.con, iaff, mosek.boundkey.lo, b, INFINITY)

                        if cons.Id is None:
                            cons.Id = {}
                        cons.Id.setdefault('mosek',[])
                        cons.Id['mosek'].append(idx_aff)
                        active_cons.append(idx_aff)

                        iaff += 1
                        idx_aff += 1

            # conic constraints:
            elif cons.typeOfConstraint.endswith('cone'):

                conexp = (cons.Exp2 // cons.Exp1[:])
                if cons.Exp3:
                    conexp = ((cons.Exp3 / 2.) // conexp)

                # parse the (i,j,v) triple
                ijv = []
                for var, fact in six.iteritems(conexp.factors):
                    if not isinstance(fact, cvx.base.spmatrix):
                        fact = cvx.sparse(fact)
                    sj = var.startIndex
                    ijv.extend(zip(fact.I, fact.J + sj, fact.V))
                ijvs = sorted(ijv)

                itojv = {}
                lasti = -1
                for (i, j, v) in ijvs:
                    if i == lasti:
                        itojv[i].append((j, v))
                    else:
                        lasti = i
                        itojv[i] = [(j, v)]

                # add new eq. cons
                szcons = conexp.size[0] * conexp.size[1]
                rhstmp = conexp.constant
                if rhstmp is None:
                    rhstmp = cvx.matrix(0., (szcons, 1))

                istart = icone
                fxd = []
                conevars = []
                for i in range(szcons):
                    jv = itojv.get(i, [])
                    J = [jvk[0] for jvk in jv]
                    V = [-jvk[1] for jvk in jv]
                    h = rhstmp[i]
                    if self.options['handleBarVars']:
                        J, V, mats = self._separate_linear_cons(
                            J, V, idxsdpvars)
                        for imat, mat in enumerate(mats):
                            if mat:
                                matij = task.appendsparsesymmat(
                                    mat.size[0],
                                    mat.I, mat.J, mat.V)
                                task.putbaraij(
                                    iaff, indsdpvar[imat], [matij], [1.0])
                    else:  # for algorithmic simplicity
                        mats = [0]
                    # do we add the variable directly in a cone ?
                    if (self.options['handleConeVars'] and
                        len(J) == 1 and  # a single var in the expression
                        J[0] not in allconevars and  # not in a cone yet
                        # no coef on bar vars
                                not(any([mat for mat in mats])) and
                                h == 0 and  # no constant term
                                #(V[0]==-1 or (J[0] not in ints)) #commented (int vars in cone yield a bug with mosek <6.59)
                            J[0] not in ints  # int. variables cannot be scaled
                        ):
                        conevars.append(J[0])
                        allconevars.append(J[0])
                        fxd.append((i, J[0]))
                        if V[0] != -1:
                            scaled_cols[J[0]] = -V[0]
                            new_scaled_cols.append(J[0])

                    else:  # or do we need an extra variable equal to this expression ?
                        J.append(icone)
                        V.append(1)
                        if version7:
                            task.appendcons(1)
                            task.appendvars(1)
                        else:
                            task.append(mosek.accmode.con, 1)
                            task.append(mosek.accmode.var, 1)
                        NUMCON += 1
                        NUMVAR += 1
                        if version7:
                            task.putarow(iaff, J, V)
                        else:
                            task.putaijlist([iaff] * len(J), J, V)
                        task.putbound(
                            mosek.accmode.con, iaff, mosek.boundkey.fx, h, h)
                        conevars.append(icone)

                        if cons.Id is None:
                            cons.Id = {}
                        cons.Id.setdefault('mosek',[])
                        cons.Id['mosek'].append(idx_aff)
                        active_cons.append(idx_aff)

                        iaff += 1
                        idx_aff += 1
                        icone += 1
                iend = icone
                # sk in quadratic cone
                if cons.Exp3:
                    task.appendcone(mosek.conetype.rquad, 0.0, conevars)
                else:
                    task.appendcone(mosek.conetype.quad, 0.0, conevars)

                if cons.Id is None:
                    cons.Id = {}
                cons.Id.setdefault('mosek',[])
                cons.Id['mosek'].append(('cone',idx_cone))
                active_cones.append(idx_cone)
                idx_cone += 1

                for j in range(istart, iend):  # make extra variable free
                    task.putbound(
                        mosek.accmode.var, j, mosek.boundkey.fr, 0., 0.)
                fxdconevars.append(fxd)

            # SDP constraints:
            elif cons.typeOfConstraint.startswith('sdp'):
                if cons.Id is None:
                        cons.Id = {}
                cons.Id.setdefault('mosek',[])
                cons.Id['mosek'].append(('sdp',isdp))

                if self.options['handleBarVars'] and cons.semidefVar:
                    isdp += 1
                    continue

                szk = BARVARDIM[isdp]
                if szk not in tridex:
                    # tridex[szk] contains a list of all tuples of the form
                    #(E_ij,index(ij)),
                    # where ij is an index of a element in the lower triangle
                    # E_ij is the symm matrix s.t. <Eij|X> = X_ij
                    # and index(ij) is the index of the pair(ij) counted in
                    # column major order
                    tridex[szk] = []
                    idx = -1
                    for j in range(szk):
                        for i in range(szk):
                            idx += 1
                            if i >= j:  # (in lowtri)
                                if i == j:
                                    subi = [i]
                                    subj = [i]
                                    val = [1.]
                                else:
                                    subi = [i]
                                    subj = [j]
                                    val = [0.5]
                                Eij = task.appendsparsesymmat(
                                    BARVARDIM[isdp],
                                    subi, subj, val)
                                tridex[szk].append((Eij, idx))

                if cons.typeOfConstraint == 'sdp<':
                    sdexp = (cons.Exp2 - cons.Exp1)
                else:
                    sdexp = (cons.Exp1 - cons.Exp2)

                # parse the (i,j,v) triple
                ijv = []
                for var, fact in six.iteritems(sdexp.factors):
                    if not isinstance(fact, cvx.base.spmatrix):
                        fact = cvx.sparse(fact)
                    sj = var.startIndex
                    ijv.extend(zip(fact.I, fact.J + sj, fact.V))
                ijvs = sorted(ijv)

                itojv = {}
                lasti = -1
                for (i, j, v) in ijvs:
                    if i == lasti:
                        itojv[i].append((j, v))
                    else:
                        lasti = i
                        itojv[i] = [(j, v)]

                szcons = sdexp.size[0] * sdexp.size[1]
                rhstmp = sdexp.constant
                if rhstmp is None:
                    rhstmp = cvx.matrix(0., (szcons, 1))

                szsym = (szk * (szk + 1)) // 2
                if version7:
                    task.appendcons(szsym)
                else:
                    task.append(mosek.accmode.con, szsym)
                NUMCON += szsym
                for (Eij, idx) in tridex[szk]:
                    J = [jvk[0] for jvk in itojv.get(idx, [])]
                    V = [-jvk[1] for jvk in itojv.get(idx, [])]
                    h = rhstmp[idx]
                    if self.options['handleBarVars']:
                        J, V, mats = self._separate_linear_cons(
                            J, V, idxsdpvars)
                        for imat, mat in enumerate(mats):
                            if mat:
                                matij = task.appendsparsesymmat(
                                    mat.size[0],
                                    mat.I, mat.J, mat.V)
                                task.putbaraij(
                                    iaff, indsdpvar[imat], [matij], [1.0])

                    if J:
                        task.putarow(iaff, J, V)
                    task.putbaraij(iaff, isdp, [Eij], [1.0])
                    task.putbound(
                        mosek.accmode.con, iaff, mosek.boundkey.fx, h, h)

                    if cons.Id is None:
                            cons.Id = {}
                    cons.Id.setdefault('mosek',[])
                    cons.Id['mosek'].append(idx_aff)
                    active_cons.append(idx_aff)

                    iaff += 1
                    idx_aff += 1
                isdp += 1
            # quadratic constraints:
            elif cons.typeOfConstraint == 'quad':
                subI = []
                subJ = []
                subV = []
                # quad part

                qexpr = cons.Exp1
                for i, j in qexpr.quad:
                    si, ei = i.startIndex, i.endIndex
                    sj, ej = j.startIndex, j.endIndex
                    Qij = qexpr.quad[i, j]
                    if not isinstance(Qij, cvx.spmatrix):
                        Qij = cvx.sparse(Qij)
                    if si == sj:  # put A+A' on the diag
                        sI = list((Qij + Qij.T).I + si)
                        sJ = list((Qij + Qij.T).J + sj)
                        sV = list((Qij + Qij.T).V)
                        for u in range(len(sI) - 1, -1, -1):
                            # remove when j>i
                            if sJ[u] > sI[u]:
                                del sI[u]
                                del sJ[u]
                                del sV[u]
                    elif si >= ej:  # add A in the lower triang
                        sI = list(Qij.I + si)
                        sJ = list(Qij.J + sj)
                        sV = list(Qij.V)
                    else:  # add B' in the lower triang
                        sI = list((Qij.T).I + sj)
                        sJ = list((Qij.T).J + si)
                        sV = list((Qij.T).V)
                    subI.extend(sI)
                    subJ.extend(sJ)
                    subV.extend(sV)
                # aff part
                J = []
                V = []
                for var in qexpr.aff.factors:
                    mat = qexpr.aff.factors[var]
                    for j, v in zip(mat.J, mat.V):
                        V.append(v)
                        J.append(var.startIndex + j)

                if self.options['handleBarVars']:
                    J, V, mats = self._separate_linear_cons(J, V, idxsdpvars)
                    subI, subV, mat2 = self._separate_linear_cons(
                        subI, subV, idxsdpvars)
                    subJ, _, mat3 = self._separate_linear_cons(
                        subJ, [0.] * len(subJ), idxsdpvars)

                    if (any([bool(mat) for mat in mats]) or
                            any([bool(mat) for mat in mat2]) or
                            any([bool(mat) for mat in mat3])):
                        raise Exception(
                            'quads with sdp bar vars are not supported')

                rhs = qexpr.aff.constant
                if rhs is None:
                    rhs = 0.
                else:
                    rhs = -rhs[0]

                if J:
                    if version7:
                        task.putarow(iaff, J, V)
                    else:
                        task.putaijlist([iaff] * len(J), J, V)
                task.putqconk(iaff, subI, subJ, subV)
                task.putbound(mosek.accmode.con, iaff,
                              mosek.boundkey.up, -INFINITY, rhs)

                if cons.Id is None:
                    cons.Id = {}
                cons.Id.setdefault('mosek',[])
                cons.Id['mosek'].append(idx_aff)
                active_cons.append(idx_aff)

                iaff += 1
                idx_aff += 1

            else:
                raise Exception(
                    'type of constraint not handled (yet ?) for mosek:{0}'.format(
                        cons.typeOfConstraint))

        # bounds on vars and bar vars
        bndjj = []
        bndlo = []
        bndup = []
        for jj in sorted(vbnds.keys()):
            bndjj.append(jj)
            (lo, up) = vbnds[jj]
            if lo is None:
                lo = -INFINITY
            if up is None:
                up = INFINITY
            bndlo.append(lo)
            bndup.append(up)

        if self.options['handleBarVars']:
            _, bndlo, matslo = self._separate_linear_cons(
                bndjj, bndlo, idxsdpvars)
            bndjj, bndup, matsup = self._separate_linear_cons(
                bndjj, bndup, idxsdpvars)
        for j, lo, up in zip(bndjj, bndlo, bndup):
            if up >= INFINITY:
                task.putbound(
                    mosek.accmode.var,
                    j,
                    mosek.boundkey.lo,
                    lo,
                    INFINITY)
            elif lo <= -INFINITY:
                task.putbound(
                    mosek.accmode.var, j, mosek.boundkey.up, -INFINITY, up)
            elif up == lo:
                task.putbound(mosek.accmode.var, j, mosek.boundkey.fx, lo, lo)
            else:
                task.putbound(mosek.accmode.var, j, mosek.boundkey.ra, lo, up)

        if self.options['handleBarVars']:
            # bounds on bar vars by taking the matslo and matsup one by one
            for imat, (mlo, mup) in enumerate(zip(matslo, matsup)):
                for (i, j, v) in zip(mlo.I, mlo.J, mlo.V):
                    if i == j:
                        matij = task.appendsparsesymmat(
                            mlo.size[0],
                            [i], [i], [1.])
                        lo = v
                        up = mup[i, j]
                    else:
                        matij = task.appendsparsesymmat(
                            mlo.size[0],
                            [i], [j], [0.5])
                        lo = v #* (2**0.5) [no sqrt(2), because hard-coded bounds have already this factor saved internally]
                        up = mup[i, j] #* (2**0.5)

                    if version7:
                        task.appendcons(1)
                    else:
                        task.append(mosek.accmode.con, 1)
                    NUMCON += 1

                    task.putbaraij(iaff, indsdpvar[imat], [matij], [1.0])
                    if lo <= -INFINITY/1.42: #because infinity, however, is not scaled wrt sqrt(2)
                        task.putbound(
                            mosek.accmode.con, iaff, mosek.boundkey.up, -INFINITY, up)
                    elif up >= INFINITY/1.42:
                        task.putbound(
                            mosek.accmode.con, iaff, mosek.boundkey.lo, lo, INFINITY)
                    else:
                        task.putbound(
                            mosek.accmode.con, iaff, mosek.boundkey.ra, lo, up)
                    active_cons.append(iaff)
                    iaff += 1


        # scale columns of variables in cones (simple change of variable which
        # avoids adding an extra var)
        for (j, v) in six.iteritems(scaled_cols):
            sj = [0] * NUMCON
            vj = [0] * NUMCON
            isnewcone = j in new_scaled_cols
            if version7:  # scale all terms if this is a new cone, only the new rows otherwise
                nzj = task.getacol(j, sj, vj)
                task.putacol(j,
                             sj[:nzj],
                             [(vji / v if (isnewcone or i >= NUMCON_OLD) else vji) for (i,
                                                                                        vji) in zip(sj[:nzj],
                                                                                                    vj[:nzj])])
            else:
                nzj = task.getavec(mosek.accmode.var, j, sj, vj)
                task.putavec(mosek.accmode.var, j, sj[:nzj], [
                             (vji / v if (isnewcone or i >= NUMCON_OLD) else vji) for (i, vji) in zip(sj[:nzj], vj[:nzj])])

            if newobj or isnewcone:  # scale the objective coef
                cj = [0.]
                task.getcslice(j, j + 1, cj)
                task.putcj(j, cj[0] / v)


        # constraints deletion

        print_message_not_printed_yet = True
        cones_to_remove = []
        aff_to_remove = []
        for cs in self._deleted_constraints:
            if 'mosek' in cs.passed:
                continue
            else:
                cs.passed.append('mosek')

            if print_message_not_printed_yet and self.options['verbose'] > 0:
                print()
                print('Removing constraints from Mosek instance...')
                print_message_not_printed_yet = False


                warning_message_not_printed_yet = False

            if (cs.Id is None) or ('mosek' not in cs.Id):
                raise NotImplementedError("\033[1;31m*** You tried to remove a constraint from a problem that is solve via its dual. This is not implemented (yet)."+
                                                  "Try to set the option ``solve_via_dual`` to False"+
                                                  "For now, you can run ``reset_solver_instances()`` and solve the problem again.\033[0m")

            for i in cs.Id['mosek']:
                if isinstance(i,tuple):
                    if i[0]=='var':
                        raise NotImplementedError("\033[1;31m*** You have been removing a constraint that can be "+
                            "(partly) interpreted as a variable bound. This is not safe when "+
                            "the option ``pass_simple_cons_as_bound`` is set to True."+
                            "Try to run ``reset_solver_instances()`` and solve the problem again.\033[0m")
                    elif i[0]=='sdp':
                        raise NotImplementedError("\033[1;31m*** You tried to remove an sdp constraint. This is not implemented (yet)."+
                                                  "Try to run ``reset_solver_instances()`` and solve the problem again.\033[0m")
                    elif i[0]=='cone':
                        cones_to_remove.append(i[1])
                else:
                    aff_to_remove.append(i)


        idx_aff_to_remove = np.searchsorted(active_cons,aff_to_remove)
        task.removecons(idx_aff_to_remove)
        for ii in idx_aff_to_remove[::-1]:
            del active_cons[ii]

        idx_cones_to_remove = np.searchsorted(active_cones,cones_to_remove)
        task.removecones(idx_cones_to_remove)
        for ii in idx_cones_to_remove[::-1]:
            del active_cones[ii]


        # objective sense
        if self.objective[0] == 'max':
            task.putobjsense(mosek.objsense.maximize)
        else:
            task.putobjsense(mosek.objsense.minimize)

        self.msk_env = env
        self.msk_task = task
        self.msk_fxd = fxdvars
        self.msk_scaledcols = scaled_cols
        self.msk_fxdconevars = fxdconevars
        self.msk_active_cones = active_cones
        self.msk_active_cons = active_cons

        if reset_hbv_True:
            self.options._set('handleBarVars', True)

        if self.options['verbose'] > 0:
            print('mosek instance built')

    def _make_sdpaopt(self, sdpa_executable='sdpa'):
        """
        Defines the variables sdpa_executable, sdpa_dats_filename, and
        sdpa_out_filename used by the sdpa solver.
        """
        if any([('sdpa' not in cs.passed) for cs in self._deleted_constraints]):
            for cs in self._deleted_constraints:
                if 'sdpa' not in cs.passed:
                    cs.passed.append('sdpa')
            self.reset_sdpa_instance(True)

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

        self.sdpa_executable = sdpa_executable
        if which(sdpa_executable) is None:
            raise OSError(sdpa_executable + " is not in the path")

        import tempfile
        tempfile_ = tempfile.NamedTemporaryFile()
        tmp_filename = tempfile_.name
        tempfile_.close()
        self.sdpa_dats_filename = tmp_filename + ".dat-s"
        self.sdpa_out_filename = tmp_filename + ".out"
        self._write_sdpa(self.sdpa_dats_filename)

    def _convert_picos_exp_to_scip_exp(self,expression):
        """
        input: picos Affine Expression or None (expression = 0 in this case)
        returns: list of (scalar) scip expression (one for each coordinate of the Affine expression given in input
        """
        import pyscipopt
        if hasattr(pyscipopt,'linexpr') and hasattr(pyscipopt.scip,'LinExpr'):
            scip_expr = pyscipopt.linexpr.LinExpr
        else:
            scip_expr = pyscipopt.scip.Expr
            
        if expression is None:
            lhs = [scip_expr()]
        elif isinstance(expression,QuadExp):
            lhs = self._convert_picos_exp_to_scip_exp(expression.aff)
            for var1,var2 in expression.quad:
                Q = expression.quad[var1,var2]
                start_index_1 = var1.scip_startIndex
                start_index_2 = var2.scip_startIndex
                for i,j,v in zip(Q.I, Q.J, Q.V):
                    lhs[0] += v*self.scip_vars[i+start_index_1] * self.scip_vars[j+start_index_2]

        else:
            lhs = []
            for _ in range(expression.size[0]*expression.size[1]):
                lhs.append(scip_expr())
            for variable in expression.factors:
                start_index = variable.scip_startIndex
                for i,j,v in zip(expression.factors[variable].I, expression.factors[variable].J, expression.factors[variable].V):
                    lhs[i] += v*self.scip_vars[j+start_index]
            if expression.constant:
                for i in range(expression.size[0]*expression.size[1]):
                    lhs[i] +=  expression.constant[i]
        return lhs

    def _make_zibopt(self):
        """
        Defines the variables scip_solver, scip_vars and scip_obj,
        used by the zibopt solver.
        """
        if any([('scip' not in cs.passed) for cs in self._deleted_constraints]):
            for cs in self._deleted_constraints:
                if 'scip' not in cs.passed:
                    cs.passed.append('scip')
            self.reset_scip_instance(True)

        try:
            import pyscipopt
        except:
            raise ImportError('scip library not found')

        obj_sense, obj_exp = self.objective
        if isinstance(obj_exp,QuadExp):
            self.convert_quadobj_to_constraint()
            obj_sense, obj_exp = self.objective

        if (self.scip_model is None):
            self.scip_model = pyscipopt.Model()
            self.scip_vars = []
            current_index = 0
        else:
            current_index = self.scip_var_index


        picvtype = 'None'
        current_index = 0

        for name, variable in six.iteritems(self.variables):
            if 'scip' in variable.passed:
                continue

            variable.passed.append('scip')
            variable.scip_startIndex = current_index
            sz = variable.size[0]*variable.size[1]
            for i in range(sz):
                INFINITY_SCIP=1e14
                (li,ui) = variable.bnd.get(i,(None,None))
                if li is None:
                    li = -INFINITY_SCIP
                if ui is None:
                    ui = INFINITY_SCIP
                if variable.vtype == 'binary':
                    self.scip_vars.append(self.scip_model.addVar(name+'_'+str(i),vtype='I',lb=0,ub=1))
                elif variable.vtype == 'integer':
                    li = int(np.ceil(li))
                    ui = int(np.floor(ui))
                    self.scip_vars.append(self.scip_model.addVar(name+'_'+str(i),vtype='I',lb=li,ub=ui))
                else:
                    self.scip_vars.append(self.scip_model.addVar(name+'_'+str(i),lb=li,ub=ui))
            current_index += sz
            self.scip_var_index = current_index

        for cons in self.constraints:
            if 'scip' in cons.passed:
                continue
            else:
                cons.passed.append('scip')

            if cons.typeOfConstraint[:3]=='lin':
                expression = cons.Exp1 - cons.Exp2
                lhs = self._convert_picos_exp_to_scip_exp(expression)
                for lhsi in lhs:
                    if cons.typeOfConstraint[3]=='<':
                        self.scip_model.addCons(lhsi <= 0)
                    elif cons.typeOfConstraint[3]=='>':
                        self.scip_model.addCons(lhsi >= 0)
                    elif cons.typeOfConstraint[3]=='=':
                        self.scip_model.addCons(lhsi == 0)
                    else:
                        raise ValueError('unknown type of constraint: '+cons.typeOfConstraint)
            elif cons.typeOfConstraint == 'quad':
                expression = cons.Exp1
                lhs = self._convert_picos_exp_to_scip_exp(expression)
                for lhsi in lhs:
                     self.scip_model.addCons(lhsi <= 0)
            elif cons.typeOfConstraint.endswith('cone'):
                if cons.typeOfConstraint[:2]=='RS':
                    expression = cons.Exp1.T*cons.Exp1 - cons.Exp2*cons.Exp3
                elif cons.typeOfConstraint[:2]=='SO':
                    expression = cons.Exp1.T*cons.Exp1 - cons.Exp2*cons.Exp2
                else:
                    raise ValueError('unknown type of constraint: '+cons.typeOfConstraint)
                lhs = self._convert_picos_exp_to_scip_exp(expression)
                for lhsi in lhs:
                     self.scip_model.addCons(lhsi <= 0)
            else:
                raise NotImplementedError('not implemented yet')

        if obj_exp is None or isinstance(obj_exp,AffinExp):
            self.scip_obj = self._convert_picos_exp_to_scip_exp(obj_exp)[0]
        else:
            raise NotImplementedError('the scip interface does not allow non-linear objective functions (yet). Try to add an auxiliary variable t and use the epigraph form min t: f(x)<=t')

        if obj_sense == 'max':
           self.scip_model.setObjective(self.scip_obj,'maximize')
        elif obj_sense in ('min','find'):
           self.scip_model.setObjective(self.scip_obj,'minimize')
        else:
           raise ValueError('unknown type of objective sense: '+obj_sense)


    """
    -----------------------------------------------
    --                CALL THE SOLVER            --
    -----------------------------------------------
    """

    def minimize(self, obj, **options):
        """
        sets the objective ('min',``obj``) and calls the function :func:`solve() <picos.Problem.solve>` .
        """
        self.set_objective('min', obj)
        return self.solve(**options)

    def maximize(self, obj, **options):
        """
        sets the objective ('max',``obj``) and calls the function :func:`solve() <picos.Problem.solve>` .
        """
        self.set_objective('max', obj)
        return self.solve(**options)

    def solve(self, **options):
        """
        Solves the problem.

        Once the problem has been solved, the optimal variables
        can be obtained thanks to the property :attr:`value <picos.Expression.value>`
        of the class :class:`Expression<picos.Expression>`.
        The optimal dual variables can be accessed by the property
        :attr:`dual <picos.Constraint.dual>` of the class
        :class:`Constraint<picos.Constraint>`.

        :keyword options: A list of options to update before
                          the call to the solver. In particular,
                          the solver can
                          be specified here,
                          under the form ``key = value``.
                          See the list of available options
                          in the doc of
                          :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`
        :returns: A dictionary which contains the objective value of the problem,
                  the time used by the solver, the status of the solver, and an object
                  which depends on the solver and contains information about the solving process.

        """
        if options is None:
            options = {}
        self.update_options(**options)
        if self.options['solver'] is None:
            self.solver_selection()

        # self._eliminate_useless_variables()

        if isinstance(self.objective[1], GeneralFun):
            return self._sqpsolve(options)

        # is it a complex SDP that we must transform to a real problem ?
        complexSDP = self.is_complex()

        if not complexSDP:
            # do we pass the primal or the dual to the solver ?
            solve_via_dual = self.options['solve_via_dual']
            if solve_via_dual is None\
                  and 'read_solution' not in self.options['sdpa_params']:
                if (self.numberSDPConstraints > 0 and len([1 for v in self.variables.values(
                ) if v.semiDef]) < 0.3 * self.numberSDPConstraints):  # thats empirical !
                    solve_via_dual = True
                else:
                    solve_via_dual = False

        # transform the problem in case of a complex SDP
        if complexSDP:
            if self.options['verbose'] > 0:
                print('*** Making the problem real...  ***')

            realP = self.to_real()
            if self.options['verbose'] > 0:
                print(
                    '*** OK, solve the real problem and transform the solution as in the original problem...  ***')
            sol = realP.solve()
            obj = sol['obj']
            if 'noprimals' in self.options and self.options['noprimals']:
                pass
            else:
                primals = {}
                for var in self.variables.values():
                    if var.vtype == 'hermitian':
                        Zi = realP.get_valued_variable(var.name + '_IM')
                        primals[
                            var.name] = realP.get_valued_variable(
                            var.name + '_RE') + 1j * Zi
                    elif var.vtype == 'complex':
                        Zr = realP.get_valued_variable(var.name + '_RE')
                        Zi = realP.get_valued_variable(var.name + '_IM')
                        primals[var.name] = Zr + 1j * Zi
                    else:
                        primals[var.name] = realP.get_valued_variable(var.name)

            if 'noduals' in self.options and self.options['noduals']:
                pass
            else:
                duals = []
                for cst in realP.constraints:
                    if cst.typeOfConstraint.startswith('sdp'):
                        n = int(cst.dual.size[0] / 2.)
                        if cst.dual.size[1] != cst.dual.size[0]:
                            raise Exception('Dual must be square matrix')
                        if cst.dual.size[1] != 2 * n:
                            raise Exception('dims must be even numbers')
                        F1 = cst.dual[:n, :n]
                        F1a = cst.dual[n:, n:]
                        F2a = cst.dual[:n, n:]
                        duals.append((F1 + 1j * F2a) + (F1a + 1j * F2a).H)
                    else:
                        duals.append(cst.dual)

        # solve the dual problem instead
        elif solve_via_dual:
            converted = False
            raiseexp = False
            remove_cons_exp = False
            try:
                if self.options['verbose'] > 0:
                    print('*** Dualizing the problem...  ***')
                dual = self.dualize()
                if self.options['solver'].startswith('mosek') and any([('mosek' not in cs.passed) for cs in self._deleted_constraints]):
                    remove_cons_exp = True
                    for cs in self._deleted_constraints:
                        if 'mosek' not in cs.passed:
                            cs.passed.append('mosek')
                        self.reset_mosek_instance(True)
                    raise NotImplementedError("\033[1;31m*** You tried to remove a constraint from a problem that is solved via its dual. This is not implemented (yet)."+
                                  "The mosek instance has been reset. "+
                                  "Now, you can retry to solve it.\033[0m")
            except QuadAsSocpError as ex:
                if self.options['convert_quad_to_socp_if_needed']:
                    pcop = self.copy()
                    try:
                        pcop.convert_quad_to_socp()
                        converted = True
                        dual = pcop.dualize()
                    except NonConvexError as ex:
                        raiseexp = True
                else:
                    raiseexp = True
            except Exception as ex:
                raiseexp = True
            finally:
                # if nonconvex:
                    #raise NonConvexError('Problem is nonconvex')
                # if nosocpquad:
                    # raise QuadAsSocpError('Try to convert the quadratic constraints as cone constraints '+
                        #'with the function convert_quad_to_socp().')
                if remove_cons_exp:
                    raise(ex)
                if raiseexp:
                    # raise(ex)
                    if self.options['verbose'] > 0:
                        print(
                            "\033[1;31m Error raised when dualizing the problem: \033[0m")
                        print(ex)
                        print ('I retry to solve without dualizing')
                    return self.solve(solve_via_dual=False)

                for cs in self.constraints:
                    if self.options['solver'].startswith('mosek'):
                        cs.passed.append('mosek')
                    elif self.options['solver'] in ('cvxopt','smcp'):
                        cs.passed.append('cvxopt')
                    elif self.options['solver']=='sdpa':
                        cs.passed.append('sdpa')

                sol = dual.solve()
                if self.objective[0] == 'min':
                    obj = sol['obj']
                else:
                    obj = -sol['obj']
                if 'noprimals' in self.options and self.options['noprimals']:
                    pass
                else:
                    primals = {}
                    # the primal variables are the duals of the dual (up to a
                    # sign)
                    xx = dual.constraints[-1].dual
                    if xx is None:
                        if self.options['verbose'] > 0:
                            raise Exception(
                                "\033[1;31m no Primals retrieved from the dual problem \033[0m")
                    else:
                        xx = -xx
                        indices = [(v.startIndex, v.endIndex, v)
                                   for v in self.variables.values()]
                        indices = sorted(indices, reverse=True)

                        (start, end, var) = indices.pop()
                        varvect = []
                        if converted:
                            xx = xx[:-1]
                        for i, x in enumerate(xx):
                            if i < end:
                                varvect.append(x)
                            else:
                                if var.vtype in ('symmetric',):
                                    varvect = svecm1(cvx.matrix(varvect))
                                primals[var.name] = cvx.matrix(
                                    varvect, var.size)
                                varvect = [x]
                                (start, end, var) = indices.pop()
                        if var.vtype in ('symmetric',):
                            varvect = svecm1(cvx.matrix(varvect))
                        primals[var.name] = cvx.matrix(varvect, var.size)


                if converted:
                    self.set_option('noduals', True)
                if 'noduals' in self.options and self.options['noduals']:
                    pass
                else:
                    duals = []
                    icone = 0  # cone index
                    isdp = 0  # semidef index
                    if 'mue' in dual.variables:
                        eqiter = iter(dual.get_valued_variable('mue'))
                    if 'mul' in dual.variables:
                        initer = iter(dual.get_valued_variable('mul'))
                    for cons in self.constraints:
                        if cons.typeOfConstraint[2:] == 'cone':
                            z = dual.get_valued_variable(
                                'zs[{0}]'.format(icone))
                            lb = dual.get_valued_variable(
                                'lbda[{0}]'.format(icone))
                            duals.append(cvx.matrix([lb, z]))
                            icone += 1
                        elif cons.typeOfConstraint == 'lin=':
                            szcons = cons.Exp1.size[0] * cons.Exp1.size[1]
                            dd = []
                            for i in range(szcons):
                                dd.append(six.next(eqiter))
                            duals.append(cvx.matrix(dd))
                        # lin ineq
                        elif cons.typeOfConstraint.startswith('lin'):
                            szcons = cons.Exp1.size[0] * cons.Exp1.size[1]
                            dd = []
                            for i in range(szcons):
                                dd.append(six.next(initer))
                            duals.append(cvx.matrix(dd))
                        elif cons.typeOfConstraint.startswith('sdp'):
                            X = dual.get_valued_variable('X[{0}]'.format(isdp))
                            duals.append(X)
                            isdp += 1
        else:
            try:
                # WARNING: Bug with cvxopt-mosek ?
                if (self.options['solver'] == 'CVXOPT'  # obolete name, use lower case
                        or self.options['solver'] == 'cvxopt-mosek'
                        or self.options['solver'] == 'smcp'
                        or self.options['solver'] == 'cvxopt'):
                    primals, duals, obj, sol = self._cvxopt_solve()

                # For glpk
                elif (self.options['solver'] == 'glpk'):
                    primals, duals, obj, sol = self._glpk_solve()

                # For cplex
                elif (self.options['solver'] == 'cplex'):

                    primals, duals, obj, sol = self._cplex_solve()

                # for mosek
                elif (self.options['solver'] == 'MSK'  # obsolete value, use lower case
                        or self.options['solver'] == 'mosek'
                        or self.options['solver'] == 'mosek7'
                        or self.options['solver'] == 'mosek6'):

                    primals, duals, obj, sol = self._mosek_solve()

                # for scip
                elif (self.options['solver'] in ('zibopt', 'scip')):

                    primals, duals, obj, sol = self._zibopt_solve()

                # for gurobi
                elif (self.options['solver'] == 'gurobi'):
                    primals, duals, obj, sol = self._gurobi_solve()

                # for SDPA
                elif (self.options['solver'] == 'sdpa'):
                    primals, duals, obj, sol = self._sdpa_solve()
                else:
                    raise Exception('unknown solver')
            except QuadAsSocpError:
                if self.options['convert_quad_to_socp_if_needed']:
                    pcop = self.copy()
                    pcop.convert_quad_to_socp()
                    sol = pcop.solve()
                    self.status = sol['status']
                    for vname, v in six.iteritems(self.variables):
                        v.value = pcop.get_variable(vname).value
                    for i, cs in enumerate(self.constraints):
                        dui = pcop.constraints[i].dual
                        if not(dui is None):
                            cs.set_dualVar(dui)
                    return sol
                else:
                    raise

        if 'noprimals' in self.options and self.options['noprimals']:
            pass
        else:
            for k in primals:
                if not primals[k] is None:
                    self.set_var_value(k, primals[k], optimalvar=True)

        if 'noduals' in self.options and self.options['noduals']:
            pass
        else:
            for i, d in enumerate(duals):
                self.constraints[i].set_dualVar(d)

        if obj == 'toEval' and not(self.objective[1] is None):
            obj = self.objective[1].eval()[0]
        sol['obj'] = obj
        self.status = sol['status']
        return sol

    def _cvxopt_solve(self):
        """
        Solves a problem with the cvxopt solver.
        """

        #-----------------------------#
        # Can we solve this problem ? #
        #-----------------------------#

        if self.type in ('unknown type',
                         'MISDP',
                         'MISOCP',
                         'MIQCP',
                         'MIQP',
                         'MIP',
                         'Mixed (MISOCP+quad)') and (self.options['solver'] == 'cvxopt'):
            raise NotAppropriateSolverError(
                "'cvxopt' cannot solve problems of type {0}".format(
                    self.type))

        elif self.type in ('unknown type', 'GP', 'MISDP', 'MISOCP', 'MIQCP', 'MIQP', 'MIP', 'Mixed (MISOCP+quad)') and (
                self.options['solver'] == 'smcp'):
            raise NotAppropriateSolverError(
                "'smcp' cannot solve problems of type {0}".format(
                    self.type))

        elif self.type in ('Mixed (SDP+quad)', 'Mixed (SOCP+quad)', 'QCQP', 'QP'):
            raise QuadAsSocpError(
                'Please convert the quadratic constraints as cone constraints ' +
                'with the function convert_quad_to_socp().')
        #--------------------#
        # makes the instance #
        #--------------------#

        self._make_cvxopt_instance(
            reset=False,
            new_cvxopt_cons_only=True,
            hard_coded_bounds=True)

        #--------------------#
        #  sets the options  #
        #--------------------#
        import cvxopt.solvers
        abstol = self.options['abstol']
        if abstol is None:
            abstol = self.options['tol']
        reltol = self.options['reltol']
        if reltol is None:
            reltol = 10 * self.options['tol']
        feastol = self.options['feastol']
        if feastol is None:
            feastol = self.options['tol']
        maxit = self.options['maxit']
        if maxit is None:
            maxit = 999999
        cvx.solvers.options['maxiters'] = maxit
        cvx.solvers.options['abstol'] = abstol
        cvx.solvers.options['feastol'] = feastol
        cvx.solvers.options['reltol'] = reltol
        cvx.solvers.options['show_progress'] = bool(
            self.options['verbose'] > 0)
        try:
            import smcp.solvers
            smcp.solvers.options['maxiters'] = maxit
            smcp.solvers.options['abstol'] = abstol
            smcp.solvers.options['feastol'] = feastol
            smcp.solvers.options['reltol'] = reltol
            smcp.solvers.options['show_progress'] = bool(
                self.options['verbose'] > 0)
        except:
            #smcp is not available
            pass

        if self.options['solver'].upper() == 'CVXOPT':
            currentsolver = None
        elif self.options['solver'] == 'cvxopt-mosek':
            currentsolver = 'mosek'
        elif self.options['solver'] == 'smcp':
            currentsolver = 'smcp'
        #-------------------------------#
        #  runs the appropriate solver  #
        #-------------------------------#
        import time
        tstart = time.time()

        if self.numberLSEConstraints > 0:  # GP
            probtype = 'GP'
            if self.options['verbose'] > 0:
                print('-----------------------------------')
                print('         cvxopt GP solver')
                print('-----------------------------------')
            sol = cvx.solvers.gp(self.cvxoptVars['K'],
                                 self.cvxoptVars['F'], self.cvxoptVars['g'],
                                 self.cvxoptVars['Gl'], self.cvxoptVars['hl'],
                                 self.cvxoptVars['A'], self.cvxoptVars['b'])
        # changes to adapt the problem for the conelp interface:
        elif currentsolver == 'mosek':
            if len(self.cvxoptVars['Gs']) > 0:
                raise Exception('CVXOPT does not handle SDP with MOSEK')
            if len(self.cvxoptVars['Gq']) + len(self.cvxoptVars['Gs']):
                if self.options['verbose'] > 0:
                    print('------------------------------------------')
                    print('  mosek LP solver interfaced by cvxopt')
                    print('------------------------------------------')
                sol = cvx.solvers.lp(
                    self.cvxoptVars['c'],
                    self.cvxoptVars['Gl'],
                    self.cvxoptVars['hl'],
                    self.cvxoptVars['A'],
                    self.cvxoptVars['b'],
                    solver=currentsolver)
                probtype = 'LP'
            else:
                if self.options['verbose'] > 0:
                    print('-------------------------------------------')
                    print('  mosek SOCP solver interfaced by cvxopt')
                    print('-------------------------------------------')
                sol = cvx.solvers.socp(
                    self.cvxoptVars['c'],
                    self.cvxoptVars['Gl'],
                    self.cvxoptVars['hl'],
                    self.cvxoptVars['Gq'],
                    self.cvxoptVars['hq'],
                    self.cvxoptVars['A'],
                    self.cvxoptVars['b'],
                    solver=currentsolver)
                probtype = 'SOCP'
        else:
            dims = {}
            dims['s'] = [int(np.sqrt(Gsi.size[0]))
                         for Gsi in self.cvxoptVars['Gs']]
            dims['l'] = self.cvxoptVars['Gl'].size[0]
            dims['q'] = [Gqi.size[0] for Gqi in self.cvxoptVars['Gq']]

            G = self.cvxoptVars['Gl']
            h = self.cvxoptVars['hl']
            # handle the equalities as 2 ineq for smcp
            if currentsolver == 'smcp':
                if self.cvxoptVars['A'].size[0] > 0:
                    G = cvx.sparse([G, self.cvxoptVars['A']])
                    G = cvx.sparse([G, -self.cvxoptVars['A']])
                    h = cvx.matrix([h, self.cvxoptVars['b']])
                    h = cvx.matrix([h, -self.cvxoptVars['b']])
                    dims['l'] += (2 * self.cvxoptVars['A'].size[0])

            for i in range(len(dims['q'])):
                G = cvx.sparse([G, self.cvxoptVars['Gq'][i]])
                h = cvx.matrix([h, self.cvxoptVars['hq'][i]])

            for i in range(len(dims['s'])):
                G = cvx.sparse([G, self.cvxoptVars['Gs'][i]])
                h = cvx.matrix([h, self.cvxoptVars['hs'][i]])

            # Remove the lines in A and b corresponding to 0==0
            JP = list(set(self.cvxoptVars['A'].I))
            IP = range(len(JP))
            VP = [1] * len(JP)

            idx_0eq0 = [
                i for i in range(
                    self.cvxoptVars['A'].size[0]) if i not in JP]

            # is there a constraint of the form 0==a(a not 0) ?
            if any([b for (i, b) in enumerate(
                    self.cvxoptVars['b']) if i not in JP]):
                raise Exception('infeasible constraint of the form 0=a')
            P = spmatrix(
                VP, IP, JP, (len(IP), self.cvxoptVars['A'].size[0]))
            self.cvxoptVars['A'] = P * self.cvxoptVars['A']
            self.cvxoptVars['b'] = P * self.cvxoptVars['b']

            tstart = time.time()
            if currentsolver == 'smcp':
                try:
                    import smcp
                except:
                    raise Exception('library smcp not found')
                if self.options['smcp_feas']:
                    sol = smcp.solvers.conelp(
                        self.cvxoptVars['c'], G, h, dims, feas=self.options['smcp_feas'])
                else:
                    sol = smcp.solvers.conelp(self.cvxoptVars['c'],
                                              G, h, dims)
            else:
                if self.options['verbose'] > 0:
                    print('--------------------------')
                    print('  cvxopt CONELP solver')
                    print('--------------------------')
                sol = cvx.solvers.conelp(self.cvxoptVars['c'],
                                         G, h, dims,
                                         self.cvxoptVars['A'],
                                         self.cvxoptVars['b'])
            probtype = 'ConeLP'

        tend = time.time()

        status = sol['status']
        solv = currentsolver
        if solv is None:
            solv = 'cvxopt'
        if self.options['verbose'] > 0:
            print(solv + ' status: ' + status)

        #----------------------#
        # retrieve the primals #
        #----------------------#
        primals = {}
        if 'noprimals' in self.options and self.options['noprimals']:
            pass
        else:
            try:

                for var in self.variables.values():
                    si = var.startIndex
                    ei = var.endIndex
                    varvect = sol['x'][si:ei]
                    if var.vtype in ('symmetric',):
                        varvect = svecm1(varvect)  # varvect was the svec
                        # representation of X

                    primals[var.name] = cvx.matrix(varvect, var.size)
            except Exception as ex:
                primals = {}
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Primal Solution not found\033[0m")

        #--------------------#
        # retrieve the duals #
        #--------------------#
        duals = []
        if 'noduals' in self.options and self.options['noduals']:
            pass
        else:
            try:
                printnodual = False
                (indy, indzl, indzq, indznl, indzs) = (0, 0, 0, 0, 0)
                if probtype == 'LP' or probtype == 'ConeLP':
                    zkey = 'z'
                else:
                    zkey = 'zl'
                zqkey = 'zq'
                zskey = 'zs'
                if probtype == 'ConeLP':
                    indzq = dims['l']
                    zqkey = 'z'
                    zskey = 'z'
                    indzs = dims['l'] + _bsum(dims['q'])

                if currentsolver == 'smcp':
                    ieq = self.cvxoptVars['Gl'].size[0]
                    neq = (dims['l'] - ieq) // 2
                    soleq = sol['z'][ieq:ieq + neq]
                    soleq -= sol['z'][ieq + neq:ieq + 2 * neq]
                else:
                    soleq = sol['y']

                for k, consk in enumerate(self.constraints):
                    # Equality
                    if consk.typeOfConstraint == 'lin=':
                        if not (soleq is None):
                            consSz = np.product(consk.Exp1.size)
                            duals.append((P.T * soleq)[indy:indy + consSz])
                            indy += consSz
                        else:
                            printnodual = True
                            duals.append(None)
                    # Inequality
                    elif consk.typeOfConstraint[:3] == 'lin':
                        if not (sol[zkey] is None):
                            consSz = np.product(consk.Exp1.size)
                            duals.append(sol[zkey][indzl:indzl + consSz])
                            indzl += consSz
                        else:
                            printnodual = True
                            duals.append(None)
                    # SOCP constraint [Rotated or not]
                    elif consk.typeOfConstraint[2:] == 'cone':
                        if not (sol[zqkey] is None):
                            if probtype == 'ConeLP':
                                consSz = np.product(consk.Exp1.size) + 1
                                if consk.typeOfConstraint[:2] == 'RS':
                                    consSz += 1
                                duals.append(sol[zqkey][indzq:indzq + consSz])
                                duals[-1][1:] = -duals[-1][1:]
                                indzq += consSz
                            else:
                                duals.append(sol[zqkey][indzq])
                                duals[-1][1:] = -duals[-1][1:]
                                indzq += 1
                        else:
                            printnodual = True
                            duals.append(None)
                    # SDP constraint
                    elif consk.typeOfConstraint[:3] == 'sdp':
                        if not (sol[zskey] is None):
                            if probtype == 'ConeLP':
                                matsz = consk.Exp1.size[0]
                                consSz = matsz * matsz
                                duals.append(
                                    cvx.matrix(
                                        sol[zskey][
                                            indzs:indzs + consSz], (matsz, matsz)))
                                indzs += consSz
                            else:
                                matsz = consk.Exp1.size[0]
                                duals.append(
                                    cvx.matrix(
                                        sol[zskey][indzs], (matsz, matsz)))
                                indzs += 1
                        else:
                            printnodual = True
                            duals.append(None)
                    # GP constraint
                    elif consk.typeOfConstraint == 'lse':
                        if not (sol['znl'] is None):
                            consSz = np.product(consk.Exp1.size)
                            duals.append(sol['znl'][indznl:indznl + consSz])
                            indznl += consSz
                        else:
                            printnodual = True
                            duals.append(None)
                    else:
                        raise Exception('constraint cannot be handled')

                if printnodual and self.options['verbose'] > 0:
                    print("\033[1;31m*** Dual Solution not found\033[0m")

            except Exception as ex:
                duals = []
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Dual Solution not found\033[0m")

        #-----------------#
        # objective value #
        #-----------------#
        if self.numberLSEConstraints > 0:  # GP
            obj = 'toEval'
        else:  # LP or SOCP
            if sol['primal objective'] is None:
                if sol['dual objective'] is None:
                    obj = None
                else:
                    obj = sol['dual objective']
            else:
                if sol['dual objective'] is None:
                    obj = sol['primal objective']
                else:
                    obj = 0.5 * (sol['primal objective'] +
                                 sol['dual objective'])

            if self.objective[0] == 'max' and not obj is None:
                obj = -obj

        solt = {'cvxopt_sol': sol, 'status': status, 'time': tend - tstart}
        return primals, duals, obj, solt

    def _glpk_solve(self):
        """
        Solves a problem with the GLPK solver.
        """
        import swiglpk as glpk

        # Check if GLPK can solve this type of problem.
        if self.type not in ("LP", "MIP"):
            raise NotAppropriateSolverError(
                "'glpk' cannot solve problems of type {0}".format(self.type))
        continuous = (self.type == "LP")

        # Create a GLPK problem instance.
        self._make_glpk_instance()
        p = self.glpk_Instance

        # Select LP solver (Simplex or Interior Point Method).
        if continuous:
            if self.options["lp_root_method"] == "interior":
                interior = True
            else:
                # Default to Simplex.
                interior = False
            simplex = not interior
        else:
            simplex = interior = False

        # Select appropriate options container.
        if simplex:
            options = glpk.glp_smcp()
            glpk.glp_init_smcp(options)
        elif interior:
            options = glpk.glp_iptcp()
            glpk.glp_init_iptcp(options)
        else:
            options = glpk.glp_iocp()
            glpk.glp_init_iocp(options)

        # Handle "verbose" option.
        verbosity = self.options["verbose"]
        if verbosity < 0:
            options.msg_lev = glpk.GLP_MSG_OFF
        elif verbosity == 0:
            options.msg_lev = glpk.GLP_MSG_ERR
        elif verbosity == 1:
            options.msg_lev = glpk.GLP_MSG_ON
        elif verbosity > 1:
            options.msg_lev = glpk.GLP_MSG_ALL

        # Handle "tol" option.
        # Note that GLPK knows three different tolerances for Simplex but none
        # for the Interior Point Method, while PICOS states that "tol" is meant
        # only for the IPM.
        # XXX: The option is unsupported but does not default to None, so we
        #      cannot warn the user.
        pass

        # Handle "maxit" option.
        if self.options["maxit"] is not None:
            if simplex:
                options.it_lim = int(self.options["maxit"])
            else:
                raise NotImplementedError(
                    "GLPK supports the 'maxit' option only with Simplex.")

        # Handle "lp_root_method" option.
        # Note that the PICOS option is explicitly also meant for the MIP
        # preprocessing step but GLPK does not support it in that scenario.
        if self.options["lp_root_method"] is not None:
            if not continuous:
                raise NotImplementedError(
                    "GLPK supports the 'lp_root_method' option only for LPs.")
            elif self.options["lp_root_method"] == "psimplex":
                assert simplex
                options.meth = glpk.GLP_PRIMAL
            elif self.options["lp_root_method"] == "dsimplex":
                assert simplex
                options.meth = glpk.GLP_DUAL

        # Handle "lp_node_method" option.
        if self.options["lp_node_method"] is not None:
            raise NotImplementedError(
                "GLPK does not support the 'lp_node_method' option.")

        # Handle "timelimit" option.
        if self.options["timelimit"] is not None:
            if interior:
                raise NotImplementedError(
                    "GLPK does not support the 'timelimit' option with the "
                    "Interior Point Method.")
            options.tm_lim = 1000 * int(self.options["timelimit"])

        # Handle "treememory" option.
        if self.options["treememory"] is not None:
            raise NotImplementedError(
                "GLPK does not support the 'treememory' option.")

        # Handle "gaplim" option.
        # TODO: Find out if "mip_gap" is really equivalent to "gaplim".
        if self.options["gaplim"] is not None:
            if continuous:
                # Every LP is a MIP, so setting the option for an LP is fine.
                pass
            options.mip_gap = float(self.options["gaplim"])

        # Handle "nbsol" option.
        if self.options["nbsol"] is not None:
            raise NotImplementedError(
                "GLPK does not support the 'nbsol' option.")

        # Handle "hotstart" option.
        if self.options["hotstart"]:
            raise NotImplementedError(
                "GLPK does not support the 'hotstart' option.")

        # Handle "pass_simple_cons_as_bound" option.
        # Note that this option is solver-specific, while it should not be.
        if self.options["pass_simple_cons_as_bound"]:
            raise NotImplementedError(
                "GLPK does not support the 'pass_simple_cons_as_bound' option. "
                "(Note that this option is currently solver-specific.)"
            )

        # TODO: Add GLPK-sepcific options. Candidates are:
        #       For both Simplex and MIPs:
        #           tol_*, out_*
        #       For Simplex:
        #           pricing, r_test, obj_*
        #       For the Interior Point Method:
        #           ord_alg
        #       For MIPs:
        #           *_tech, *_heur, ps_tm_lim, *_cuts, cb_size, binarize

        if verbosity > 0:
            print('-----------------------------------')
            print('    GNU Linear Programming Kit')
            print('-----------------------------------')
            sys.stdout.flush()

        # Attempt to solve the problem.
        import time
        startTime = time.time()
        if simplex:
            # TODO: glp_exact.
            error = glpk.glp_simplex(p, options)
        elif interior:
            error = glpk.glp_interior(p, options)
        else:
            options.presolve = glpk.GLP_ON
            error = glpk.glp_intopt(p, options)
        endTime = time.time()

        # TODO: Handle errors.
        if verbosity >= 0:
            # Catch internal failures and bad problem formulations.
            if error == glpk.GLP_EBADB:
                print("Unable to start the search, because the initial "
                    "basis specified in the problem object is invalid.")
            elif error == glpk.GLP_ESING:
                print("Unable to start the search, because the basis matrix "
                    "corresponding to the initial basis is singular within "
                    "the working precision.")
            elif error == glpk.GLP_ECOND:
                print("Unable to start the search, because the basis matrix "
                    "corresponding to the initial basis is ill-conditioned.")
            elif error == glpk.GLP_EBOUND:
                print("Unable to start the search, because some double-bounded "
                    "variables have incorrect bounds.")
            elif error == glpk.GLP_EFAIL:
                print("The search was prematurely terminated due to a solver "
                    "failure.")
            elif error == glpk.GLP_EOBJLL:
                print("The search was prematurely terminated, because the "
                    "objective function being maximized has reached its lower "
                    "limit and continues decreasing.")
            elif error == glpk.GLP_EOBJUL:
                print("The search was prematurely terminated, because the "
                    "objective function being minimized has reached its upper "
                    "limit and continues increasing.")
            elif error == glpk.GLP_EITLIM:
                print("The search was prematurely terminated, because the "
                    "simplex iteration limit has been exceeded.")
            elif error == glpk.GLP_ETMLIM:
                print("The search was prematurely terminated, because the time "
                    "limit has been exceeded.")
            # Catch problem infeasibility.
            elif error == glpk.GLP_ENOPFS and verbosity > 0:
                print("The LP has no primal feasible solution.")
            elif error == glpk.GLP_ENODFS and verbosity > 0:
                print("The LP has no dual feasible solution.")
            # Catch unknown errors.
            elif error != 0:
                print("An unknown GLPK error (code {:d}) occured during search."
                    .format(error))

        # Retrieve primals.
        primals = {}
        if not self.options["noprimals"]:
            for varName in self.varNames:
                var = self.variables[varName]
                values = []
                for localIndex, picosIndex in enumerate(range(var.startIndex, var.endIndex)):
                    glpkIndex = self._picos2glpk_variable_index(picosIndex)
                    if simplex:
                        localValue = glpk.glp_get_col_prim(p, glpkIndex);
                    elif interior:
                        localValue = glpk.glp_ipt_col_prim(p, glpkIndex);
                    else:
                        localValue = glpk.glp_mip_col_val(p, glpkIndex);
                    values.append(localValue)
                primals[varName] = values

        # Retrieve duals.
        # XXX: Returns the duals as a flat cvx.matrix to be consistent with
        #      other solvers. This feels incorrect when the constraint was given
        #      as a proper two dimensional matrix.
        duals = []
        if not self.options["noduals"] and continuous:
            rowOffset = 1
            for constraintNum, constraint in enumerate(self.constraints):
                numRows = constraint.Exp1.size[0] * constraint.Exp1.size[1]
                values = []
                for localConstraintIndex in range(numRows):
                    glpkConstraintIndex = rowOffset + localConstraintIndex
                    if simplex:
                        localValue = glpk.glp_get_row_dual(p, glpkConstraintIndex);
                    elif interior:
                        localValue = glpk.glp_ipt_row_dual(p, glpkConstraintIndex);
                    else:
                        assert False
                    values.append(localValue)
                constraintRelation = constraint.typeOfConstraint[3]
                assert constraintRelation in ("<", ">", "=")
                if constraintRelation in ("<", "="):
                    duals.append(cvx.matrix(values))
                else:
                    duals.append(-cvx.matrix(values))
                rowOffset += numRows
        if self.objective[0] == "min":
            duals = [-d for d in duals]

        # Retrieve objective value.
        if simplex:
            objectiveValue = glpk.glp_get_obj_val(p)
        elif interior:
            objectiveValue = glpk.glp_ipt_obj_val(p)
        else:
            objectiveValue = glpk.glp_mip_obj_val(p)

        # Retrieve solution metadata.
        solution = {}

        if simplex:
            # Set common entry "status".
            status = glpk.glp_get_status(p)
            if status is glpk.GLP_OPT:
                solution["status"] = "optimal"
            elif status is glpk.GLP_FEAS:
                solution["status"] = "feasible"
            elif status in (glpk.GLP_INFEAS, glpk.GLP_NOFEAS):
                solution["status"] = "infeasible"
            elif status is glpk.GLP_UNBND:
                solution["status"] = "unbounded"
            elif status is glpk.GLP_UNDEF:
                solution["status"] = "undefined"
            else:
                solution["status"] = "unknown"

            # Set GLPK-specific entry "primal_status".
            primalStatus = glpk.glp_get_prim_stat(p)
            if primalStatus is glpk.GLP_FEAS:
                solution["primal_status"] = "feasible"
            elif primalStatus in (glpk.GLP_INFEAS, glpk.GLP_NOFEAS):
                solution["primal_status"] = "infeasible"
            elif primalStatus is glpk.GLP_UNDEF:
                solution["primal_status"] = "undefined"
            else:
                solution["primal_status"] = "unknown"

            # Set GLPK-specific entry "dual_status".
            dualStatus = glpk.glp_get_dual_stat(p)
            if dualStatus is glpk.GLP_FEAS:
                solution["dual_status"] = "feasible"
            elif dualStatus in (glpk.GLP_INFEAS, glpk.GLP_NOFEAS):
                solution["dual_status"] = "infeasible"
            elif dualStatus is glpk.GLP_UNDEF:
                solution["dual_status"] = "undefined"
            else:
                solution["dual_status"] = "unknown"
        elif interior:
            # Set common entry "status".
            status = glpk.glp_ipt_status(p)
            if status is glpk.GLP_OPT:
                solution["status"] = "optimal"
            elif status in (glpk.GLP_INFEAS, glpk.GLP_NOFEAS):
                solution["status"] = "infeasible"
            elif status is glpk.GLP_UNDEF:
                solution["status"] = "undefined"
            else:
                solution["status"] = "unknown"
        else:
            # Set common entry "status".
            status = glpk.glp_mip_status(p)
            if status is glpk.GLP_OPT:
                solution["status"] = "optimal"
            elif status is glpk.GLP_FEAS:
                solution["status"] = "feasible"
            elif status is glpk.GLP_NOFEAS:
                solution["status"] = "infeasible"
            elif status is glpk.GLP_UNDEF:
                solution["status"] = "undefined"
            else:
                solution["status"] = "unknown"

        # Set common entry "time".
        solution["time"] = endTime - startTime

        return (primals, duals, objectiveValue, solution)

    def _cplex_solve(self):
        """
        Solves a problem with the cvxopt solver.
        """

        #-------------------------------#
        #  can we solve it with cplex ? #
        #-------------------------------#

        if self.type in (
                'unknown type',
                'MISDP',
                'GP',
                'SDP',
                'ConeP',
                'Mixed (SDP+quad)'):
            raise NotAppropriateSolverError(
                "'cplex' cannot solve problems of type {0}".format(
                    self.type))

        #----------------------------#
        #  create the cplex instance #
        #----------------------------#
        import cplex
        self._make_cplex_instance()
        c = self.cplex_Instance

        if c is None:
            raise ValueError(
                'a cplex instance should have been created before')

        if not self.options['treememory'] is None:
            c.parameters.mip.limits.treememory.set(self.options['treememory'])
        if not self.options['gaplim'] is None:
            c.parameters.mip.tolerances.mipgap.set(self.options['gaplim'])
        # pool of solutions
        if not self.options['pool_size'] is None:
            c.parameters.mip.limits.solutions.set(self.options['pool_size'])
        if not self.options['pool_gap'] is None:
            c.parameters.mip.pool.relgap.set(self.options['pool_gap'])
        # verbosity
        c.parameters.barrier.display.set(min(2, self.options['verbose']))
        c.parameters.simplex.display.set(min(2, self.options['verbose']))
        if self.options['verbose'] == 0:
            c.parameters.mip.display.set(0)

        # convergence tolerance
        c.parameters.barrier.qcpconvergetol.set(self.options['tol'])
        c.parameters.barrier.convergetol.set(self.options['tol'])

        # iterations limit
        if not(self.options['maxit'] is None):

            c.parameters.barrier.limits.iteration.set(self.options['maxit'])
            c.parameters.simplex.limits.iterations.set(self.options['maxit'])

        # lpmethod
        if not self.options['lp_root_method'] is None:
            if self.options['lp_root_method'] == 'psimplex':
                c.parameters.lpmethod.set(1)
            elif self.options['lp_root_method'] == 'dsimplex':
                c.parameters.lpmethod.set(2)
            elif self.options['lp_root_method'] == 'interior':
                c.parameters.lpmethod.set(4)
            else:
                raise Exception('unexpected value for lp_root_method')
        if not self.options['lp_node_method'] is None:
            if self.options['lp_node_method'] == 'psimplex':
                c.parameters.mip.strategy.subalgorithm.set(1)
            elif self.options['lp_node_method'] == 'dsimplex':
                c.parameters.mip.strategy.subalgorithm.set(2)
            elif self.options['lp_node_method'] == 'interior':
                c.parameters.mip.strategy.subalgorithm.set(4)
            else:
                raise Exception('unexpected value for lp_node_method')

        if not self.options['nbsol'] is None:
            c.parameters.mip.limits.solutions.set(self.options['nbsol'])
            # variant with a call back (count the incumbents)
            #import cplex_callbacks
            #nbsol_cb = c.register_callback(cplex_callbacks.nbIncCallback)
            #nbsol_cb.aborted = 0
            #nbsol_cb.cursol = 0
            #nbsol_cb.nbsol = self.options['nbsol']

        if ((not self.options['timelimit'] is None) or
                (not self.options['uboundlimit'] is None) or
                (not self.options['lboundlimit'] is None) or
                (self.options['boundMonitor'])):
            from . import cplex_callbacks
            picos_cb = c.register_callback(cplex_callbacks.PicosInfoCallback)
            picos_cb.aborted = 0
            picos_cb.ub = INFINITY
            picos_cb.lb = -INFINITY

            if (not self.options['timelimit']
                    is None) or self.options['boundMonitor']:
                import time
                picos_cb.starttime = time.time()

            if (not self.options['timelimit'] is None):
                picos_cb.timelimit = self.options['timelimit']
                if not self.options['acceptable_gap_at_timelimit'] is None:
                    picos_cb.acceptablegap = 100 * \
                        self.options['acceptable_gap_at_timelimit']
                else:
                    picos_cb.acceptablegap = None
            else:
                picos_cb.timelimit = None

            if not self.options['uboundlimit'] is None:
                picos_cb.ubound = self.options['uboundlimit']
            else:
                picos_cb.ubound = None

            if not self.options['lboundlimit'] is None:
                picos_cb.lbound = self.options['lboundlimit']
            else:
                picos_cb.lbound = None

            if self.options['boundMonitor']:
                picos_cb.bounds = []
            else:
                picos_cb.bounds = None

        """ old version, was not possible to use several callbacks at a time?
                if not self.options['timelimit'] is None:
                        import cplex_callbacks
                        import time
                        timelim_cb = c.register_callback(cplex_callbacks.TimeLimitCallback)
                        timelim_cb.starttime = time.time()
                        timelim_cb.timelimit = self.options['timelimit']
                        if not self.options['acceptable_gap_at_timelimit'] is None:
                                timelim_cb.acceptablegap =100*self.options['acceptable_gap_at_timelimit']
                        else:
                                timelim_cb.acceptablegap = None
                        timelim_cb.aborted = 0
                        #c.parameters.tuning.timelimit.set(self.options['timelimit']) #DOES NOT WORK LIKE THIS ?

                if not self.options['uboundlimit'] is None:
                        import cplex_callbacks
                        bound_cb =  c.register_callback(cplex_callbacks.uboundCallback)
                        bound_cb.aborted = 0
                        bound_cb.ub = INFINITY
                        bound_cb.bound = self.options['uboundlimit']

                if not self.options['lboundlimit'] is None:
                        import cplex_callbacks
                        bound_cb =  c.register_callback(cplex_callbacks.lboundCallback)
                        bound_cb.aborted = 0
                        bound_cb.ub = -INFINITY
                        bound_cb.bound = self.options['lboundlimit']

                if self.options['boundMonitor']:
                        import cplex_callbacks
                        import time
                        monitor_cb = c.register_callback(cplex_callbacks.boundMonitorCallback)
                        monitor_cb.starttime = time.time()
                        monitor_cb.bounds = []
                """

        # other cplex parameters
        for par, val in six.iteritems(self.options['cplex_params']):
            try:
                cplexpar = eval('c.parameters.' + par)
                cplexpar.set(val)
            except AttributeError:
                raise Exception('unknown cplex param')

        #--------------------#
        #  call the solver   #
        #--------------------#
        import time
        tstart = time.time()

        if not self.options['pool_size'] is None:
            try:
                c.populate_solution_pool()
            except:
                print("Exception raised during populate")
        else:
            try:
                c.solve()
            except cplex.exceptions.CplexSolverError as ex:
                if ex.args[2] == 5002:
                    raise NonConvexError(
                        'Error raised during solve. Problem is nonconvex')
                else:
                    print("Exception raised during solve")
        tend = time.time()

        self.cplex_Instance = c

        # solution.get_status() returns an integer code
        if self.options['verbose'] > 0:
            print("Solution status = " + str(c.solution.get_status()) + ":")
            # the following line prints the corresponding string
            print(c.solution.status[c.solution.get_status()])
        status = c.solution.status[c.solution.get_status()]

        #----------------------#
        # retrieve the primals #
        #----------------------#
        primals = {}
        obj = c.solution.get_objective_value()
        if 'noprimals' in self.options and self.options['noprimals']:
            pass
        else:
            # primals
            try:
                if self.options['pool_size'] is None:
                    numsol = 0
                else:
                    numsol = c.solution.pool.get_num()
                if numsol > 1:
                    objvals = []
                    for i in range(numsol):
                        objvals.append(
                            (c.solution.pool.get_objective_value(i), i))
                    indsols = []
                    rev = (self.objective[0] == 'max')
                    for ob, ind in sorted(objvals, reverse=rev)[
                            :self.options['pool_size']]:
                        indsols.append(ind)

                for var in self.variables.values():
                    value = c.solution.get_values(
                        var.startIndex, var.endIndex - 1)

                    # much slowlier !
                    #value = []
                    #sz_var = var.endIndex-var.startIndex
                    # for i in range(sz_var):
                    #name = var.name + '_' + str(i)
                    # value.append(c.solution.get_values(name))

                    if var.vtype in ('symmetric',):
                        value = svecm1(
                            cvx.matrix(value))  # varvect was the svec
                        # representation of X
                    primals[var.name] = cvx.matrix(value, var.size)

                if numsol > 1:
                    for ii, ind in enumerate(indsols):
                        for var in self.variables.values():
                            value = []
                            sz_var = var.endIndex - var.startIndex
                            for i in range(sz_var):
                                name = var.name + '_' + str(i)
                                value.append(
                                    c.solution.pool.get_values(
                                        ind, name))
                            if var.vtype in ('symmetric',):
                                value = svecm1(
                                    cvx.matrix(value))  # varvect was the svec
                                # representation of X
                            primals[(ii, var.name)] = cvx.matrix(
                                value, var.size)
            except Exception as ex:
                import warnings
                warnings.warn('error while retrieving primals')
                primals = {}
                obj = None
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Primal Solution not found\033[0m")

        #--------------------#
        # retrieve the duals #
        #--------------------#

        duals = []
        if not(self.isContinuous()) or (
                'noduals' in self.options and self.options['noduals']):
            pass
        else:
            try:
                version = [int(v) for v in c.get_version().split('.')]
                # older versions
                if (version[0] < 12) or (version[0] == 12 and version[1] < 4):
                    pos_cplex = 0  # row position in the cplex linear constraints
                    #basis_status = c.solution.basis.get_col_basis()
                    #>0 and <0
                    pos_conevar = self.numberOfVars + 1  # plus 1 for the __noconstant__ variable
                    seen_bounded_vars = []
                    for k, constr in enumerate(self.constraints):
                        if constr.typeOfConstraint[:3] == 'lin':
                            dim = constr.Exp1.size[0] * constr.Exp1.size[1]
                            dim = dim - len(self.cplex_boundcons[k])
                            dual_lines = range(pos_cplex, pos_cplex + dim)
                            if len(dual_lines) == 0:
                                dual_values = []
                            else:
                                dual_values = c.solution.get_dual_values(
                                    dual_lines)
                            if constr.typeOfConstraint[3] == '>':
                                dual_values = [-dvl for dvl in dual_values]
                            for (i, j, b, v) in self.cplex_boundcons[k]:
                                xj = c.solution.get_values(j)
                                if ((b == '=') or abs(xj - b) <
                                        1e-7) and (j not in seen_bounded_vars):
                                    # does j appear in another equality
                                    # constraint ?
                                    if b != '=':
                                        boundsj = [b0 for k0 in range(len(self.constraints)) for (
                                            i0, j0, b0, v0) in self.cplex_boundcons[k0] if j0 == j]
                                        if '=' in boundsj:
                                            dual_values.insert(
                                                i, 0.)  # dual will be set later, only for the equality case
                                            continue
                                    else:  # equality
                                        seen_bounded_vars.append(j)
                                        du = c.solution.get_reduced_costs(
                                            j) / v
                                        dual_values.insert(i, du)
                                        continue
                                    # what kind of inequality ?
                                    du = c.solution.get_reduced_costs(j)
                                    if (((v > 0 and constr.typeOfConstraint[3] == '<') or
                                         (v < 0 and constr.typeOfConstraint[3] == '>')) and
                                            du > 0):  # upper bound
                                        seen_bounded_vars.append(j)
                                        dual_values.insert(i, du / abs(v))
                                    elif (((v > 0 and constr.typeOfConstraint[3] == '>') or
                                           (v < 0 and constr.typeOfConstraint[3] == '<')) and
                                            du < 0):  # lower bound
                                        seen_bounded_vars.append(j)
                                        dual_values.insert(i, -du / abs(v))
                                    else:
                                        # unactive constraint
                                        dual_values.insert(i, 0.)
                                else:
                                    dual_values.insert(i, 0.)
                            pos_cplex += dim
                            duals.append(cvx.matrix(dual_values))

                        elif constr.typeOfConstraint == 'SOcone':
                            szcons = constr.Exp1.size[0] * constr.Exp1.size[1]
                            dual_cols = range(
                                pos_conevar, pos_conevar + szcons + 1)
                            dual_values = c.solution.get_reduced_costs(
                                dual_cols)
                            # duals.append(int(np.sign(dual_values[-1])) * cvx.matrix(
                            # [dual_values[-1]]+dual_values[:-1]))
                            duals.append(cvx.matrix(
                                [-dual_values[-1]] + dual_values[:-1]))
                            pos_conevar += szcons + 1

                        elif constr.typeOfConstraint == 'RScone':
                            szcons = constr.Exp1.size[0] * constr.Exp1.size[1]
                            dual_cols = range(
                                pos_conevar, pos_conevar + szcons + 2)
                            dual_values = c.solution.get_reduced_costs(
                                dual_cols)
                            # duals.append(int(np.sign(dual_values[-1])) * cvx.matrix(
                            #                [dual_values[-1]]+dual_values[:-1]))
                            duals.append(cvx.matrix(
                                [-dual_values[-1]] + dual_values[:-1]))
                            pos_conevar += szcons + 2

                        else:
                            if self.options['verbose'] > 0:
                                print(
                                    'duals for this type of constraint not supported yet')
                            duals.append(None)
                #version >= 12.4
                else:
                    seen_bounded_vars = []
                    for k, constr in enumerate(self.constraints):
                        if constr.typeOfConstraint[:3] == 'lin':
                            dim = constr.Exp1.size[0] * constr.Exp1.size[1]
                            dual_values = [None] * dim

                            # rows with var bounds
                            for (i, j, b, v) in self.cplex_boundcons[k]:
                                xj = c.solution.get_values(j)
                                if ((b == '=') or abs(xj - b) <
                                        1e-4) and (j not in seen_bounded_vars):
                                    # does j appear in another equality
                                    # constraint ?
                                    if b != '=':
                                        boundsj = [b0 for k0 in range(len(self.constraints)) for (
                                            i0, j0, b0, v0) in self.cplex_boundcons[k0] if j0 == j]
                                        if '=' in boundsj:
                                            dual_values[
                                                i] = 0.  # dual will be set later, only for the equality case
                                            continue
                                    else:  # equality
                                        seen_bounded_vars.append(j)
                                        du = c.solution.get_reduced_costs(
                                            j) / v
                                        if self.objective[0] == 'min':
                                            du = -du
                                        dual_values[i] = du
                                        continue
                                    # what kind of inequality ?
                                    du = c.solution.get_reduced_costs(j)
                                    if self.objective[0] == 'min':
                                        du = -du
                                    if (((v > 0 and constr.typeOfConstraint[3] == '<') or
                                         (v < 0 and constr.typeOfConstraint[3] == '>')) and
                                            du > 0):  # upper bound
                                        seen_bounded_vars.append(j)
                                        dual_values[i] = du / abs(v)
                                    elif (((v > 0 and constr.typeOfConstraint[3] == '>') or
                                           (v < 0 and constr.typeOfConstraint[3] == '<')) and
                                            du < 0):  # lower bound
                                        seen_bounded_vars.append(j)
                                        dual_values[i] = -du / abs(v)
                                    else:
                                        dual_values[
                                            i] = 0.  # unactive constraint
                                else:
                                    dual_values[i] = 0.

                            # rows with other constraints
                            if hasattr(constr,'zero_rows'):
                                for i in constr.zero_rows:
                                    dual_values[i] = 0. #constr 0*z <=> b
                            for consname in constr.Id['cplex']:
                                id = int(consname[consname.rfind('_')+1:])
                                du = c.solution.get_dual_values(consname)
                                if self.objective[0] == 'min':
                                    du = -du
                                if constr.typeOfConstraint[3] == '>':
                                    dual_values[id] = -du
                                else:
                                    dual_values[id] = du

                            duals.append(cvx.matrix(dual_values))

                        elif constr.typeOfConstraint == 'SOcone':
                            dual_values = []
                            rhs_id = [id for id in constr.Id['cplex'] if '_rhs_' in id][0]
                            dual_values.append(
                                c.solution.get_dual_values(rhs_id))
                            dim = constr.Exp1.size[0] * constr.Exp1.size[1]
                            lhs_id = [id for id in constr.Id['cplex'] if '_lhs_' in id]
                            assert dim == len(lhs_id)
                            for ids in lhs_id:
                                dual_values.append(-c.solution.get_dual_values(ids))
                            if self.objective[0] == 'min':
                                duals.append(-cvx.matrix(dual_values))
                            else:
                                duals.append(cvx.matrix(dual_values))

                        elif constr.typeOfConstraint == 'RScone':
                            dual_values = []
                            rhs_id = [id for id in constr.Id['cplex'] if '_rhs_' in id][0]
                            dual_values.append(
                                c.solution.get_dual_values(rhs_id))
                            dim = 1 + constr.Exp1.size[0] * constr.Exp1.size[1]
                            lhs_id = [id for id in constr.Id['cplex'] if '_lhs_' in id]
                            assert dim == len(lhs_id)
                            for ids in lhs_id:
                                dual_values.append(-c.solution.get_dual_values(ids))
                            if self.objective[0] == 'min':
                                duals.append(-cvx.matrix(dual_values))
                            else:
                                duals.append(cvx.matrix(dual_values))

                        else:
                            if self.options['verbose'] > 0:
                                print(
                                    'duals for this type of constraint not supported yet')
                            duals.append(None)

            except Exception as ex:
                import pdb;pdb.set_trace()
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Dual Solution not found\033[0m")

        #-----------------#
        # return statement#
        #-----------------#
        sol = {
            'cplex_solution': c.solution,
            'status': status,
            'time': (
                tend - tstart)}
        if self.options['boundMonitor']:
            sol['bounds_monitor'] = picos_cb.bounds  # monitor_cb.bounds
        return (primals, duals, obj, sol)

    def _gurobi_solve(self):
        """
        Solves a problem with the cvxopt solver.
        """

        #--------------------------------#
        #  can we solve it with gurobi ? #
        #--------------------------------#

        if self.type in (
                'unknown type',
                'MISDP',
                'GP',
                'SDP',
                'ConeP',
                'Mixed (SDP+quad)'):
            raise NotAppropriateSolverError(
                "'gurobi' cannot solve problems of type {0}".format(
                    self.type))

        #----------------------------#
        #  create the gurobi instance #
        #----------------------------#
        import gurobipy as grb
        self._make_gurobi_instance()
        m = self.gurobi_Instance

        if m is None:
            raise ValueError(
                'a gurobi instance should have been created before')

        # verbosity
        if self.options['verbose'] == 0:
            m.setParam('OutputFlag', 0)

        if not self.options['timelimit'] is None:
            m.setParam('TimeLimit', self.options['timelimit'])
        if not self.options['treememory'] is None:
            if self.options['verbose']:
                print('option treememory ignored with gurobi')
            # m.setParam('NodefileStart',self.options['treememory']/1024.)
            # -> NO In fact this is a limit after which node files are written to disk
        if not self.options['gaplim'] is None:
            m.setParam('MIPGap', self.options['gaplim'])
            # m.setParam('MIPGapAbs',self.options['gaplim'])

        # convergence tolerance
        m.setParam('BarQCPConvTol', self.options['tol'])
        m.setParam('BarConvTol', self.options['tol'])
        m.setParam('OptimalityTol', self.options['tol'])

        # iterations limit
        if not(self.options['maxit'] is None):
            m.setParam('BarIterLimit', self.options['maxit'])
            m.setParam('IterationLimit', self.options['maxit'])
        # lpmethod
        if not self.options['lp_root_method'] is None:
            if self.options['lp_root_method'] == 'psimplex':
                m.setParam('Method', 0)
            elif self.options['lp_root_method'] == 'dsimplex':
                m.setParam('Method', 1)
            elif self.options['lp_root_method'] == 'interior':
                m.setParam('Method', 2)
            else:
                raise Exception('unexpected value for lp_root_method')
        if not self.options['lp_node_method'] is None:
            if self.options['lp_node_method'] == 'psimplex':
                m.setParam('SiftMethod', 0)
            elif self.options['lp_node_method'] == 'dsimplex':
                m.setParam('SiftMethod', 1)
            elif self.options['lp_node_method'] == 'interior':
                m.setParam('SiftMethod', 2)
            else:
                raise Exception('unexpected value for lp_node_method')

        # number of feasible solutions found
        if not self.options['nbsol'] is None:
            m.setParam('SolutionLimit', self.options['nbsol'])

        # other gurobi parameters
        for par, val in six.iteritems(self.options['gurobi_params']):
            m.setParam(par, val)

        # QCPDuals

        if not(self.isContinuous()) or (
                'noduals' in self.options and self.options['noduals']):
            m.setParam('QCPDual', 0)
        else:
            m.setParam('QCPDual', 1)

        #--------------------#
        #  call the solver   #
        #--------------------#

        import time
        tstart = time.time()

        try:
            m.optimize()
        except Exception as ex:
            if str(ex).startswith('Objective Q not PSD'):
                raise NonConvexError(
                    'Error raised during solve. Problem is nonconvex')
            else:
                print("Exception raised during solve")
        tend = time.time()

        self.gurobi_Instance = m

        status = None
        for st in dir(grb.GRB.Status):
            if st[0] != '_':
                if m.status == eval('grb.GRB.' + st):
                    status = st
        if status is None:
            import warnings
            warnings.warn('gurobi status not found')
            status = m.status
            if self.options['verbose'] > 0:
                print("\033[1;31m*** gurobi status not found \033[0m")

        #----------------------#
        # retrieve the primals #
        #----------------------#
        primals = {}
        obj = m.getObjective().getValue()
        if 'noprimals' in self.options and self.options['noprimals']:
            pass
        else:
            # primals
            try:
                for var in self.variables.values():
                    value = []
                    sz_var = var.endIndex - var.startIndex
                    for i in range(sz_var):
                        name = var.name + '_' + str(i)
                        xi = m.getVarByName(name)
                        value.append(xi.X)
                    if var.vtype in ('symmetric',):
                        value = svecm1(cvx.matrix(value))  # value was the svec
                        # representation of X

                    primals[var.name] = cvx.matrix(value, var.size)

            except Exception as ex:
                import warnings
                warnings.warn('error while retrieving primals')
                primals = {}
                obj = None
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Primal Solution not found\033[0m")

        #--------------------#
        # retrieve the duals #
        #--------------------#

        duals = []
        if not(self.isContinuous()) or (
                'noduals' in self.options and self.options['noduals']):
            pass
        else:
            try:
                seen_bounded_vars = []
                for k, constr in enumerate(self.constraints):
                    if constr.typeOfConstraint[:3] == 'lin':
                        dim = constr.Exp1.size[0] * constr.Exp1.size[1]
                        dual_values = [None] * dim
                        for (i, name, b, v) in self.grb_boundcons[k]:
                            xj = self.gurobi_Instance.getVarByName(name).X
                            if ((b == '=') or abs(xj - b) <
                                    1e-4) and (name not in seen_bounded_vars):
                                # does j appear in another equality constraint
                                # ?
                                if b != '=':
                                    boundsj = [b0 for k0 in range(len(self.constraints))
                                               for (i0, name0, b0, v0) in self.grb_boundcons[k0]
                                               if name0 == name]
                                    if '=' in boundsj:
                                        dual_values[
                                            i] = 0.  # dual will be set later, only for the equality case
                                        continue
                                else:  # equality
                                    seen_bounded_vars.append(name)
                                    du = self.gurobi_Instance.getVarByName(
                                        name).RC / v
                                    if self.objective[0] == 'min':
                                        du = -du
                                    dual_values[i] = du
                                    continue
                                # what kind of inequality ?
                                du = self.gurobi_Instance.getVarByName(name).RC
                                if self.objective[0] == 'min':
                                    du = -du
                                if (((v > 0 and constr.typeOfConstraint[3] == '<') or
                                        (v < 0 and constr.typeOfConstraint[3] == '>')) and
                                        du > 0):  # upper bound
                                    seen_bounded_vars.append(name)
                                    dual_values[i] = (du / abs(v))
                                elif (((v > 0 and constr.typeOfConstraint[3] == '>') or
                                       (v < 0 and constr.typeOfConstraint[3] == '<')) and
                                        du < 0):  # lower bound
                                    seen_bounded_vars.append(name)
                                    dual_values[i] = (-du / abs(v))
                                else:
                                    dual_values[i] = 0.  # unactive constraint
                            else:
                                dual_values[i] = 0.

                        # rows with other constraints
                        if hasattr(constr,'zero_rows'):
                            for i in constr.zero_rows:
                                dual_values[i] = 0. #constr 0*z <=> b
                        for consname in constr.Id['gurobi']:
                            id = int(consname[consname.rfind('_')+1:])
                            du = self.grbcons[consname].pi
                            if self.objective[0] == 'min':
                                du = -du
                            if constr.typeOfConstraint[3] == '>':
                                dual_values[id] = -du
                            else:
                                dual_values[id] = du

                        duals.append(cvx.matrix(dual_values))

                    elif constr.typeOfConstraint == 'SOcone':
                        dual_values = []
                        rhs_id = [id for id in constr.Id['gurobi'] if '_rhs_' in id][0]
                        dual_values.append(self.grbcons[rhs_id].pi)
                        dim = constr.Exp1.size[0] * constr.Exp1.size[1]
                        lhs_id = [id for id in constr.Id['gurobi'] if '_lhs_' in id]
                        assert dim == len(lhs_id)
                        for ids in lhs_id:
                            dual_values.append(-self.grbcons[ids].pi)
                        if self.objective[0] == 'min':
                            duals.append(-cvx.matrix(dual_values))
                        else:
                            duals.append(cvx.matrix(dual_values))

                    elif constr.typeOfConstraint == 'RScone':
                        dual_values = []
                        rhs_id = [id for id in constr.Id['gurobi'] if '_rhs_' in id][0]
                        dual_values.append(self.grbcons[rhs_id].pi)
                        dim = 1 + constr.Exp1.size[0] * constr.Exp1.size[1]
                        lhs_id = [id for id in constr.Id['gurobi'] if '_lhs_' in id]
                        assert dim == len(lhs_id)
                        for ids in lhs_id:
                            dual_values.append(-self.grbcons[ids].pi)
                        if self.objective[0] == 'min':
                            duals.append(-cvx.matrix(dual_values))
                        else:
                            duals.append(cvx.matrix(dual_values))

                    else:
                        if self.options['verbose'] > 0:
                            print(
                                'duals for this type of constraint not supported yet')
                        duals.append(None)

            except Exception as ex:
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Dual Solution not found\033[0m")
        #-----------------#
        # return statement#
        #-----------------#

        sol = {'gurobi_model': m, 'status': status, 'time': tend - tstart}

        return (primals, duals, obj, sol)

    def _mosek_solve(self):
        """
        Solves the problem with mosek
        """

        #----------------------------#
        #  LOAD MOSEK OR MOSEK 7     #
        #----------------------------#

        # force to use version 6.0 of mosek.
        if self.options['solver'] == 'mosek6':
            import mosek as mosek
            version7 = not(hasattr(mosek, 'cputype'))
            if version7:
                raise ImportError(
                    "I couldn't find mosek 6.0; the package named mosek is the v7.0")
        # try to load mosek7, else use the default mosek package (which can be
        # any version)
        else:
            try:
                import mosek7 as mosek
            except ImportError:
                try:
                    import mosek as mosek
                    # True if this is the version 7 of MOSEK
                    version7 = not(hasattr(mosek, 'cputype'))
                    if self.options['solver'] == 'mosek7' and not(version7):
                        print(
                            "\033[1;31m mosek7 not found. using default mosek instead.\033[0m")
                except:
                    raise ImportError('mosek library not found')

        # True if this is the version 7 of MOSEK
        version7 = not(hasattr(mosek, 'cputype'))

        #-------------------------------#
        #  Can we solve it with mosek ? #
        #-------------------------------#
        if self.type in ('unknown type', 'MISDP', 'GP'):
            raise NotAppropriateSolverError(
                "'mosek' cannot solve problems of type {0}".format(
                    self.type))

        elif (self.type in ('SDP', 'ConeP')) and not(version7):
            raise NotAppropriateSolverError(
                "This version of mosek does not support SDP. Try with mosek v7.0")

        elif self.type in ('Mixed (SDP+quad)', 'Mixed (SOCP+quad)', 'Mixed (MISOCP+quad)', 'MIQCP', 'MIQP'):
            raise QuadAsSocpError(
                'Please convert the quadratic constraints as cone constraints ' +
                'with the function convert_quad_to_socp().')

        #----------------------------#
        #  create the mosek instance #
        #----------------------------#

        self._make_mosek_instance()
        task = self.msk_task

        if self.options['verbose'] > 0:
            if version7:
                print('-----------------------------------')
                print('         MOSEK version 7')
                print('-----------------------------------')
            else:
                print('-----------------------------------')
                print('            MOSEK solver')
                print('-----------------------------------')

        #---------------------#
        #  setting parameters #
        #---------------------#

        # tolerance (conic + LP interior points)
        task.putdouparam(mosek.dparam.intpnt_tol_dfeas, self.options['tol'])
        task.putdouparam(mosek.dparam.intpnt_tol_pfeas, self.options['tol'])
        task.putdouparam(mosek.dparam.intpnt_tol_mu_red, self.options['tol'])
        task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, self.options['tol'])

        task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas, self.options['tol'])
        task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas, self.options['tol'])
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_mu_red,
            self.options['tol'])
        task.putdouparam(
            mosek.dparam.intpnt_co_tol_rel_gap,
            self.options['tol'])

        # tolerance (interior points)
        task.putdouparam(mosek.dparam.mio_tol_rel_gap, self.options['gaplim'])

        # maxiters
        if not(self.options['maxit'] is None):
            task.putintparam(
                mosek.iparam.intpnt_max_iterations,
                self.options['maxit'])
            task.putintparam(
                mosek.iparam.sim_max_iterations,
                self.options['maxit'])

        # lpmethod
        if not self.options['lp_node_method'] is None:
            if self.options['lp_node_method'] == 'interior':
                task.putintparam(
                    mosek.iparam.mio_node_optimizer,
                    mosek.optimizertype.intpnt)
            elif self.options['lp_node_method'] == 'psimplex':
                task.putintparam(
                    mosek.iparam.mio_node_optimizer,
                    mosek.optimizertype.primal_simplex)
            elif self.options['lp_node_method'] == 'dsimplex':
                task.putintparam(
                    mosek.iparam.mio_node_optimizer,
                    mosek.optimizertype.dual_simplex)
            else:
                raise Exception('unexpected value for option lp_node_method')
        if not self.options['lp_root_method'] is None:
            if self.options['lp_root_method'] == 'interior':
                task.putintparam(
                    mosek.iparam.mio_root_optimizer,
                    mosek.optimizertype.intpnt)
                if self.type == 'LP':
                    task.putintparam(
                        mosek.iparam.optimizer,
                        mosek.optimizertype.intpnt)
            elif self.options['lp_root_method'] == 'psimplex':
                task.putintparam(
                    mosek.iparam.mio_root_optimizer,
                    mosek.optimizertype.primal_simplex)
                if self.type == 'LP':
                    task.putintparam(
                        mosek.iparam.optimizer,
                        mosek.optimizertype.primal_simplex)
            elif self.options['lp_root_method'] == 'dsimplex':
                task.putintparam(
                    mosek.iparam.mio_root_optimizer,
                    mosek.optimizertype.dual_simplex)
                if self.type == 'LP':
                    task.putintparam(
                        mosek.iparam.optimizer,
                        mosek.optimizertype.dual_simplex)
            else:
                raise Exception('unexpected value for option lp_root_method')

        if not self.options['timelimit'] is None:
            task.putdouparam(
                mosek.dparam.mio_max_time,
                self.options['timelimit'])
            task.putdouparam(
                mosek.dparam.optimizer_max_time,
                self.options['timelimit'])
            # task.putdouparam(mosek.dparam.mio_max_time_aprx_opt,self.options['timelimit'])
        else:
            task.putdouparam(mosek.dparam.mio_max_time, -1.0)
            task.putdouparam(mosek.dparam.optimizer_max_time, -1.0)
            # task.putdouparam(mosek.dparam.mio_max_time_aprx_opt,-1.0)

        # number feasible solutions
        if not self.options['nbsol'] is None:
            task.putintparam(
                mosek.iparam.mio_max_num_solutions,
                self.options['nbsol'])

        # hotstart
        if self.options['hotstart']:
            task.putintparam(mosek.iparam.mio_construct_sol, mosek.onoffkey.on)

        for par, val in six.iteritems(self.options['mosek_params']):
            try:
                mskpar = eval('mosek.iparam.' + par)
                task.putintparam(mskpar, val)
            except AttributeError:
                try:
                    mskpar = eval('mosek.dparam.' + par)
                    task.putdouparam(mskpar, val)
                except AttributeError:
                    raise Exception('unknown mosek parameter')

        #--------------------#
        #  call the solver   #
        #--------------------#

        import time
        tstart = time.time()

        # optimize
        try:
            task.optimize()
        except mosek.Error as ex:
            # catch non-convexity exception
            if self.numberQuadConstraints > 0 and (str(ex) == '(0) ' or str(
                    ex).startswith('(1296)') or str(ex).startswith('(1295)')):
                raise NonConvexError(
                    'Error raised during solve. Problem nonconvex ?')
            else:
                print("Error raised during solve")

        tend = time.time()

        # Print a summary containing information
        # about the solution for debugging purposes
        task.solutionsummary(mosek.streamtype.msg)
        prosta = []
        solsta = []

        if self.is_continuous():
            if not(self.options['lp_root_method'] is None) and (
                    self.options['lp_root_method'].endswith('simplex')):
                soltype = mosek.soltype.bas
            else:
                soltype = mosek.soltype.itr
            intg = False
        else:
            soltype = mosek.soltype.itg
            intg = True

        if version7:
            solsta = task.getsolsta(soltype)
        else:
            [prosta, solsta] = task.getsolutionstatus(soltype)
        status = repr(solsta)
        #----------------------#
        # retrieve the primals #
        #----------------------#
        # OBJ
        try:
            obj = task.getprimalobj(soltype)
        except Exception as ex:
            obj = None
            if self.options['verbose'] > 0:
                print("\033[1;31m*** Primal Solution not found\033[0m")

        # PRIMAL VARIABLES

        primals = {}

        if 'noprimals' in self.options and self.options['noprimals']:
            pass
        else:
            if self.options['verbose'] > 0:
                print('Solution status is ' + repr(solsta))
            try:
                # Output a solution
                indices = [(v.startIndex, v.endIndex, v)
                           for v in self.variables.values()]
                indices = sorted(indices)
                if self.options['handleBarVars']:
                    idxsdpvars = [
                        (si, ei) for (
                            si, ei, v) in indices[
                            ::-1] if v.semiDef]
                    indsdpvar = [i for i, cons in
                                 enumerate([cs for cs in self.constraints if cs.typeOfConstraint.startswith('sdp')])
                                 if cons.semidefVar]
                    isdpvar = 0
                else:
                    idxsdpvars = []
                for si, ei, var in indices:
                    if self.options['handleBarVars'] and var.semiDef:
                        #xjbar = np.zeros(int((var.size[0]*(var.size[0]+1))/2),float)
                        xjbar = [0.] * \
                            int((var.size[0] * (var.size[0] + 1)) // 2)
                        task.getbarxj(
                            mosek.soltype.itr, indsdpvar[isdpvar], xjbar)
                        xjbar = ltrim1(cvx.matrix(xjbar))
                        primals[var.name] = cvx.matrix(xjbar, var.size)
                        isdpvar += 1

                    else:
                        #xx = np.zeros((ei-si),float)
                        # list instead of np.zeros to avoid PEEP 3118 buffer
                        # warning
                        xx = [0.] * (ei - si)
                        (nsi, eim), _, _ = self._separate_linear_cons(
                            [si, ei - 1], [0, 0], idxsdpvars)
                        task.getsolutionslice(
                            soltype, mosek.solitem.xx, nsi, eim + 1, xx)
                        scaledx = [
                            (j, v) for (
                                j, v) in six.iteritems(
                                self.msk_scaledcols) if j >= si and j < ei]
                        # do the change of variable the other way around.
                        for (j, v) in scaledx:
                            xx[j - si] /= v
                        if var.vtype in ('symmetric',):
                            xx = svecm1(cvx.matrix(xx))
                        primals[var.name] = cvx.matrix(xx, var.size)

                """OLD VERSION, but too slow
                                xx = np.zeros(self.numberOfVars, float)
                                task.getsolutionslice(soltype,
                                        mosek.solitem.xx, 0,self.numberOfVars, xx)

                                for var in self.variables.keys():
                                        si=self.variables[var].startIndex
                                        ei=self.variables[var].endIndex
                                        varvect=xx[si:ei]
                                        if self.variables[var].vtype in ('symmetric',):
                                                varvect=svecm1(cvx.matrix(varvect)) #varvect was the svec
                                                                                #representation of X
                                        primals[var]=cvx.matrix(varvect, self.variables[var].size)
                                """
            except Exception as ex:
                primals = {}
                obj = None
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Primal Solution not found\033[0m")

        #--------------------#
        # retrieve the duals #
        #--------------------#
        duals = []
        if intg or ('noduals' in self.options and self.options['noduals']):
            pass
        else:
            try:
                if self.options['handleBarVars']:
                    idvarcone = int(_bsum([(var.endIndex - var.startIndex)
                                           for var in self.variables.values() if not(var.semiDef)]))
                else:
                    idvarcone = self.numberOfVars  # index of variables in cone

                # index of equality constraint in mosekcons (without fixed
                # vars)
                idconin = 0
                idin = 0  # index of inequality constraint in cvxoptVars['Gl']
                idcone = 0  # number of seen cones
                idsdp = 0  # number of seen sdp cons
                szcones = [
                    ((cs.Exp1.size[0] *
                      cs.Exp1.size[1] +
                        2) if cs.Exp3 else (
                        cs.Exp1.size[0] *
                        cs.Exp1.size[1] +
                        1)) for cs in self.constraints if cs.typeOfConstraint.endswith('cone')]

                seen_bounded_vars = []

                # now we parse the constraints
                for k, cons in enumerate(self.constraints):
                    # conic constraint
                    if cons.typeOfConstraint[2:] == 'cone':
                        szcone = szcones[idcone]
                        fxd = self.msk_fxdconevars[idcone]
                        # v=np.zeros(szcone,float)
                        v = [0.] * (szcone - len(fxd))
                        task.getsolutionslice(soltype, mosek.solitem.snx,
                                              idvarcone, idvarcone + len(v), v)
                        for i, j in fxd:
                            vj = [0.]
                            task.getsolutionslice(soltype, mosek.solitem.snx,
                                                  j, j + 1, vj)
                            v.insert(i, vj[0])

                        if cons.typeOfConstraint.startswith('SO'):
                            duals.append(cvx.matrix(v))
                            duals[-1][0] = -duals[-1][0]
                        else:
                            vr = [-0.25 * v[0] - 0.5 * v[1]] + [0.5 *
                                                                vi for vi in v[2:]] + [-0.25 * v[0] + 0.5 * v[1]]
                            duals.append(cvx.matrix(vr))
                        idvarcone += szcone - len(fxd)
                        idconin += szcone - len(fxd)
                        idcone += 1

                    elif cons.typeOfConstraint == 'lin=':
                        szcons = int(np.product(cons.Exp1.size))
                        fxd = self.msk_fxd[k]
                        # v=np.zeros(szcons-len(fxd),float)
                        v = [0.] * (szcons - len(fxd))
                        if len(v) > 0:
                            task.getsolutionslice(
                                soltype, mosek.solitem.y, idconin, idconin + szcons - len(fxd), v)
                        for (
                                l,
                                var,
                                coef) in fxd:  # dual of fixed var constraints
                            duu = [0.]
                            dul = [0.]
                            # duu=np.zeros(1,float)
                            # dul=np.zeros(1,float)
                            task.getsolutionslice(soltype, mosek.solitem.sux,
                                                  var, var + 1, duu)
                            task.getsolutionslice(soltype, mosek.solitem.slx,
                                                  var, var + 1, dul)
                            if (var not in seen_bounded_vars):
                                v.insert(l, (dul[0] - duu[0]) / coef)
                                seen_bounded_vars.append(var)
                            else:
                                v.insert(l, 0.)
                        duals.append(cvx.matrix(v))
                        idin += szcons
                        idconin += (szcons - len(fxd))

                    elif cons.typeOfConstraint[:3] == 'lin':  # inequality
                        szcons = int(np.product(cons.Exp1.size))
                        fxd = self.msk_fxd[k]
                        # v=np.zeros(szcons-len(fxd),float)
                        v = [0.] * (szcons - len(fxd))
                        if len(v) > 0:
                            task.getsolutionslice(
                                soltype, mosek.solitem.y, idconin, idconin + szcons - len(fxd), v)
                        if cons.typeOfConstraint[3] == '>':
                            v = [-vi for vi in v]
                        for (
                                l,
                                var,
                                coef) in fxd:  # dual of simple bound constraints
                            # du=np.zeros(1,float)
                            du = [0.]
                            bound = (cons.Exp2 - cons.Exp1).constant
                            if bound is None:
                                bound = 0
                            elif cons.typeOfConstraint[3] == '>':
                                bound = -bound[l] / float(coef)
                            else:
                                bound = bound[l] / float(coef)

                            bk, bl, bu = task.getbound(mosek.accmode.var, var)
                            duu = [0.]
                            dul = [0.]
                            # duu=np.zeros(1,float)
                            # dul=np.zeros(1,float)
                            task.getsolutionslice(soltype, mosek.solitem.sux,
                                                  var, var + 1, duu)
                            task.getsolutionslice(soltype, mosek.solitem.slx,
                                                  var, var + 1, dul)

                            if coef > 0:  # upper bound
                                if bound == bu and (var not in seen_bounded_vars) and(
                                        abs(duu[0]) > 1e-8
                                        and abs(dul[0]) < 1e-5
                                        and abs(duu[0]) > abs(dul[0])):  # active bound:
                                    v.insert(l, -duu[0] / coef)
                                    seen_bounded_vars.append(var)
                                else:
                                    v.insert(
                                        l, 0.)  # inactive bound, or active already seen
                            else:  # lower bound
                                if bound == bl and (var not in seen_bounded_vars) and(
                                        abs(dul[0]) > 1e-8
                                        and abs(duu[0]) < 1e-5
                                        and abs(dul[0]) > abs(duu[0])):  # active bound
                                    v.insert(l, dul[0] / coef)
                                    seen_bounded_vars.append(var)
                                else:
                                    v.insert(
                                        l, 0.)  # inactive bound, or active already seen
                        duals.append(cvx.matrix(v))
                        idin += szcons
                        idconin += (szcons - len(fxd))

                    elif cons.typeOfConstraint[:3] == 'sdp':
                        sz = cons.Exp1.size
                        sz = cons.Exp1.size
                        xx = [0.] * ((sz[0] * (sz[0] + 1)) // 2)
                        # xx=np.zeros((sz[0]*(sz[0]+1))/2,float)
                        task.getbarsj(mosek.soltype.itr, idsdp, xx)
                        idsdp += 1
                        M = ltrim1(cvx.matrix(xx))
                        duals.append(-cvx.matrix(M))
                    else:
                        if self.options['verbose'] > 0:
                            print('dual for this constraint is not handled yet')
                        duals.append(None)
                if self.objective[0] == 'min':
                    duals = [-d for d in duals]
            except Exception as ex:
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Dual Solution not found\033[0m")
                duals = []
        #-----------------#
        # return statement#
        #-----------------#
        # OBJECTIVE
        sol = {'mosek_task': task, 'status': status, 'time': tend - tstart}

        return (primals, duals, obj, sol)

    def _zibopt_solve(self):

        if self.type in (
               'unknown type',
               'GP',
               'SDP',
               'ConeP',
               'Mixed (SDP+quad)',
               'MISDP'):
           raise NotAppropriateSolverError(
               "'zibopt' cannot solve problems of type {0}".format(
                   self.type))

        #-----------------------------#
        #  create the zibopt instance #
        #-----------------------------#

        self._make_zibopt()

        timelimit = 10000000.
        gaplim = self.options['tol']
        nbsol = -1
        if not self.options['timelimit'] is None:
            timelimit = self.options['timelimit']
        if not(self.options['gaplim'] is None or self.is_continuous()):
            gaplim = self.options['gaplim']
        if not self.options['nbsol'] is None:
            nbsol = self.options['nbsol']

        #--------------#
        # set options  #
        #--------------#

        try:
            self.scip_model.setRealParam('numerics/barrierconvtol',gaplim)
            if self.options['feastol']:
                self.scip_model.setRealParam('numerics/feastol',self.options['feastol'])
            self.scip_model.setRealParam('limits/time',timelimit)
            self.scip_model.setIntParam('limits/solutions',nbsol)
            if self.options['treememory']:
                self.scip_model.setRealParam('limits/memory',self.options['treememory'])

            for par, val in six.iteritems(self.options['scip_params']):
                if isinstance(par,bool):
                    self.scip_model.setBoolParam(par,val)
                elif isinstance(par,str):
                    try:
                        self.scip_model.setStringParam(par,val)
                    except:
                        self.scip_model.setCharParam(par,val)
                elif isinstance(par,float):
                    self.scip_model.setRealParam(par,val)
                elif isinstance(par,int):
                    try:
                        self.scip_model.setIntParam(par,val)
                    except:
                        self.scip_model.setLongintParam(par,val)

        except ValueError as e:
            print('Warning: some options were not set !')
            print(e)


        #--------------------#
        #  call the solver   #
        #--------------------#

        import time
        tstart = time.time()
        self.scip_model.optimize()
        tend = time.time()

        status = self.scip_model.getStatus()

        if self.options['verbose'] > 0:
            print('zibopt solution status: ' + status)

        #----------------------#
        # retrieve the primals #
        #----------------------#
        primals = {}
        obj = self.scip_model.getObjVal()
        if 'noprimals' in self.options and self.options['noprimals']:
            pass
        else:
            try:
                primals = {}
                for var in self.variables.values():
                    si = var.scip_startIndex
                    ei = si + var.size[0]*var.size[1]

                    varvect = self.scip_vars[si:ei]
                    value = [self.scip_model.getVal(v) for v in varvect]

                    if var.vtype in ('symmetric',):
                        value = svecm1(cvx.matrix(value))  # value was the svec
                        # representation of X
                    primals[var.name] = cvx.matrix(value,var.size)

            except Exception as ex:
                primals = {}
                obj = None
                if self.options['verbose'] > 0:
                    print("\033[1;31m*** Primal Solution not found\033[0m")

        #----------------------#
        # retrieve the duals #
        #----------------------#

        # not available by python-zibopt (yet ? )
        duals = []

        #------------------#
        # return statement #
        #------------------#

        solt = {}
        solt['status'] = status
        solt['time'] = tend - tstart
        return (primals, duals, obj, solt)

    def _sdpa_solve(self):
        """
        Solves the problem with SDPA
        """

        #-------------------------------#
        #  Can we solve it with SDPA? #
        #-------------------------------#
        if self.type in ( 'unknown type', 'GP', 'general-obj', 'MIP', 'MIQCP', 'MIQP',
                         'Mixed (MISOCP+quad)', 'MISOCP',
                         #'Mixed (SOCP+quad)', 'Mixed (SDP+quad)',
                         #'LP', 'QCQP','QP', 'SOCP', 'ConeP',
                         ):
            raise NotAppropriateSolverError(
                "'SDPA' cannot solve problems of type {0}".format(
                    self.type))
        #-----------------------------#
        # create the sdpaopt instance #
        #-----------------------------#
        import time
        import os
        #--------------------#
        #  call the solver   #
        #--------------------#
        if 'read_solution' in self.options['sdpa_params']:
            tstart = time.time()
            self.sdpa_out_filename = self.options['sdpa_params']['read_solution']
            if not any(self.cvxoptVars.values()):
                self._make_cvxopt_instance(hard_coded_bounds=True)
        else:
            from subprocess import call
            self._make_sdpaopt(self.options['sdpa_executable'])
            tstart = time.time()
            params = [self.sdpa_executable, '-ds', self.sdpa_dats_filename,
                      '-o', self.sdpa_out_filename]
            for key in self.options['sdpa_params']:
                params += [key, str(self.options['sdpa_params'][key])]
            if self.options['verbose'] >= 1:
                call(params)
            else:
                with open(os.devnull, "w") as fnull:
                    call(params, stdout=fnull, stderr=fnull)
        tend = time.time()
        #-----------------------#
        # retrieve the solution #
        #-----------------------#
        file_ = open(self.sdpa_out_filename, 'r')
        for line in file_:
            if line.find("phase.value") > -1:
                if line.find("pdOPT") > -1:
                    status = 'optimal'
                elif line.find("pdFEAS") > -1:
                    status = 'primal-dual feasible'
                elif line.find("INF") > -1:
                    status = 'infeasible'
                elif line.find("UNBD") > -1:
                    status = 'unbounded'
                else:
                    status = 'unknown'
            if line.find("objValPrimal") > -1:
                obj = float((line.split())[2])
                if self.objective[0]=='max':
                    obj *= -1
            if line.find("xVec =") > -1:
                line = six.next(file_)
                x_vec = [
                    float(x) for x in line[
                        line.rfind('{') + 1:line.find('}')].strip().split(',')]
            if line.find("yMat =") > -1:
                dual_solution = []
                while True:
                    sol_mat = None
                    in_matrix = False
                    i = 0
                    for row in file_:
                        if row.find('}') < 0:
                            continue
                        if row.startswith('}'):
                            break
                        if row.find('{') != row.rfind('{'):
                            in_matrix = True
                        numbers = row[row.rfind('{')+1:
                                      row.find('}')].strip().split(',')
                        if sol_mat is None:
                            if in_matrix:
                                sol_mat = cvx.matrix(0.0, (len(numbers),
                                                           len(numbers)))
                            else:
                                sol_mat = cvx.matrix(0.0, (1, len(numbers)))
                        for j, number in enumerate(numbers):
                            sol_mat[i, j] = float(number)
                        if row.find('}') != row.rfind('}') or not in_matrix:
                            break
                        i += 1
                    dual_solution.append(sol_mat)
                    if row.startswith('}'):
                        break
                if len(dual_solution) > 0 and dual_solution[-1] is None:
                    dual_solution = dual_solution[:-1]

        file_.close()
        if 'read_solution' not in self.options['sdpa_params']:
            os.remove(self.sdpa_dats_filename)
            os.remove(self.sdpa_out_filename)
        dims = {}
        dims['s'] = [int(np.sqrt(Gsi.size[0]))
                     for Gsi in self.cvxoptVars['Gs']]
        dims['l'] = self.cvxoptVars['Gl'].size[0]
        dims['q'] = [Gqi.size[0] for Gqi in self.cvxoptVars['Gq']]
        if self.cvxoptVars['A'].size[0] > 0:
            dims['l'] += (2 * self.cvxoptVars['A'].size[0])
        if self.options['verbose'] > 0:
            print('SDPA solution status: ' + status)
        JP = list(set(self.cvxoptVars['A'].I))
        IP = range(len(JP))
        VP = [1] * len(JP)

        # is there a constraint of the form 0==a(a not 0) ?
        if any([b for (i, b) in enumerate(
                self.cvxoptVars['b']) if i not in JP]):
            raise Exception('infeasible constraint of the form 0=a')
        from cvxopt import spmatrix
        P = spmatrix(VP, IP, JP, (len(IP), self.cvxoptVars['A'].size[0]))
        # Convert primal solution
        primals = {}
        for var in self.variables.keys():
            si = self.variables[var].startIndex
            ei = self.variables[var].endIndex
            value = x_vec[si:ei]
            if self.variables[var].vtype in ('symmetric',):
                value = svecm1(cvx.matrix(value))  # value was the svec
                # representation of X
            primals[var] = cvx.matrix(value, self.variables[var].size)
        duals = []
        if 'noduals' in self.options and self.options['noduals']:
            pass
        else:
            printnodual = False
            indy, indzl, indzq, indzs = 0, 0, 0, 0
            ieq = self.cvxoptVars['Gl'].size[0]
            neq = (dims['l'] - ieq) // 2
            if neq > 0:
                soleq = dual_solution[0][0, ieq:ieq + neq]
                soleq -= dual_solution[0][0, ieq + neq:ieq + 2 * neq]
            else:
                soleq = None
            if ieq + neq > 0:
                indzq = indzs = 1
            indzs += len(dims['q'])
            for k, consk in enumerate(self.constraints):
                # Equality
                if consk.typeOfConstraint == 'lin=':
                    if not (soleq is None):
                        consSz = np.product(consk.Exp1.size)
                        duals.append((P.T * soleq.T)[indy:indy + consSz])
                        indy += consSz
                    else:
                        printnodual = True
                        duals.append(None)
                # Inequality
                elif consk.typeOfConstraint[:3] == 'lin':
                    consSz = np.product(consk.Exp1.size)
                    duals.append(dual_solution[0][indzl:indzl + consSz])
                    indzl += consSz
                # SOCP constraint [Rotated or not]
                elif consk.typeOfConstraint[2:] == 'cone':
                    M = dual_solution[indzq]
                    duals.append(cvx.matrix([np.trace(M), -2*M[-1,:-1].T]))
                    indzq += 1
                # SDP constraint
                elif consk.typeOfConstraint[:3] == 'sdp':
                    duals.append(dual_solution[indzs])
                    indzs += 1
            if printnodual and self.options['verbose'] > 0:
                    print("\033[1;31m*** Dual Solution not found\033[0m")
        #------------------#
        # return statement #
        #------------------#
        solt = {}
        solt['status'] = status
        solt['time'] = tend - tstart
        return (primals, duals, obj, solt)

    def _sqpsolve(self, options):
        """
        Solves the problem by sequential Quadratic Programming.
        """
        import copy
        for v in self.variables:
            if self.variables[v].value is None:
                self.set_var_value(v, cvx.uniform(self.variables[v].size))
        # lower the display level for mosek
        self.options['verbose'] -= 1
        oldvar = self._eval_all()
        subprob = copy.deepcopy(self)
        if self.options['verbose'] > 0:
            print('solve by SQP method with proximal convexity enforcement')
            print('it:     crit\t\tproxF\tstep')
            print('---------------------------------------')
        converged = False
        k = 1
        while not converged:
            obj, grad, hess = self.objective[
                1].fun(self.objective[1].Exp.eval())
            diffExp = self.objective[1].Exp - self.objective[1].Exp.eval()
            quadobj0 = obj + grad.T * diffExp + 0.5 * diffExp.T * hess * diffExp
            proxF = self.options['step_sqp']
            # VARIANT IN CONSTRAINTS: DO NOT FORCE CONVEXITY...
            # for v in subprob.variables.keys():
            #        x=subprob.get_varExp(v)
            #        x0=self.get_variable(v).eval()
            #        subprob.add_constraint((x-x0).T*(x-x0)<0.5)
            solFound = False
            while (not solFound):
                if self.objective[0] == 'max':
                    # (proximal + force convexity)
                    quadobj = quadobj0 - proxF * abs(diffExp)**2
                else:
                    # (proximal + force convexity)
                    quadobj = quadobj0 + proxF * abs(diffExp)**2
                subprob = copy.deepcopy(self)
                subprob.set_objective(self.objective[0], quadobj)
                if self.options['harmonic_steps'] and k > 1:
                    for v in subprob.variables.keys():
                        x = subprob.get_varExp(v)
                        x0 = self.get_variable(v).eval()
                        subprob.add_constraint(
                            (x - x0).T * (x - x0) < (10. / float(k - 1)))
                try:
                    sol = subprob.solve()
                    solFound = True
                except Exception as ex:
                    if str(ex)[:6] == '(1296)':  # function not convex
                        proxF *= (1 + cvx.uniform(1))
                    else:
                        # reinit the initial verbosity
                        self.options['verbose'] += 1
                        raise
            if proxF >= 100 * self.options['step_sqp']:
                # reinit the initial verbosity
                self.options['verbose'] += 1
                raise Exception(
                    'function not convex before proxF reached 100 times the initial value')

            for v in subprob.variables:
                self.set_var_value(v, subprob.get_valued_variable(v))
            newvar = self._eval_all()
            step = np.linalg.norm(newvar - oldvar)
            if isinstance(step, cvx.matrix):
                step = step[0]
            oldvar = newvar
            if self.options['verbose'] > 0:
                if k == 1:
                    print(
                        '  {0}:         --- \t{1:6.3f} {2:10.4e}'.format(k, proxF, step))
                else:
                    print(
                        '  {0}:   {1:16.9e} {2:6.3f} {3:10.4e}'.format(
                            k, obj, proxF, step))
            k += 1
            # have we converged ?
            if step < self.options['tol']:
                converged = True
            if k > self.options['maxit']:
                converged = True
                print('Warning: no convergence after {0} iterations'.format(k))

        # reinit the initial verbosity
        self.options['verbose'] += 1
        sol['lastStep'] = step
        return sol

    def what_type(self):

        iv = [v for v in self.variables.values() if v.vtype not in (
            'continuous', 'symmetric', 'hermitian', 'complex')]
        # continuous problem
        if len(iv) == 0:
            # general convex
            if not(
                    self.objective[1] is None) and isinstance(
                    self.objective[1],
                    GeneralFun):
                return 'general-obj'
            # GP
            if self.numberLSEConstraints > 0:
                if (self.numberConeConstraints == 0
                        and self.numberQuadConstraints == 0
                        and self.numberSDPConstraints == 0):
                    return 'GP'
                else:
                    return 'unknown type'
            # SDP
            if self.numberSDPConstraints > 0:
                if (self.numberConeConstraints == 0
                        and self.numberQuadConstraints == 0):
                    return 'SDP'
                elif self.numberQuadConstraints == 0:
                    return 'ConeP'
                else:
                    return 'Mixed (SDP+quad)'
            # SOCP
            if self.numberConeConstraints > 0:
                if self.numberQuadConstraints == 0:
                    return 'SOCP'
                else:
                    return 'Mixed (SOCP+quad)'

            # quadratic problem
            if self.numberQuadConstraints > 0:
                if any([cs.typeOfConstraint == 'quad' for cs in self.constraints]):
                    return 'QCQP'
                else:
                    return 'QP'

            return 'LP'
        else:
            if not(
                    self.objective[1] is None) and isinstance(
                    self.objective[1],
                    GeneralFun):
                return 'unknown type'
            if self.numberLSEConstraints > 0:
                return 'unknown type'
            if self.numberSDPConstraints > 0:
                return 'MISDP'
            if self.numberConeConstraints > 0:
                if self.numberQuadConstraints == 0:
                    return 'MISOCP'
                else:
                    return 'Mixed (MISOCP+quad)'
            if self.numberQuadConstraints > 0:
                if any([cs.typeOfConstraint == 'quad' for cs in self.constraints]):
                    return 'MIQCP'
                else:
                    return 'MIQP'
            return 'MIP'  # (or simply IP)

    def set_type(self, value):
        raise AttributeError('type is not writable')

    def del_type(self):
        raise AttributeError('type is not writable')

    type = property(what_type, set_type, del_type)
    """Type of Optimization Problem ('LP', 'MIP', 'SOCP', 'QCQP',...)"""

    def solver_selection(self):
        """Selects an appropriate solver for this problem
        and sets the option ``'solver'``.
        """
        tp = self.type
        if tp == 'LP':
            order = [
                'cplex',
                'gurobi',
                'mosek7',
                'mosek6',
                'zibopt',
                'glpk',
                'cvxopt',
                'smcp']
        elif tp in ('QCQP,QP'):
            order = ['cplex', 'mosek7', 'mosek6', 'gurobi', 'cvxopt', 'zibopt']
        elif tp == 'SOCP':
            order = [
                'mosek7',
                'mosek6',
                'cplex',
                'gurobi',
                'cvxopt',
                'smcp',
                'zibopt']
        elif tp == 'SDP':
            order = ['mosek7', 'cvxopt', 'sdpa', 'smcp']
        elif tp == 'ConeP':
            order = ['mosek7', 'cvxopt', 'smcp']
        elif tp == 'GP':
            order = ['cvxopt']
        elif tp == 'general-obj':
            order = [
                'cplex',
                'mosek7',
                'mosek6',
                'gurobi',
                'zibopt',
                'cvxopt',
                'smcp']
        elif tp == 'MIP':
            order = ['cplex', 'gurobi', 'mosek7', 'mosek6', 'zibopt', 'glpk']
        elif tp in ('MIQCP', 'MIQP'):
            order = ['cplex', 'gurobi', 'mosek7', 'mosek6', 'zibopt']
        elif tp == 'Mixed (SOCP+quad)':
            order = ['mosek7', 'mosek6', 'cplex', 'gurobi', 'cvxopt', 'smcp']
        elif tp in ('MISOCP', 'Mixed (MISOCP+quad)'):
            order = ['mosek7', 'mosek6', 'cplex', 'gurobi']
        elif tp == 'Mixed (SDP+quad)':
            order = ['mosek7', 'cvxopt', 'smcp']
        else:
            raise Exception(
                'no solver available for problem of type {0}'.format(tp))
        avs = available_solvers()
        for sol in order:
            if sol in avs:
                self.set_option('solver', sol)
                return
        #not found
        raise NotAppropriateSolverError(
            'no solver available for problem of type {0}'.format(tp))

    def write_to_file(self, filename, writer='picos'):
        """
        This function writes the problem to a file.

        :param filename: The name of the file where the problem will be saved. The
                         extension of the file (if provided) indicates the format
                         of the export:

                                * ``'.cbf'``: CBF (Conic Benchmark Format). This format
                                  is suitable for optimization problems involving
                                  second order and/or semidefinite cone constraints. This is
                                  the standard to use for conic optimization problems,
                                  cf. `CBLIB <http://cblib.zib.de/>`_ and
                                  `this paper <http://www.optimization-online.org/DB_HTML/2014/03/4301.html>`_ from Henrik Friberg.

                                * ``'.lp'``: `LP format <http://docs.mosek.com/6.0/pyapi/node022.html>`_
                                  . This format handles only linear constraints, unless the writer ``'cplex'``
                                  is used, and the file is saved in the extended
                                  `cplex LP format <http://pic.dhe.ibm.com/infocenter/cplexzos/v12r4/index.jsp?topic=%2Fcom.ibm.cplex.zos.help%2Fhomepages%2Freffileformatscplex.html>`_

                                * ``'.mps'``: `MPS format <http://docs.mosek.com/6.0/pyapi/node021.html>`_
                                  (recquires mosek, gurobi or cplex).

                                * ``'.opf'``: `OPF format <http://docs.mosek.com/6.0/pyapi/node023.html>`_
                                  (recquires mosek).

                                * ``'.dat-s'``: `sparse SDPA format <http://sdpa.indsys.chuo-u.ac.jp/sdpa/download.html#sdpa>`_
                                  This format is suitable to save semidefinite programs (SDP). SOC constraints are
                                  stored as semidefinite constraints with an *arrow pattern*.

        :type filename: str.
        :param writer: The default writer is ``picos``, which has its own *LP*, *CBF*, and
                       *sparse SDPA* write functions. If cplex, mosek or gurobi is installed,
                       the user can pass the option ``writer='cplex'``, ``writer='gurobi'`` or
                       ``writer='mosek'``, and the write function of this solver
                       will be used.
        :type writer: str.

        .. Warning :: * In the case of a SOCP, when the selected writer is ``'mosek'``, the written file may
                        contain some changes of variables with respect to the original formulation
                        when the option ``handleConeVars`` is set to ``True`` (this is the default).

                        If this is an issue, turn the option ``handleConeVars`` to ``False`` and reset the
                        mosek instance by calling :func:`reset_mosek_instance() <picos.Problem.reset_mosek_instance>`,
                        but turning off this option may increase the number of variables and constraints.

                        Otherwise, the set of change of variables can be queried by ``self.msk_scaledcols``.
                        Each (Key,Value) pair ``i -> alpha`` of this dictionary indicates that
                        the ``i`` th column has been rescaled by a factor ``alpha``.


                      * The CBF writer tries to write symmetric variables :math:`X` in
                        the section ``PSDVAR`` of the .cbf file. However, this is possible
                        only if the constraint :math:`X\succeq 0` appears in the problem,
                        and no other LMI involves :math:`X` . If these two conditions are
                        not satisfied, then the symmetric-vectorization of :math:`X` is
                        used as a (free) variable of the section ``VAR`` in the .cbf file (cf. next paragraph).

                      * For problems involving a symmetric matrix variable :math:`X`
                        (typically, semidefinite programs), the expressions
                        involving :math:`X` are stored in PICOS as a function of
                        :math:`svec(X)`, the symmetric vectorized form of
                        X (see `Dattorro, ch.2.2.2.1 <http://meboo.convexoptimization.com/Meboo.html>`_).
                        As a result, the symmetric matrix variables
                        are written in :math:`svec()` form in the files created by this function.
                        So if you use another solver to solve
                        a problem that is described in a file created by PICOS, the optimal symmetric variables
                        returned will also be in symmetric vectorized form.
        """
        if self.numberLSEConstraints:
            raise Exception('gp are not supported')
        if not(
                self.objective[1] is None) and isinstance(
                self.objective[1],
                GeneralFun):
            raise Exception('general-obj are not supported')

        # automatic extension recognition
        if not(filename[-4:] in ('.mps', '.opf', '.cbf') or
               filename[-3:] == '.lp' or
               filename[-6:] == '.dat-s'):
            if writer in ('mosek', 'gurobi'):
                if (self.numberSDPConstraints > 0):
                    raise Exception('no sdp with mosek/gurobi')
                if (self.numberConeConstraints +
                        self.numberQuadConstraints) == 0:
                    filename += '.lp'
                else:
                    filename += '.mps'
            elif writer == 'cplex':
                if (self.numberSDPConstraints > 0):
                    raise Exception('no sdp with cplex')
                else:
                    filename += '.lp'
            elif writer == 'picos':
                if (self.numberQuadConstraints > 0):
                    if self.options['convert_quad_to_socp_if_needed']:
                        pcop = self.copy()
                        pcop.convert_quad_to_socp()
                        pcop.write_to_file(filename, writer)
                        return
                    else:
                        raise QuadAsSocpError(
                            'no quad constraints in sdpa format.' +
                            ' Try to convert to socp with the function convert_quad_to_socp().')
                if (self.numberConeConstraints +
                        self.numberSDPConstraints) == 0:
                    filename += '.lp'
                elif self.numberConeConstraints == 0:
                    filename += '.dat-s'
                else:
                    filename += '.cbf'
            else:
                raise Exception('unexpected writer')

        if writer == 'cplex':
            if self.cplex_Instance is None:
                self._make_cplex_instance()
            self.cplex_Instance.write(filename)
        elif writer == 'mosek':
            if self.msk_task is None:
                self._make_mosek_instance()
            self.msk_task.writedata(filename)
        elif writer == 'gurobi':
            if self.gurobi_Instance is None:
                self._make_gurobi_instance()
            self.gurobi_Instance.write(filename)
        elif writer == 'picos':
            if filename[-3:] == '.lp':
                self._write_lp(filename)
            elif filename[-6:] == '.dat-s':
                self._write_sdpa(filename)
            elif filename[-4:] == '.cbf':
                self._write_cbf(filename)
            else:
                raise Exception('unexpected file extension')
        else:
            raise Exception('unknown writer')

    def _write_lp(self, filename):
        """
        writes problem in  lp format
        """
        # add extension
        if filename[-3:] != '.lp':
            filename += '.lp'
        # check lp compatibility
        if (self.numberConeConstraints +
                self.numberQuadConstraints +
                self.numberLSEConstraints +
                self.numberSDPConstraints) > 0:
            raise Exception('the picos LP writer only accepts (MI)LP')
        # open file
        f = open(filename, 'w')
        f.write("\\* file " + filename + " generated by picos*\\\n")
        # cvxoptVars
        if not any(self.cvxoptVars.values()):
            self._make_cvxopt_instance()
        # variable names
        varnames = {}
        for name, v in six.iteritems(self.variables):
            j = 0
            k = 0
            for i in range(v.startIndex, v.endIndex):
                if v.size == (1, 1):
                    varnames[i] = name
                elif v.size[1] == 1:
                    varnames[i] = name + '(' + str(j) + ')'
                    j += 1
                else:
                    varnames[i] = name + '(' + str(j) + ',' + str(k) + ')'
                    j += 1
                    if j == v.size[0]:
                        k += 1
                        j = 0
                varnames[i] = varnames[i].replace('[', '(')
                varnames[i] = varnames[i].replace(']', ')')
        # affexpr writer

        def affexp_writer(name, indices, coefs):
            s = ''
            s += name
            s += ' : '
            start = True
            for (i, v) in zip(indices, coefs):
                if v > 0 and not(start):
                    s += '+ '
                s += "%.12g" % v
                s += ' '
                s += varnames[i]
                # not the first term anymore
                start = False
            if not(coefs):
                s += '0.0 '
                s += varnames[0]
            return s

        print('writing problem in ' + filename + '...')

        # objective
        if self.objective[0] == 'max':
            f.write("Maximize\n")
            # max handled directly
            self.cvxoptVars['c'] = -self.cvxoptVars['c']
        else:
            f.write("Minimize\n")
        I = cvx.sparse(self.cvxoptVars['c']).I
        V = cvx.sparse(self.cvxoptVars['c']).V

        f.write(affexp_writer('obj', I, V))
        f.write('\n')

        f.write("Subject To\n")
        bounds = {}
        # equality constraints:
        Ai, Aj, Av = (self.cvxoptVars['A'].I, self.cvxoptVars[
                      'A'].J, self.cvxoptVars['A'].V)
        ijvs = sorted(zip(Ai, Aj, Av))
        del Ai, Aj, Av
        itojv = {}
        lasti = -1
        for (i, j, v) in ijvs:
            if i == lasti:
                itojv[i].append((j, v))
            else:
                lasti = i
                itojv[i] = [(j, v)]
        ieq = 0
        for i, jv in six.iteritems(itojv):
            J = [jvk[0] for jvk in jv]
            V = [jvk[1] for jvk in jv]
            if len(J) == 1:
                # fixed variable
                b = self.cvxoptVars['b'][i] / V[0]
                bounds[J[0]] = (b, b)
            else:
                # affine equality
                b = self.cvxoptVars['b'][i]
                f.write(affexp_writer('eq' + str(ieq), J, V))
                f.write(' = ')
                f.write("%.12g" % b)
                f.write('\n')
                ieq += 1

        # inequality constraints:
        Gli, Glj, Glv = (self.cvxoptVars['Gl'].I, self.cvxoptVars[
                         'Gl'].J, self.cvxoptVars['Gl'].V)
        ijvs = sorted(zip(Gli, Glj, Glv))
        del Gli, Glj, Glv
        itojv = {}
        lasti = -1
        for (i, j, v) in ijvs:
            if i == lasti:
                itojv[i].append((j, v))
            else:
                lasti = i
                itojv[i] = [(j, v)]
        iaff = 0
        for i, jv in six.iteritems(itojv):
            J = [jvk[0] for jvk in jv]
            V = [jvk[1] for jvk in jv]
            if len(J) == 1 and self.options['pass_simple_cons_as_bound'] and (not (i in [t[1]
                                           for t in self.cvxoptVars['quadcons']])):
                # bounded variable
                if J[0] in bounds:
                    bl, bu = bounds[J[0]]
                else:
                    bl, bu = -INFINITY, INFINITY
                b = self.cvxoptVars['hl'][i] / V[0]
                if V[0] > 0:
                    # less than
                    bu = min(b, bu)
                if V[0] < 0:
                    # greater than
                    bl = max(b, bl)
                bounds[J[0]] = (bl, bu)
            else:
                # affine inequality
                b = self.cvxoptVars['hl'][i]
                f.write(affexp_writer('in' + str(iaff), J, V))
                f.write(' <= ')
                f.write("%.12g" % b)
                f.write('\n')
                iaff += 1

        # bounds
        #hard-coded
        for varname in self.varNames:
            var = self.variables[varname]
            for ind, (lo, up) in six.iteritems(var.bnd):
                (clo, cup) = bounds.get(var.startIndex + ind, (None, None))
                if lo is None:
                    lo = -INFINITY
                if up is None:
                    up = INFINITY
                if clo is None:
                    clo = -INFINITY
                if cup is None:
                    cup = INFINITY
                nlo = max(clo, lo)
                nup = min(cup, up)
                bounds[var.startIndex + ind] = (nlo, nup)


        f.write("Bounds\n")
        for i in range(self.numberOfVars):
            if i in bounds:
                bl, bu = bounds[i]
            else:
                bl, bu = -INFINITY, INFINITY
            if bl == -INFINITY and bu == INFINITY:
                f.write(varnames[i] + ' free')
            elif bl == bu:
                f.write(varnames[i] + (" = %.12g" % bl))
            elif bl < bu:
                if bl == -INFINITY:
                    f.write('-inf <= ')
                else:
                    f.write("%.12g" % bl)
                    f.write(' <= ')
                f.write(varnames[i])
                if bu == INFINITY:
                    f.write('<= +inf')
                else:
                    f.write(' <= ')
                    f.write("%.12g" % bu)
            f.write('\n')

        # general integers
        f.write("Generals\n")
        for name, v in six.iteritems(self.variables):
            if v.vtype == 'integer':
                for i in range(v.startIndex, v.endIndex):
                    f.write(varnames[i] + '\n')
            if v.vtype == 'semiint' or v.vtype == 'semicont':
                raise Exception(
                    'semiint and semicont variables not handled by this LP writer')
        # binary variables
        f.write("Binaries\n")
        for name, v in six.iteritems(self.variables):
            if v.vtype == 'binary':
                for i in range(v.startIndex, v.endIndex):
                    f.write(varnames[i] + '\n')
        f.write("End\n")
        print('done.')
        f.close()

    def _write_sdpa(self, filename):
        """
        Write a problem to sdpa format

        :param problem: The PICOS problem to convert.
        :type problem: :class:`picos.Problem`.
        :param filename: The name of the file. It must have the suffix ".dat-s"
        :type filename: str.

        """
        #--------------------#
        # makes the instance #
        #--------------------#
        if not any(self.cvxoptVars.values()):
            self._make_cvxopt_instance(hard_coded_bounds=True)
        dims = {}
        dims['s'] = [int(np.sqrt(Gsi.size[0]))
                     for Gsi in self.cvxoptVars['Gs']]
        dims['l'] = self.cvxoptVars['Gl'].size[0]
        dims['q'] = [Gqi.size[0] for Gqi in self.cvxoptVars['Gq']]
        G = self.cvxoptVars['Gl']
        h = self.cvxoptVars['hl']

        # handle the equalities as 2 ineq
        if self.cvxoptVars['A'].size[0] > 0:
            G = cvx.sparse([G, self.cvxoptVars['A']])
            G = cvx.sparse([G, -self.cvxoptVars['A']])
            h = cvx.matrix([h, self.cvxoptVars['b']])
            h = cvx.matrix([h, -self.cvxoptVars['b']])
            dims['l'] += (2 * self.cvxoptVars['A'].size[0])

        for i in range(len(dims['q'])):
            G = cvx.sparse([G, self.cvxoptVars['Gq'][i]])
            h = cvx.matrix([h, self.cvxoptVars['hq'][i]])

        for i in range(len(dims['s'])):
            G = cvx.sparse([G, self.cvxoptVars['Gs'][i]])
            h = cvx.matrix([h, self.cvxoptVars['hs'][i]])

        # Remove the lines in A and b corresponding to 0==0
        JP = list(set(self.cvxoptVars['A'].I))
        IP = range(len(JP))
        VP = [1] * len(JP)

        # is there a constraint of the form 0==a(a not 0) ?
        if any([b for (i, b) in enumerate(
                self.cvxoptVars['b']) if i not in JP]):
            raise Exception('infeasible constraint of the form 0=a')

        from cvxopt import sparse, spmatrix
        P = spmatrix(VP, IP, JP, (len(IP), self.cvxoptVars['A'].size[0]))
        self.cvxoptVars['A'] = P * self.cvxoptVars['A']
        self.cvxoptVars['b'] = P * self.cvxoptVars['b']
        c = self.cvxoptVars['c']
        #-----------------------------------------------------------#
        # make A,B,and blockstruct.                                 #
        # This code is a modification of the conelp function in smcp#
        #-----------------------------------------------------------#
        Nl = dims['l']
        Nq = dims['q']
        Ns = dims['s']
        if not Nl:
            Nl = 0

        P_m = G.size[1]

        P_b = -c
        P_blockstruct = []
        if Nl:
            P_blockstruct.append(-Nl)
        for i in Nq:
            P_blockstruct.append(i)
        for i in Ns:
            P_blockstruct.append(i)

        # write data
        # add extension
        if filename[-6:] != '.dat-s':
            filename += '.dat-s'
        # check lp compatibility
        if (self.numberQuadConstraints + self.numberLSEConstraints) > 0:
            if self.options['convert_quad_to_socp_if_needed']:
                pcop = self.copy()
                pcop.convert_quad_to_socp()
                pcop._write_sdpa(filename)
                return
            else:
                raise pic.QuadAsSocpError(
                    'Problem should not have quad or gp constraints. ' +
                    'Try to convert the problem to an SOCP with the function convert_quad_to_socp()')
        # open file
        f = open(filename, 'w')
        f.write('"file ' + filename + ' generated by picos"\n')
        if self.options['verbose'] >= 1:
            print('writing problem in ' + filename + '...')
        f.write(str(self.numberOfVars) + ' = number of vars\n')
        f.write(str(len(P_blockstruct)) + ' = number of blocs\n')
        # bloc structure
        f.write(str(P_blockstruct).replace('[', '(').replace(']', ')'))
        f.write(' = BlocStructure\n')
        # c vector (objective)
        f.write(str(list(-P_b)).replace('[', '{').replace(']', '}'))
        f.write('\n')
        # coefs
        for k in range(P_m + 1):
            if k != 0:
                v = sparse(G[:, k - 1])
            else:
                v = +sparse(h)

            ptr = 0
            block = 0
            # lin. constraints
            if Nl:
                u = v[:Nl]
                for i, j, value in zip(u.I, u.I, u.V):
                    f.write(
                        '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                            k, block + 1, j + 1, i + 1, -value))
                ptr += Nl
                block += 1

            # SOC constraints
            for nq in Nq:
                u0 = v[ptr]
                u1 = v[ptr + 1:ptr + nq]
                tmp = spmatrix(
                    u1.V, [nq - 1 for j in range(len(u1))], u1.I, (nq, nq))
                if not u0 == 0.0:
                    tmp += spmatrix(u0, range(nq), range(nq), (nq, nq))
                for i, j, value in zip(tmp.I, tmp.J, tmp.V):
                    f.write(
                        '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                            k, block + 1, j + 1, i + 1, -value))
                ptr += nq
                block += 1

            # SDP constraints
            for ns in Ns:
                u = v[ptr:ptr + ns**2]
                for index_k, index in enumerate(u.I):
                    j, i = divmod(index, ns)
                    if j <= i:
                        f.write(
                            '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                                k, block + 1, j + 1, i + 1, -u.V[index_k]))
                ptr += ns**2
                block += 1

        f.close()

    def _write_cbf(self, filename, uptri=False):
        """write problem data in a cbf file
        ``uptri`` specifies whether upper triangular elements of symmetric matrices are specified
        """

        # write data
        # add extension
        if filename[-4:] != '.cbf':
            filename += '.cbf'
        # check lp compatibility
        if (self.numberQuadConstraints + self.numberLSEConstraints) > 0:
            if self.options['convert_quad_to_socp_if_needed']:
                pcop = self.copy()
                pcop.convert_quad_to_socp()
                pcop._write_cbf(filename)
                return
            else:
                raise pic.QuadAsSocpError(
                    'Problem should not have quad or gp constraints. ' +
                    'Try to convert the problem to an SOCP with the function convert_quad_to_socp()')

        # parse variables
        NUMVAR_SCALAR = int(_bsum([(var.endIndex - var.startIndex)
                                   for var in self.variables.values()
                                   if not(var.semiDef)]))

        indices = [(v.startIndex, v.endIndex, v)
                   for v in self.variables.values()]
        indices = sorted(indices)
        idxsdpvars = [(si, ei) for (si, ei, v) in indices[::-1] if v.semiDef]
        # search if some semidef vars are implied in other semidef constraints
        PSD_not_handled = []
        for c in self.constraints:
            if c.typeOfConstraint.startswith('sdp') and not(c.semidefVar):
                for v in (c.Exp1 - c.Exp2).factors:
                    if v.semiDef:
                        idx = (v.startIndex, v.endIndex)
                        if idx in idxsdpvars:
                            PSD_not_handled.append(v)
                            NUMVAR_SCALAR += (idx[1] - idx[0])
                            idxsdpvars.remove(idx)

        barvars = bool(idxsdpvars)

        # find integer variables, put 0-1 bounds on binaries
        ints = []
        for k, var in six.iteritems(self.variables):
            if var.vtype == 'binary':
                for ind, i in enumerate(range(var.startIndex, var.endIndex)):
                    ints.append(i)
                    (clb, cub) = var.bnd.get(ind, (-INFINITY, INFINITY))
                    lb = max(0., clb)
                    ub = min(1., cub)
                    var.bnd._set(ind, (lb, ub))

            elif self.variables[k].vtype == 'integer':
                for i in range(
                        self.variables[k].startIndex,
                        self.variables[k].endIndex):
                    ints.append(i)

            elif self.variables[k].vtype not in ['continuous', 'symmetric']:
                raise Exception('vtype not handled by _write_cbf()')
        if barvars:
            ints, _, mats = self._separate_linear_cons(
                ints, [0.] * len(ints), idxsdpvars)
            if any([bool(mat) for mat in mats]):
                raise Exception(
                    'semidef vars with integer elements are not supported')

        # open file
        f = open(filename, 'w')
        f.write('#file ' + filename + ' generated by picos\n')
        print('writing problem in ' + filename + '...')

        f.write("VER\n")
        f.write("1\n\n")

        f.write("OBJSENSE\n")
        if self.objective[0] == 'max':
            f.write("MAX\n\n")
        else:
            f.write("MIN\n\n")

        # VARIABLEs

        if barvars:
            f.write("PSDVAR\n")
            f.write(str(len(idxsdpvars)) + "\n")
            for si, ei in idxsdpvars:
                ni = int(((8 * (ei - si) + 1)**0.5 - 1) / 2.)
                f.write(str(ni) + "\n")
            f.write("\n")

        # bounds
        cones = []
        conecons = []
        Acoord = []
        Bcoord = []
        iaff = 0
        offset = 0
        for si, ei, v in indices:
            if v.semiDef and not (v in PSD_not_handled):
                offset += (ei - si)
            else:
                if 'nonnegative' in (v._bndtext):
                    cones.append(('L+', ei - si))
                elif 'nonpositive' in (v._bndtext):
                    cones.append(('L-', ei - si))
                else:
                    cones.append(('F', ei - si))
                if 'nonnegative' not in (v._bndtext):
                    for j, (l, u) in six.iteritems(v.bnd):
                        if l is not None:
                            Acoord.append((iaff, si + j - offset, 1.))
                            Bcoord.append((iaff, -l))
                            iaff += 1
                if 'nonpositive' not in (v._bndtext):
                    for j, (l, u) in six.iteritems(v.bnd):
                        if u is not None:
                            Acoord.append((iaff, si + j - offset, -1.))
                            Bcoord.append((iaff, u))
                            iaff += 1
        if iaff:
            conecons.append(('L+', iaff))

        f.write("VAR\n")
        f.write(str(NUMVAR_SCALAR) + ' ' + str(len(cones)) + '\n')
        for tp, n in cones:
            f.write(tp + ' ' + str(n) + '\n')

        f.write('\n')

        # integers
        if ints:
            f.write("INT\n")
            f.write(str(len(ints)) + "\n")
            for i in ints:
                f.write(str(i) + "\n")
            f.write("\n")

        # constraints
        psdcons = []
        isdp = 0
        Fcoord = []
        Hcoord = []
        Dcoord = []
        ObjAcoord = []
        ObjBcoord = []
        ObjFcoord = []
        # dummy constraint for the objective
        if self.objective[1] is None:
            dummy_cons = (AffinExp() > 0)
        else:
            dummy_cons = (self.objective[1] > 0)
        dummy_cons.typeOfConstraint = 'lin0'

        for cons in ([dummy_cons] + self.constraints):
            if cons.typeOfConstraint.startswith('sdp'):
                v = cons.semidefVar
                if not(v is None) and not(v in PSD_not_handled):
                    continue
            if (cons.typeOfConstraint.startswith('lin')
                    or cons.typeOfConstraint[2:] == 'cone'
                    or cons.typeOfConstraint.startswith('sdp')):
                # get sparse indices
                if cons.typeOfConstraint.startswith('lin'):
                    expcone = cons.Exp1 - cons.Exp2
                    if cons.typeOfConstraint[3] == '=':
                        conetype = 'L='
                    elif cons.typeOfConstraint[3] == '<':
                        conetype = 'L-'
                    elif cons.typeOfConstraint[3] == '>':
                        conetype = 'L+'
                    elif cons.typeOfConstraint[3] == '0':
                        conetype = '0'  # dummy type for the objective function
                    else:
                        raise Exception('unexpected typeOfConstraint')
                elif cons.typeOfConstraint == 'SOcone':
                    expcone = ((cons.Exp2) // (cons.Exp1[:]))
                    conetype = 'Q'
                elif cons.typeOfConstraint == 'RScone':
                    expcone = ((cons.Exp2) // (0.5 * cons.Exp3) //
                               (cons.Exp1[:]))
                    conetype = 'QR'
                elif cons.typeOfConstraint.startswith('sdp'):
                    if cons.typeOfConstraint[3] == '<':
                        conetype = None
                        expcone = cons.Exp2 - cons.Exp1
                    elif cons.typeOfConstraint[3] == '>':
                        expcone = cons.Exp1 - cons.Exp2
                        conetype = None
                    else:
                        raise Exception('unexpected typeOfConstraint')

                else:
                    raise Exception('unexpected typeOfConstraint')
                ijv = []
                for var, fact in six.iteritems((expcone).factors):
                    if not isinstance(fact, cvx.base.spmatrix):
                        fact = cvx.sparse(fact)
                    sj = var.startIndex
                    ijv.extend(zip(fact.I, fact.J + sj, fact.V))
                ijvs = sorted(ijv)

                itojv = {}
                lasti = -1
                for (i, j, v) in ijvs:
                    if i == lasti:
                        itojv[i].append((j, v))
                    else:
                        lasti = i
                        itojv[i] = [(j, v)]

                if conetype:
                    if conetype != '0':
                        dim = expcone.size[0] * expcone.size[1]
                        conecons.append((conetype, dim))
                else:
                    dim = expcone.size[0]
                    psdcons.append(dim)

                if conetype:
                    for i, jv in six.iteritems(itojv):
                        J = [jvk[0] for jvk in jv]
                        V = [jvk[1] for jvk in jv]
                        J, V, mats = self._separate_linear_cons(
                            J, V, idxsdpvars)
                        for j, v in zip(J, V):
                            if conetype != '0':
                                Acoord.append((iaff + i, j, v))
                            else:
                                ObjAcoord.append((j, v))
                        for k, mat in enumerate(mats):
                            for row, col, v in zip(mat.I, mat.J, mat.V):
                                if conetype != '0':
                                    Fcoord.append((iaff + i, k, row, col, v))
                                else:
                                    ObjFcoord.append((k, row, col, v))
                                if uptri and row != col:
                                    if conetype != '0':
                                        Fcoord.append(
                                            (iaff + i, k, col, row, v))
                                    else:
                                        ObjFcoord.append((k, col, row, v))
                    constant = expcone.constant
                    if not(constant is None):
                        constant = cvx.sparse(constant)
                        for i, v in zip(constant.I, constant.V):
                            if conetype != '0':
                                Bcoord.append((iaff + i, v))
                            else:
                                ObjBcoord.append(v)
                else:
                    for i, jv in six.iteritems(itojv):
                        col, row = divmod(i, dim)
                        if not(uptri) and row < col:
                            continue
                        J = [jvk[0] for jvk in jv]
                        V = [jvk[1] for jvk in jv]
                        J, V, mats = self._separate_linear_cons(
                            J, V, idxsdpvars)
                        if any([bool(m) for m in mats]):
                            raise Exception(
                                'SDP cons should not depend on PSD var')
                        for j, v in zip(J, V):
                            Hcoord.append((isdp, j, row, col, v))

                    constant = expcone.constant
                    if not(constant is None):
                        constant = cvx.sparse(constant)
                        for i, v in zip(constant.I, constant.V):
                            col, row = divmod(i, dim)
                            if row < col:
                                continue
                            Dcoord.append((isdp, row, col, v))

                if conetype:
                    if conetype != '0':
                        iaff += dim
                else:
                    isdp += 1
            else:
                raise Exception('unexpected typeOfConstraint')

        if iaff > 0:
            f.write("CON\n")
            f.write(str(iaff) + ' ' + str(len(conecons)) + '\n')
            for tp, n in conecons:
                f.write(tp + ' ' + str(n))
                f.write('\n')

            f.write('\n')

        if isdp > 0:
            f.write("PSDCON\n")
            f.write(str(isdp) + '\n')
            for n in psdcons:
                f.write(str(n) + '\n')
            f.write('\n')

        if ObjFcoord:
            f.write("OBJFCOORD\n")
            f.write(str(len(ObjFcoord)) + '\n')
            for (k, row, col, v) in ObjFcoord:
                f.write('{0} {1} {2} {3}\n'.format(k, row, col, v))
            f.write('\n')

        if ObjAcoord:
            f.write("OBJACOORD\n")
            f.write(str(len(ObjAcoord)) + '\n')
            for (j, v) in ObjAcoord:
                f.write('{0} {1}\n'.format(j, v))
            f.write('\n')

        if ObjBcoord:
            f.write("OBJBCOORD\n")
            v = ObjBcoord[0]
            f.write('{0}\n'.format(v))
            f.write('\n')

        if Fcoord:
            f.write("FCOORD\n")
            f.write(str(len(Fcoord)) + '\n')
            for (i, k, row, col, v) in Fcoord:
                f.write('{0} {1} {2} {3} {4}\n'.format(i, k, row, col, v))
            f.write('\n')

        if Acoord:
            f.write("ACOORD\n")
            f.write(str(len(Acoord)) + '\n')
            for (i, j, v) in Acoord:
                f.write('{0} {1} {2}\n'.format(i, j, v))
            f.write('\n')

        if Bcoord:
            f.write("BCOORD\n")
            f.write(str(len(Bcoord)) + '\n')
            for (i, v) in Bcoord:
                f.write('{0} {1}\n'.format(i, v))
            f.write('\n')

        if Hcoord:
            f.write("HCOORD\n")
            f.write(str(len(Hcoord)) + '\n')
            for (i, j, row, col, v) in Hcoord:
                f.write('{0} {1} {2} {3} {4}\n'.format(i, j, row, col, v))
            f.write('\n')

        if Dcoord:
            f.write("DCOORD\n")
            f.write(str(len(Dcoord)) + '\n')
            for (i, row, col, v) in Dcoord:
                f.write('{0} {1} {2} {3}\n'.format(i, row, col, v))
            f.write('\n')

        print('done.')
        f.close()

    def _read_cbf(self, filename):
        try:
            f = open(filename, 'r')
        except IOError:
            filename += '.cbf'
            f = open(filename, 'r')
        print('importing problem data from ' + filename + '...')
        self.__init__()

        line = f.readline()
        while not line.startswith('VER'):
            line = f.readline()

        ver = int(f.readline())
        if ver != 1:
            print('WARNING, file has version > 1')

        structure_keywords = [
            'OBJSENSE',
            'PSDVAR',
            'VAR',
            'INT',
            'PSDCON',
            'CON']
        data_keywords = ['OBJFCOORD', 'OBJACOORD', 'OBJBCOORD',
                         'FCOORD', 'ACOORD', 'BCOORD',
                         'HCOORD', 'DCOORD']

        structure_mode = True  # still parsing structure blocks
        seen_blocks = []
        parsed_blocks = {}
        while True:
            line = f.readline()
            if not line:
                break
            lsplit = line.split()
            if lsplit and lsplit[0] in structure_keywords:
                if lsplit[0] == 'INT' and ('VAR' not in seen_blocks):
                    raise Exception('INT BLOCK before VAR BLOCK')
                if lsplit[0] == 'CON' and not(
                        'VAR' in seen_blocks or 'PSDVAR' in seen_blocks):
                    raise Exception('CON BLOCK before VAR/PSDVAR BLOCK')
                if lsplit[0] == 'PSDCON' and not(
                        'VAR' in seen_blocks or 'PSDVAR' in seen_blocks):
                    raise Exception('PSDCON BLOCK before VAR/PSDVAR BLOCK')
                if lsplit[0] == 'VAR' and (
                        'CON' in seen_blocks or 'PSDCON' in seen_blocks):
                    raise Exception('VAR BLOCK after CON/PSDCON BLOCK')
                if lsplit[0] == 'PSDVAR' and (
                        'CON' in seen_blocks or 'PSDCON' in seen_blocks):
                    raise Exception('PSDVAR BLOCK after CON/PSDCON BLOCK')
                if structure_mode:
                    parsed_blocks[
                        lsplit[0]] = self._read_cbf_block(
                        lsplit[0], f, parsed_blocks)
                    seen_blocks.append(lsplit[0])
                else:
                    raise Exception('Structure keyword after first data item')
            if lsplit and lsplit[0] in data_keywords:
                if 'OBJSENSE' not in seen_blocks:
                    raise Exception('missing OBJSENSE block')
                if not('VAR' in seen_blocks or 'PSDVAR' in seen_blocks):
                    raise Exception('missing VAR/PSDVAR block')
                if lsplit[0] in (
                        'OBJFCOORD', 'FCOORD') and not(
                        'PSDVAR' in seen_blocks):
                    raise Exception('missing PSDVAR block')
                if lsplit[0] in (
                        'OBJACOORD', 'ACOORD', 'HCOORD') and not(
                        'VAR' in seen_blocks):
                    raise Exception('missing VAR block')
                if lsplit[0] in (
                        'DCOORD', 'HCOORD') and not(
                        'PSDCON' in seen_blocks):
                    raise Exception('missing PSDCON block')
                structure_mode = False
                parsed_blocks[
                    lsplit[0]] = self._read_cbf_block(
                    lsplit[0], f, parsed_blocks)
                seen_blocks.append(lsplit[0])

        f.close()
        # variables
        if 'VAR' in parsed_blocks:
            Nvars, varsz, x = parsed_blocks['VAR']
        else:
            x = None

        if 'INT' in parsed_blocks:
            x = parsed_blocks['INT']

        if 'PSDVAR' in parsed_blocks:
            psdsz, X = parsed_blocks['PSDVAR']
        else:
            X = None

        # objective
        obj_constant = parsed_blocks.get('OBJBCOORD', 0)
        bobj = new_param('bobj', obj_constant)
        obj = new_param('bobj', obj_constant)

        aobj = {}
        if 'OBJACOORD' in parsed_blocks:
            obj_vecs = _break_cols(parsed_blocks['OBJACOORD'], varsz)
            aobj = {}
            for k, v in enumerate(obj_vecs):
                if v:
                    aobj[k] = new_param('c[' + str(k) + ']', v)
                    obj += aobj[k] * x[k]

        Fobj = {}
        if 'OBJFCOORD' in parsed_blocks:
            Fbl = parsed_blocks['OBJFCOORD']
            for i, Fi in enumerate(Fbl):
                if Fi:
                    Fobj[i] = new_param('F[' + str(i) + ']', Fi)
                    obj += (Fobj[i] | X[i])

        self.set_objective(self.objective[0], obj)

        # cone constraints
        bb = {}
        AA = {}
        FF = {}
        if 'CON' in parsed_blocks:
            Ncons, structcons = parsed_blocks['CON']
            szcons = [s for tp, s in structcons]

            b = parsed_blocks.get(
                'BCOORD', spmatrix(
                    [], [], [], (Ncons, 1)))
            bvecs = _break_rows(b, szcons)
            consexp = []
            for i, bi in enumerate(bvecs):
                bb[i] = new_param('b[' + str(i) + ']', bi)
                consexp.append(new_param('b[' + str(i) + ']', bi))

            A = parsed_blocks.get(
                'ACOORD', spmatrix(
                    [], [], [], (Ncons, Nvars)))
            Ablc = _break_rows(A, szcons)
            for i, Ai in enumerate(Ablc):
                Aiblocs = _break_cols(Ai, varsz)
                for j, Aij in enumerate(Aiblocs):
                    if Aij:
                        AA[i, j] = new_param('A[' + str((i, j)) + ']', Aij)
                        consexp[i] += AA[i, j] * x[j]

            Fcoords = parsed_blocks.get('FCOORD', {})
            for k, mats in six.iteritems(Fcoords):
                i, row = _block_idx(k, szcons)
                row_exp = AffinExp()
                for j, mat in enumerate(mats):
                    if mat:
                        FF[i, j, row] = new_param(
                            'F[' + str((i, j, row)) + ']', mat)
                        row_exp += (FF[i, j, row] | X[j])

                consexp[i] += (('e_' + str(row) +
                                '(' + str(szcons[i]) + ',1)') * row_exp)

            for i, (tp, sz) in enumerate(structcons):
                if tp == 'F':
                    continue
                elif tp == 'L-':
                    self.add_constraint(consexp[i] < 0)
                elif tp == 'L+':
                    self.add_constraint(consexp[i] > 0)
                elif tp == 'L=':
                    self.add_constraint(consexp[i] == 0)
                elif tp == 'Q':
                    self.add_constraint(abs(consexp[i][1:]) < consexp[i][0])
                elif tp == 'QR':
                    self.add_constraint(
                        abs(consexp[i][2:])**2 < 2 * consexp[i][0] * consexp[i][1])
                else:
                    raise Exception('unexpected cone type')

        DD = {}
        HH = {}
        if 'PSDCON' in parsed_blocks:
            Dblocks = parsed_blocks.get(
                'DCOORD', [
                    spmatrix(
                        [], [], [], (ni, ni)) for ni in parsed_blocks['PSDCON']])
            Hblocks = parsed_blocks.get('HCOORD', {})

            consexp = []
            for i, Di in enumerate(Dblocks):
                DD[i] = new_param('D[' + str(i) + ']', Di)
                consexp.append(new_param('D[' + str(i) + ']', Di))

            for j, Hj in six.iteritems(Hblocks):
                i, col = _block_idx(j, varsz)
                for k, Hij in enumerate(Hj):
                    if Hij:
                        HH[k, i, col] = new_param(
                            'H[' + str((k, i, col)) + ']', Hij)
                        consexp[k] += HH[k, i, col] * x[i][col]

            for exp in consexp:
                self.add_constraint(exp >> 0)

        print('done.')

        params = {'aobj': aobj,
                  'bobj': bobj,
                  'Fobj': Fobj,
                  'A': AA,
                  'b': bb,
                  'F': FF,
                  'D': DD,
                  'H': HH,
                  }

        return x, X, params  # TODO interface + check returned params !

    def _read_cbf_block(self, blocname, f, parsed_blocks):
        if blocname == 'OBJSENSE':
            objsense = f.readline().split()[0].lower()
            self.objective = (objsense, None)
            return None
        elif blocname == 'PSDVAR':
            n = int(f.readline())
            vardims = []
            XX = []
            for i in range(n):
                ni = int(f.readline())
                vardims.append(ni)
                Xi = self.add_variable(
                    'X[' + str(i) + ']', (ni, ni), 'symmetric')
                XX.append(Xi)
                self.add_constraint(Xi >> 0)
            return vardims, XX
        elif blocname == 'VAR':
            Nscalar, ncones = [int(fi) for fi in f.readline().split()]
            tot_dim = 0
            var_structure = []
            xx = []
            for i in range(ncones):
                lsplit = f.readline().split()
                tp, dim = lsplit[0], int(lsplit[1])
                tot_dim += dim
                var_structure.append(dim)
                if tp == 'F':
                    xi = self.add_variable('x[' + str(i) + ']', dim)
                elif tp == 'L+':
                    xi = self.add_variable('x[' + str(i) + ']', dim, lower=0)
                elif tp == 'L-':
                    xi = self.add_variable('x[' + str(i) + ']', dim, upper=0)
                elif tp == 'L=':
                    xi = self.add_variable(
                        'x[' + str(i) + ']', dim, lower=0, upper=0)
                elif tp == 'Q':
                    xi = self.add_variable('x[' + str(i) + ']', dim)
                    self.add_constraint(abs(xi[1:]) < xi[0])
                elif tp == 'QR':
                    xi = self.add_variable('x[' + str(i) + ']', dim)
                    self.add_constraint(abs(xi[2:])**2 < 2 * xi[0] * xi[1])
                xx.append(xi)
            if tot_dim != Nscalar:
                raise Exception('VAR dimensions do not match the header')
            return Nscalar, var_structure, xx
        elif blocname == 'INT':
            n = int(f.readline())
            ints = {}
            for k in range(n):
                j = int(f.readline())
                i, col = _block_idx(j, parsed_blocks['VAR'][1])
                ints.setdefault(i, [])
                ints[i].append(col)
            x = parsed_blocks['VAR'][2]
            for i in ints:
                if len(ints[i]) == x[i].size[0]:
                    x[i].vtype = 'integer'
                else:
                    x.append(self.add_variable(
                        'x_int[' + str(i) + ']', len(ints[i]), 'integer'))
                    for k, j in enumerate(ints[i]):
                        self.add_constraint(x[i][j] == x[-1][k])
            return x
        elif blocname == 'CON':
            Ncons, ncones = [int(fi) for fi in f.readline().split()]
            cons_structure = []
            tot_dim = 0
            for i in range(ncones):
                lsplit = f.readline().split()
                tp, dim = lsplit[0], int(lsplit[1])
                tot_dim += dim
                cons_structure.append((tp, dim))
            if tot_dim != Ncons:
                raise Exception('CON dimensions do not match the header')
            return Ncons, cons_structure
        elif blocname == 'PSDCON':
            n = int(f.readline())
            psdcons_structure = []
            for i in range(n):
                ni = int(f.readline())
                psdcons_structure.append(ni)
            return psdcons_structure
        elif blocname == 'OBJACOORD':
            n = int(f.readline())
            J = []
            V = []
            for i in range(n):
                lsplit = f.readline().split()
                j, v = int(lsplit[0]), float(lsplit[1])
                J.append(j)
                V.append(v)
            return spmatrix(
                V, [0] * len(J), J, (1, parsed_blocks['VAR'][0]))
        elif blocname == 'OBJBCOORD':
            return float(f.readline())
        elif blocname == 'OBJFCOORD':
            n = int(f.readline())
            Fobj = [spmatrix([], [], [], (ni, ni))
                    for ni in parsed_blocks['PSDVAR'][0]]
            for k in range(n):
                lsplit = f.readline().split()
                j, row, col, v = (int(lsplit[0]), int(lsplit[1]),
                                  int(lsplit[2]), float(lsplit[3]))
                Fobj[j][row, col] = v
                if row != col:
                    Fobj[j][col, row] = v
            return Fobj
        elif blocname == 'FCOORD':
            n = int(f.readline())
            Fblocks = {}
            for k in range(n):
                lsplit = f.readline().split()
                i, j, row, col, v = (int(lsplit[0]), int(lsplit[1]), int(
                    lsplit[2]), int(lsplit[3]), float(lsplit[4]))
                if i not in Fblocks:
                    Fblocks[i] = [
                        spmatrix(
                            [], [], [], (ni, ni)) for ni in parsed_blocks['PSDVAR'][0]]

                Fblocks[i][j][row, col] = v
                if row != col:
                    Fblocks[i][j][col, row] = v
            return Fblocks
        elif blocname == 'ACOORD':
            n = int(f.readline())
            J = []
            V = []
            I = []
            for k in range(n):
                lsplit = f.readline().split()
                i, j, v = int(lsplit[0]), int(lsplit[1]), float(lsplit[2])
                I.append(i)
                J.append(j)
                V.append(v)
            return spmatrix(
                V, I, J, (parsed_blocks['CON'][0], parsed_blocks['VAR'][0]))
        elif blocname == 'BCOORD':
            n = int(f.readline())
            V = []
            I = []
            for k in range(n):
                lsplit = f.readline().split()
                i, v = int(lsplit[0]), float(lsplit[1])
                I.append(i)
                V.append(v)
            return spmatrix(
                V, I, [0] * len(I), (parsed_blocks['CON'][0], 1))
        elif blocname == 'HCOORD':
            n = int(f.readline())
            Hblocks = {}
            for k in range(n):
                lsplit = f.readline().split()
                i, j, row, col, v = (int(lsplit[0]), int(lsplit[1]), int(
                    lsplit[2]), int(lsplit[3]), float(lsplit[4]))
                if j not in Hblocks:
                    Hblocks[j] = [
                        spmatrix(
                            [], [], [], (ni, ni)) for ni in parsed_blocks['PSDCON']]

                Hblocks[j][i][row, col] = v
                if row != col:
                    Hblocks[j][i][col, row] = v
            return Hblocks
        elif blocname == 'DCOORD':
            n = int(f.readline())
            Dblocks = [spmatrix([], [], [], (ni, ni))
                       for ni in parsed_blocks['PSDCON']]
            for k in range(n):
                lsplit = f.readline().split()
                i, row, col, v = (int(lsplit[0]), int(
                    lsplit[1]), int(lsplit[2]), float(lsplit[3]))
                Dblocks[i][row, col] = v
                if row != col:
                    Dblocks[i][col, row] = v
            return Dblocks
        else:
            raise Exception('unexpected block name')

    def convert_quad_to_socp(self):
        """
        replace quadratic constraints by equivalent second order cone constraints
        """
        if self.options['verbose'] > 0:
            print('reformulating quads as socp...')
        for i, c in enumerate(self.constraints):
            if c.typeOfConstraint == 'quad':
                qd = c.Exp1.quad
                sqnorm = _quad2norm(qd)
                self.constraints[i] = sqnorm < -c.Exp1.aff
                self.numberQuadConstraints -= 1
                self.numberConeConstraints += 1
                szcone = sqnorm.LR[0].size
                self.numberConeVars += (szcone[0] * szcone[1]) + 2
        if isinstance(self.objective[1], QuadExp):
            if '_obj_' not in self.variables:
                obj = self.add_variable('_obj_', 1)
            else:
                obj = self.get_variable('_obj_')
            if self.objective[0] == 'min':
                qd = self.objective[1].quad
                aff = self.objective[1].aff
                sqnorm = _quad2norm(qd)
                self.add_constraint(sqnorm < obj - aff)
                self.set_objective('min', obj)
            else:
                qd = (-self.objective[1]).quad
                aff = self.objective[1].aff
                sqnorm = _quad2norm(qd)
                self.add_constraint(sqnorm < aff - obj)
                self.set_objective('max', obj)
            # self.numberQuadConstraints-=1 # no ! because
            # numberQuadConstraints is already uptodate affter set_objective()
        if self.numberQuadConstraints > 0:
            raise Exception('there should not be any quadratics left')
        self.numberQuadNNZ = 0
        # reset solver instances
        self.reset_solver_instances()
        if self.options['verbose'] > 0:
            print('done.')


    def convert_quadobj_to_constraint(self):
        """
        replace quadratic objective by equivalent quadratic constraint
        """
        if isinstance(self.objective[1], QuadExp):
            if '_obj_' not in self.variables:
                obj = self.add_variable('_obj_', 1)
            else:
                obj = self.get_variable('_obj_')
            if self.objective[0] == 'min':
                self.add_constraint(obj > self.objective[1])
                self.set_objective('min', obj)
            else:
                self.add_constraint(obj < self.objective[1])
                self.set_objective('max', obj)

    def to_real(self):
        """
        Returns an equivalent problem,
        where the n x n- hermitian matrices have been replaced by
        symmetric matrices of size 2n x 2n.
        """
        import copy
        real = Problem()
        cvars = {}
        for (iv, v) in sorted([(v.startIndex, v)
                               for v in self.variables.values()]):
            if v.vtype == 'hermitian':
                # cvars[v.name]=real.add_variable(v.name+'_full',(2*v.size[0],2*v.size[1]),'symmetric')
                cvars[
                    v.name +
                    '_RE'] = real.add_variable(
                    v.name +
                    '_RE',
                    (v.size[0],
                     v.size[1]),
                    'symmetric')
                """
                                cvars[v.name+'_IM']=real.add_variable(v.name+'_IM',(v.size[0],v.size[1]))
                                #real.add_constraint((cvars[v.name+'RE'] & -cvars[v.name+'IM'])//(cvars[v.name+'IM'] & cvars[v.name+'RE']) == cvars[v.name])
                                if force_sym:
                                        real.add_constraint(cvars[v.name+'_IM'] == -cvars[v.name+'_IM'].T)
                                """
                cvars[
                    v.name +
                    '_IM_utri'] = list(
                    real.add_variable(
                        v.name +
                        '_IM',
                        (v.size[0],
                         v.size[1]),
                        'antisym').factors.keys())[0]
            else:
                cvars[v.name] = real.add_variable(v.name, v.size, v.vtype)

        for c in self.constraints:
            if c.typeOfConstraint.startswith('sdp'):
                D = {}
                exp1 = c.Exp1
                for var, value in six.iteritems(exp1.factors):
                    try:
                        if var.vtype == 'hermitian':
                            n = int(value.size[1]**(0.5))
                            idasym = _svecm1_identity('antisym', (n, n))

                            D[cvars[
                                var.name + '_RE']] = _cplx_vecmat_to_real_vecmat(value, sym=True, times_i=False)
                            D[cvars[var.name + '_IM_utri']] = _cplx_vecmat_to_real_vecmat(
                                value, sym=False, times_i=True) * idasym
                        else:
                            D[cvars[var.name]] = _cplx_vecmat_to_real_vecmat(
                                value, sym=False)
                    except Exception as ex:
                        import pdb
                        pdb.set_trace()
                        _cplx_vecmat_to_real_vecmat(value, sym=False)
                if exp1.constant is None:
                    cst = None
                else:
                    cst = _cplx_vecmat_to_real_vecmat(exp1.constant, sym=False)
                E1 = AffinExp(
                    D, cst, (2 * exp1.size[0], 2 * exp1.size[1]), exp1.string)
                #import pdb;pdb.set_trace()

                D = {}
                exp2 = c.Exp2
                for var, value in six.iteritems(exp2.factors):
                    if var.vtype == 'hermitian':
                        D[cvars[
                            var.name + '_RE']] = _cplx_vecmat_to_real_vecmat(value, sym=True, times_i=False)
                        D[cvars[var.name + '_IM_utri']
                          ] = _cplx_vecmat_to_real_vecmat(value, sym=False, times_i=True)
                    else:
                        D[cvars[var.name]] = _cplx_vecmat_to_real_vecmat(
                            value, sym=False)
                if exp2.constant is None:
                    cst = None
                else:
                    cst = _cplx_vecmat_to_real_vecmat(exp2.constant, sym=False)
                E2 = AffinExp(
                    D, cst, (2 * exp2.size[0], 2 * exp2.size[1]), exp2.string)

                if c.typeOfConstraint[3] == '<':
                    real.add_constraint(E1 << E2)
                else:
                    real.add_constraint(E1 >> E2)
            elif c.typeOfConstraint.startswith('lin'):
                # 3 situations can occur:
                #_E1 and E2 are real, in this case we add only a constraint
                # for the real part
                #_there is an imaginary part. In this case, this must be an
                # EQUALITY constraint and we equate both the real and the im.
                # part
                if c.Exp1.is_real() and c.Exp2.is_real():
                    E1 = _copy_exp_to_new_vars(c.Exp1, cvars, complex=False)
                    E2 = _copy_exp_to_new_vars(c.Exp2, cvars, complex=False)
                    c2 = Constraint(c.typeOfConstraint, None, E1, E2, None)
                    real.add_constraint(c2, c.key)
                elif c.typeOfConstraint[3] == '=':
                    E1 = _copy_exp_to_new_vars(c.Exp1, cvars, complex=True)
                    E2 = _copy_exp_to_new_vars(c.Exp2, cvars, complex=True)
                    c2 = Constraint(c.typeOfConstraint, None, E1, E2, None)
                    real.add_constraint(c2, c.key)
                else:
                    raise Exception(
                        'A constraint involves inequality between complex numbers')

            else:
                if c.Exp1.is_real():
                    E1 = _copy_exp_to_new_vars(c.Exp1, cvars, complex=False)
                else:
                    E1 = _copy_exp_to_new_vars(c.Exp1, cvars, complex=True)
                if not(c.Exp2 is None) and not(c.Exp2.is_real()):
                    raise Exception(
                        'complex expression in the RHS of a nonlinear constraint')
                if not(c.Exp3 is None) and not(c.Exp3.is_real()):
                    raise Exception(
                        'complex expression in the RHS of a nonlinear constraint')
                E2 = _copy_exp_to_new_vars(c.Exp2, cvars, complex=False)
                E3 = _copy_exp_to_new_vars(c.Exp3, cvars, complex=False)
                c2 = Constraint(c.typeOfConstraint, None, E1, E2, E3)
                real.add_constraint(c2, c.key)

        if not(self.objective[1] is None) and not(self.objective[1].is_real()):
            raise Exception('objective is not real-valued')
        obj = _copy_exp_to_new_vars(self.objective[1], cvars, complex=False)
        # take real part
        #import pdb;pdb.set_trace()
        # if obj[1]:
        # for k,m in obj[1].factors.iteritems():
        # if m.typecode == 'z':
        #obj[1].factors[k] = m.real()
        # if obj[1].constant:
        #m = obj[1].constant
        # if m.typecode == 'z':
        #obj[1].constant = m.real()

        real.set_objective(self.objective[0], obj)

        real.consNumbering = copy.deepcopy(self.consNumbering)
        real.groupsOfConstraints = copy.deepcopy(self.groupsOfConstraints)
        real._options = _NonWritableDict(self.options)

        return real

    def dualize(self):
        """
        Returns a Problem containing the Lagrangian dual of the current problem ``self``.
        More precisely, the current problem is parsed as a problem
        in a canonical primal form (cf. the :ref:`note on dual variables <noteduals>` of the tutorial),
        and the corresponding dual form is returned.
        """
        if self.numberLSEConstraints > 0:
            raise DualizationError('GP cannot be dualized by PICOS')
        if not self.is_continuous():
            raise DualizationError(
                'Mixed integer problems cannot be dualized by picos')
        if self.numberQuadConstraints > 0:
            raise QuadAsSocpError(
                'try to convert the quads as socp before dualizing')

        if self.is_complex():
            raise Exception(
                'dualization of complex SDPs is not supported (yet). Try to convert ' +
                'the problem to an equivalent real-valued problem with to_real() first')

        dual = Problem()
        self._make_cvxopt_instance(hard_coded_bounds=True)
        cc = new_param('cc', self.cvxoptVars['c'])
        lincons = cc
        obj = 0
        # equalities
        Ae = new_param('Ae', self.cvxoptVars['A'])
        be = new_param('be', -self.cvxoptVars['b'])
        if Ae.size[0] > 0:
            mue = dual.add_variable('mue', Ae.size[0])
            lincons += (Ae.T * mue)
            obj += be.T * mue
        # inequalities
        Al = new_param('Al', self.cvxoptVars['Gl'])
        bl = new_param('bl', -self.cvxoptVars['hl'])
        if Al.size[0] > 0:
            mul = dual.add_variable('mul', Al.size[0])
            dual.add_constraint(mul > 0)
            lincons += (Al.T * mul)
            obj += bl.T * mul
        # soc cons
        i = 0
        As, bs, fs, ds, zs, lbda = [], [], [], [], [], []
        for Gq, hq in zip(self.cvxoptVars['Gq'], self.cvxoptVars['hq']):
            As.append(new_param('As[' + str(i) + ']', -Gq[1:, :]))
            bs.append(new_param('bs[' + str(i) + ']', hq[1:]))
            fs.append(new_param('fs[' + str(i) + ']', -Gq[0, :].T))
            ds.append(new_param('ds[' + str(i) + ']', hq[0]))
            zs.append(dual.add_variable('zs[' + str(i) + ']', As[i].size[0]))
            lbda.append(dual.add_variable('lbda[' + str(i) + ']', 1))
            dual.add_constraint(abs(zs[i]) < lbda[i])
            lincons += (As[i].T * zs[i] - fs[i] * lbda[i])
            obj += (bs[i].T * zs[i] - ds[i] * lbda[i])
            i += 1
        # sdp cons
        j = 0
        X = []
        M0 = []
        factors = {}
        for Gs, hs in zip(self.cvxoptVars['Gs'], self.cvxoptVars['hs']):
            nbar = int(Gs.size[0]**0.5)
            svecs = [svec(cvx.matrix(Gs[:, k], (nbar, nbar)),
                          ignore_sym=True).T for k in range(Gs.size[1])]
            msvec = cvx.sparse(svecs)
            X.append(
                dual.add_variable(
                    'X[' + str(j) + ']', (nbar, nbar), 'symmetric'))
            factors[X[j]] = -msvec
            dual.add_constraint(X[j] >> 0)
            M0.append(new_param('M0[' + str(j) + ']', -
                                cvx.matrix(hs, (nbar, nbar))))
            obj += (M0[j] | X[j])
            j += 1

        if factors:
            maff = AffinExp(
                factors=factors, size=(
                    msvec.size[0], 1), string='M dot X')
        else:
            maff = 0
        dual.add_constraint(lincons == maff)
        dual.set_objective('max', obj)
        dual._options = _NonWritableDict(self.options)
        # deactivate the solve_via_dual option (to avoid further dualization)
        dual.set_option('solve_via_dual', False)

        # because there is a bug with retrieval of dual variables of a linear constraints involving a symmetric matrix (due to triangularization?)
        dual.set_option('handleBarVars', True)
        return dual

    """TODO primalize function (in development)
        def primalize(self):
                if self.numberLSEConstraints>0:
                        raise DualizationError('GP cannot be dualized by PICOS')
                if not self.is_continuous():
                        raise DualizationError('Mixed integer problems cannot be dualized by picos')
                if self.numberQuadConstraints>0:
                        raise QuadAsSocpError('try to convert the quads as socp before dualizing')

                #we first create a copy of the problem with the desired "nice dual form"
                pcop = self.copy()

                socones = [] #list of list of (var index,coef) in a so cone
                rscones = [] #list of list of (var index,coef) in a rotated so cone
                semidefs = [] #list of list of var indices in a sdcone
                semidefset = set([]) #set of var indices in a sdcone
                conevarset = set([]) #set of var indices in a (rotated) so cone
                indlmi = 0
                indzz= 0
                XX=[]
                zz=[]
                #add new variables for LMI
                listsdpcons = [(i,cons) for (i,cons) in enumerate(pcop.constraints) if cons.typeOfConstraint.startswith('sdp')]
                for (i,cons) in reversed(listsdpcons):
                        if cons.semidefVar:
                                var = cons.semidefVar
                                semidefs.append(range(var.startIndex,var.endIndex))
                                semidefset.update(range(var.startIndex,var.endIndex))
                        else:
                                sz = cons.Exp1.size
                                pcop.remove_constraint(i)
                                XX.append(pcop.add_variable('_Xlmi['+str(indlmi)+']',sz,'symmetric'))
                                pcop.add_constraint(XX[indlmi]>>0)
                                if cons.typeOfConstraint[3]=='<':
                                        pcop.add_constraint(lowtri(XX[indlmi]) == lowtri(cons.Exp2-cons.Exp1))
                                else:
                                        pcop.add_constraint(lowtri(XX[indlmi]) == lowtri(cons.Exp1-cons.Exp2))
                                semidefs.append(range(XX[indlmi].startIndex,XX[indlmi].endIndex))
                                semidefset.update(range(XX[indlmi].startIndex,XX[indlmi].endIndex))
                                indlmi+=1
                #add new variables for soc cones
                listconecons = [(idcons,cons) for (idcons,cons) in enumerate(pcop.constraints) if cons.typeOfConstraint.endswith('cone')]
                for (idcons,cons) in reversed(listconecons):
                        conexp = (cons.Exp2 // cons.Exp1[:])
                        if cons.Exp3:
                                conexp = ((cons.Exp3) // conexp)

                        #parse the (i,j,v) triple
                        ijv=[]
                        for var,fact in conexp.factors.iteritems():
                                if type(fact)!=cvx.base.spmatrix:
                                        fact = cvx.sparse(fact)
                                sj = var.startIndex
                                ijv.extend(zip( fact.I, fact.J +sj,fact.V))
                        ijvs=sorted(ijv)

                        itojv={}
                        lasti=-1
                        for (i,j,v) in ijvs:
                                if i==lasti:
                                        itojv[i].append((j,v))
                                else:
                                        lasti=i
                                        itojv[i]=[(j,v)]


                        szcons = conexp.size[0] * conexp.size[1]
                        rhstmp = conexp.constant
                        if rhstmp is None:
                                rhstmp = cvx.matrix(0.,(szcons,1))

                        newconexp = new_param(' ',cvx.matrix([]))
                        thiscone = []
                        oldcone = []
                        newvars = []
                        #find the vars which we can keep
                        for i in range(szcons):
                                jv = itojv.get(i,[])
                                if len(jv) == 1 and not(rhstmp[i]) and (jv[0][0] not in semidefset) and (jv[0][0] not in conevarset):
                                        conevarset.update([jv[0][0]])
                                        oldcone.append(jv[0])
                                else:
                                        newvars.append(i)

                        #add new vars
                        countnewvars = len(newvars)
                        if countnewvars>0:
                                zz.append(pcop.add_variable('_zz['+str(indzz)+']',countnewvars))
                                stz = zz[indzz].startIndex
                                indzz += 1
                                conevarset.update(range(stz,stz+countnewvars))


                        #construct the new variable, add (vars,coefs) in 'thiscone'
                        oldind = 0
                        newind = 0
                        for i in range(szcons):
                                jv = itojv.get(i,[])
                                if i not in newvars:
                                        newconexp //= conexp[i]
                                        thiscone.append(oldcone[oldind])
                                        oldind += 1
                                else:
                                        newconexp //= zz[-1][newind]
                                        thiscone.append((stz+newind,1))
                                        pcop.add_constraint(zz[-1][newind] == conexp[i])
                                        newind += 1

                        if countnewvars>0:
                                pcop.remove_constraint(idcons)
                                if cons.Exp3:
                                        nwcons = abs(newconexp[2:])**2 < newconexp[0] * newconexp[1]
                                        if not(newvars in ([0],[1],[0,1])):
                                                ncstring = '||sub(x;_zz[{0}])||**2 < '.format(indzz-1)
                                        else:
                                                ncstring = '||' + cons.Exp1.string + '||**2 < '
                                        if 0 in newvars:
                                                ncstring += '_zz[{0}][0]'.format(indzz-1)
                                        else:
                                                ncstring += cons.Exp2.string
                                        if cons.Exp3.string !='1':
                                                if 1 in newvars:
                                                        if 0 in newvars:
                                                                ncstring += '* _zz[{0}][1]'.format(indzz-1)
                                                        else:
                                                                ncstring += '* _zz[{0}][0]'.format(indzz-1)
                                                else:
                                                        ncstring += '* '+cons.Exp3.string
                                        nwcons.myconstring = ncstring
                                        pcop.add_constraint(nwcons)
                                else:
                                        nwcons = abs(newconexp[1:]) < newconexp[0]
                                        if not(newvars==[0]):
                                                ncstring = '||sub(x;_zz[{0}])|| < '.format(indzz-1)
                                        else:
                                                ncstring = '||' + cons.Exp1.string + '|| < '
                                        if 0 in newvars:
                                                ncstring += '_zz[{0}][0]'.format(indzz-1)
                                        else:
                                                ncstring += cons.Exp2.string
                                        nwcons.myconstring = ncstring
                                        pcop.add_constraint(nwcons)
                        if cons.Exp3:
                                rscones.append(thiscone)
                        else:
                                socones.append(thiscone)


                #TODO think about bounds
                return pcop #tmp return
                """

#----------------------------------------
#                 Obsolete functions
#----------------------------------------

    def set_varValue(self, name, value):
        self.set_var_value(name, value)

    def defaultOptions(self, **opt):
        self.set_all_options_to_default(opt)

    def set_options(self, **options):
        self.update_options(**options)

    def addConstraint(self, cons):
        self.add_constraint(cons)

    def isContinuous(self):
        return self.is_continuous()

    def makeCplex_Instance(self):
        self._make_cplex_instance()

    def makeCVXOPT_Instance(self):
        self._make_cvxopt_instance()
