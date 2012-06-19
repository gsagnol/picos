# coding: utf-8
import cvxopt as cvx
import numpy as np
import sys

from .tools import *
from .expression import *
from .constraint import *

__all__=[ 'Problem','Variable']

global INFINITY
INFINITY=1e16

class Problem:
        """This class represents an optimization problem.
        The constructor creates an empty problem.
        Some options can be provided under the form
        ``key = value``.
        See the list of available options
        in the doc of :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`
        """
        
        def __init__(self,**options):
                self.objective = ('find',None) #feasibility problem only
                self.constraints = []
                """list of all constraints"""
                self.variables = {}
                """dictionary of variables indexed by variable names"""
                self.countVar=0
                """number of (multidimensional) variables"""
                self.countCons=0
                """numner of (multidimensional) constraints"""
                self.numberOfVars=0
                """total number of (scalar) variables"""
                self.numberAffConstraints=0
                """total number of (scalar) affine constraints"""
                self.numberConeVars=0
                """number of auxilary variables required to handle the SOC constraints"""
                self.numberConeConstraints=0
                """number of SOC constraints"""
                self.numberLSEConstraints=0
                """number of LogSumExp constraints (+1 if the objective is a LogSumExp)"""
                self.numberLSEVars=0
                """number of vars in LogSumExp expressions"""
                self.numberQuadConstraints=0
                """number of quadratic constraints (+1 if the objective is quadratic)"""
                self.numberQuadNNZ=0
                """number of nonzero entries in the matrices defining the quadratic expressions"""
                self.numberSDPConstraints=0
                """number of SDP constraints"""
                self.numberSDPVars=0
                """size of the s-vecotrized matrices involved in SDP constraints"""
                self.cvxoptVars={'c':None,'A':None,'b':None,'Gl':None,
                                'hl':None,'Gq':None,'hq':None,'Gs':None,'hs':None,
                                'F':None,'g':None, 'quadcons': None}
                
                self.gurobi_Instance = None
                self.grbvar = {}
                
                self.cplex_Instance = None
                self.cplex_boundcons = None
                
                self.msk_env=None
                self.msk_task=None

                self.scip_solver = None
                self.scip_vars = None
                self.scip_obj = None
                
                self.groupsOfConstraints = {}
                self.listOfVars = {}
                self.consNumbering=[]
                
                self.options = {}
                if options is None: options={}
                self.set_all_options_to_default()
                self.update_options(**options)

                self.number_solutions=0
                                
                self.longestkey=0 #for a nice display of constraints
                self.varIndices=[]
                
                self.status='unsolved'
                """status returned by the solver. The default when
                   a new problem is created is 'unsolved'.
                """
                

        def __str__(self):
                probstr='---------------------\n'               
                probstr+='optimization problem  ({0}):\n'.format(self.type)
                probstr+='{0} variables, {1} affine constraints'.format(
                                self.numberOfVars,self.numberAffConstraints)
                                
                if self.numberConeVars>0:
                        probstr+=', {0} vars in {1} SO cones'.format(
                                self.numberConeVars,self.numberConeConstraints)
                if self.numberLSEConstraints>0:
                        probstr+=', {0} vars in {1} LOG-SUM-EXP'.format(
                                self.numberLSEVars,self.numberLSEConstraints)
                if self.numberSDPConstraints>0:
                        probstr+=', {0} vars in {1} SD cones'.format(
                                self.numberSDPVars,self.numberSDPConstraints)
                probstr+='\n'

                printedlis=[]
                for vkey in self.variables.keys():
                        if '[' in vkey and ']' in vkey:
                                lisname=vkey[:vkey.index('[')]
                                if not lisname in printedlis:
                                        printedlis.append(lisname)
                                        var=self.listOfVars[lisname]
                                        probstr+='\n'+lisname+' \t: '
                                        probstr+=var['type']+' of '+str(var['numvars'])+' variables, '
                                        if var['size']=='different':
                                                probstr+='different sizes'
                                        else:
                                                probstr+=str(var['size'])
                                        if var['vtype']=='different':
                                                probstr+=', different type'
                                        else:
                                                probstr+=', '+var['vtype']
                        else:                   
                                var = self.variables[vkey]
                                probstr+='\n'+vkey+' \t: '+str(var.size)+', '+var.vtype
                probstr+='\n'
                if self.objective[0]=='max':
                        probstr+='\n\tmaximize '+self.objective[1].string+'\n'
                elif self.objective[0]=='min':
                        probstr+='\n\tminimize '+self.objective[1].string+'\n'
                elif self.objective[0]=='find':
                        probstr+='\n\tfind vars\n'
                probstr+='such that\n'
                if self.countCons==0:
                        probstr+='  []\n'
                k=0
                while k<self.countCons:
                        if k in self.groupsOfConstraints.keys():
                                lcur=len(self.groupsOfConstraints[k][2])                                
                                if lcur>0:
                                        lcur+=2
                                        probstr+='('+self.groupsOfConstraints[k][2]+')'
                                if self.longestkey==0:
                                        ntabs=0
                                else:
                                        ntabs=int(np.ceil((self.longestkey+2)/8.0))
                                missingtabs=int(  np.ceil(((ntabs*8)-lcur)/8.0)  )
                                for i in range(missingtabs):
                                        probstr+='\t'
                                if lcur>0:
                                        probstr+=': '
                                else:
                                        probstr+='  '
                                probstr+=self.groupsOfConstraints[k][1]
                                k=self.groupsOfConstraints[k][0]+1
                        else:
                                probstr+=self.constraints[k].keyconstring(self.longestkey)+'\n'
                                k+=1
                probstr+='---------------------'
                return probstr
        

        """
        ----------------------------------------------------------------
        --                       Utilities                            --
        ----------------------------------------------------------------
        """

        def remove_all_constraints(self):
                """
                Removes all constraints from the problem
                """
                self.numberConeConstraints = 0
                self.numberAffConstraints = 0
                self.numberQuadConstraints = 0
                self.numberSDPConstraints = 0
                self.numberLSEConstraints = 0
                self.consNumbering=[]
                self.groupsOfConstraints ={}
                self.numberConeVars=0
                self.numberSDPVars=0
                self.countCons=0
                self.constraints = []
                self.numberQuadNNZ=0
                self.numberLSEVars = 0
                if self.objective[0] is not 'find':
                        if self.objective[1] is not None:
                                expr=self.objective[1]
                                if isinstance(expr,QuadExp):
                                        self.numberQuadNNZ=expr.nnz()
                                if isinstance(expr,LogSumExp):
                                        self.numberLSEVars=expr.Exp.size[0]*expr.Exp.size[1]

        
        def obj_value(self):
                """
                If the problem was already solved, returns the objective value.
                Otherwise, it raises an ``AttributeError``.
                """
                return self.objective[1].eval()[0]

        def get_varName(self,Id):
                return [k for k in self.variables.keys() if  self.variables[k].Id==Id][0]
        
        def set_objective(self,typ,expr):
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
                if (isinstance(expr,AffinExp) and expr.size<>(1,1)):
                        raise Exception('objective should be scalar')
                if not (isinstance(expr,AffinExp) or isinstance(expr,LogSumExp)
                        or isinstance(expr,QuadExp) or isinstance(expr,GeneralFun)):
                        raise Exception('unsupported objective')
                if isinstance(expr,LogSumExp):
                        self.numberLSEVars+=expr.Exp.size[0]*expr.Exp.size[1]
                        self.numberLSEConstraints+=1
                if isinstance(expr,QuadExp):
                        self.numberQuadConstraints+=1
                        self.numberQuadNNZ+=expr.nnz()
                self.objective=(typ,expr)
        
        def set_var_value(self,name,value,optimalvar=False):
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
                >>> prob.set_var_value('x',[3,4])
                >>> abs(x)**2
                #quadratic expression: ||x||**2 #
                >>> print (abs(x)**2)
                25.0
                """
                ind = None
                if isinstance(name,tuple): # alternative solution
                        ind=name[0]
                        name=name[1]

                if not name in self.variables.keys():
                        raise Exception('unknown variable name')
                valuemat,valueString=_retrieve_matrix(value,self.variables[name].size)
                if valuemat.size<>self.variables[name].size:
                        raise Exception('should be of size {0}'.format(self.variables[name].size))
                if self.variables[name].vtype=='symmetric':
                        valuemat=svec(valuemat)
                if ind is None:
                        self.variables[name].value=valuemat
                        if optimalvar:
                                self.number_solutions=max(self.number_solutions,1)
                else:
                        self.variables[name].value_alt[ind]=valuemat
                        if optimalvar:
                                self.number_solutions=max(self.number_solutions,ind+1)

        def _makeGandh(self,affExpr):
                """if affExpr is an affine expression,
                this method creates a bloc matrix G to be multiplied by the large
                vectorized vector of all variables,
                and returns the vector h corresponding to the constant term.
                """
                n1=affExpr.size[0]*affExpr.size[1]
                #matrix G               
                I=[]
                J=[]
                V=[]
                for var in affExpr.factors:
                        si = var.startIndex
                        facvar=affExpr.factors[var]
                        if type(facvar)!=cvx.base.spmatrix:
                                facvar=cvx.sparse(facvar)
                        I.extend(facvar.I)
                        J.extend([si+j for j in facvar.J])
                        V.extend(facvar.V)
                G=cvx.spmatrix(V,I,J,(n1,self.numberOfVars))
                
                #is it really sparse ?
                #if cvx.nnz(G)/float(G.size[0]*G.size[1])>0.5:
                #       G=cvx.matrix(G,tc='d')
                #vector h
                if affExpr.constant is None:
                        h=cvx.matrix(0,(n1,1),tc='d')
                else:
                        h=affExpr.constant
                if not isinstance(h,cvx.matrix):
                        h=cvx.matrix(h,tc='d')
                if h.typecode<>'d':
                        h=cvx.matrix(h,tc='d')
                return G,h

                
        def set_all_options_to_default(self):
                """set all the options to their default.
                The following options are available:
                
                * General options common to all solvers:
                
                  * ``verbose = 1`` : verbosity level [0(quiet)|1|2(loud)]
                  
                  * ``solver = None`` : currently the available solvers are
                    ``'cvxopt'``, ``'cplex'``, ``'mosek'``, ``'smcp'``, ``'zibopt'``.
                    The default
                    ``None`` means that you let picos select a suitable solver for you.
                  
                  * ``tol = 1e-7`` : Relative gap termination tolerance
                    for interior-point optimizers (feasibility and complementary slackness).
                  
                  * ``maxit = 400`` : maximum number of iterations (for simplex or interior-point optimizers)
                  
                  * ``lp_root_method = None`` : algorithm used to solve continuous LP
                    problems, including the root relaxation of mixed integer problems.
                    The default ``None`` selects automatically an algorithm.
                    If set to ``psimplex`` (resp. ``dsimplex``, ``interior``), the solver
                    will use a primal simplex (resp. dual simplex, interior-point) algorithm.
                    *This option currently works only with cplex and mosek*.
                    
                  * ``lp_node_method = None`` : algorithm used to solve subproblems
                    at nodes of the branching trees of mixed integer programs.
                    The default ``None`` selects automatically an algorithm.
                    If set to ``psimplex`` (resp. ``dsimplex``, ``interior``), the solver
                    will use a primal simplex (resp. dual simplex, interior-point) algorithm.
                    *This option currently works only with cplex and mosek*.
                  
                  * ``timelimit = None`` : time limit for the solver, in seconds. The default
                    ``None`` means no time limit.
                    *This option is currently ignored by cvxopt and smcp*.
                
                  * ``treememory = None``  : size of the buffer for the branch and bound tree,
                    in Megabytes. 
                    *This option currently works only with cplex*.
                    
                  * ``gaplim = 1e-4`` : For mixed integer problems,
                    the solver returns a solution as soon as this value for the gap is reached
                    (relative gap between the primal and the dual bound).
                    
                
                  * ``onlyChangeObjective = False`` : set this option to **True** if you have already
                    solved the problem, and want to recompute the solution with a different
                    objective funtion or different parameter settings. This way, the constraints
                    of the problem will not be parsed by picos next time :func:`solve` is called
                    (this can lead to a huge gain of time).
                    
                  * ``noduals = False`` : if ``True``, do not try to retrieve the dual variables.

                  * ``nbsol = None`` : maximum number of feasible solution nodes visited
                    when solving a mixed integer problem.
                                    
                * Specific options available for cvxopt/smcp:
                
                  * ``feastol = 1e-7`` : feasibility tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                  
                  * ``abstol = 1e-7`` : absolute tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                  
                  * ``reltol = 1e-6`` : relative tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                  
                * Specific oprions available for cplex:
                  
                  * ``acceptable_gap_at_timelimit = None`` : If the the time limit is reached,
                    the optimization process is aborted only if the current gap is less
                    than this value. The default value ``None`` means that we
                    interrupt the computation regardless of the achieved gap.
                   
                .. Warning:: Some other options may be available for certain solvers,
                             but they are not (yet) interfaced with picos.
                
                .. Todo:: Organize above options (dire quels solvers si restrictions)

                """
                #Additional, hidden option (requires a patch of smcp, to use conlp to
                #interface the feasible starting point solver):
                #
                #* 'smcp_feas'=False [if True, use the feasible start solver with SMCP]
                default_options={'tol'            :1e-7,
                                 'feastol'        :1e-7,
                                 'abstol'         :1e-7,
                                 'reltol'         :1e-6,
                                 'maxit'          :50,
                                 'verbose'        :1,
                                 'solver'         :None,
                                 'step_sqp'       :1,
                                 'harmonic_steps' :1,
                                 'onlyChangeObjective':False,
                                 'noduals'        :False,
                                 'smcp_feas'      :False,
                                 'nbsol'          :None,
                                 'timelimit'      :None,
                                 'acceptable_gap_at_timelimit'  :None,
                                 'treememory'     :None,
                                 'gaplim'         :None,
                                 'pool_gap'       :None,
                                 'pool_size'      :None,
                                 'lp_root_method' :None,
                                 'lp_node_method' :None,
                                 }
                                 
                                 
                self.options=default_options

        def set_option(self,key,val):
                """
                Sets the option **key** to the value **val**.
                
                :param key: The key of an option
                            (see the list of keys in the doc of
                            :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`).
                :type key: str.
                :param val: New value for the option.
                """
                if key not in self.options:
                        raise AttributeError('unkown option key :'+str(key))
                self.options[key]=val
                if key=='tol':
                        self.options['feastol']=val
                        self.options['abstol']=val
                        self.options['reltol']=val*10
                if key=='verbose' and isinstance(val,bool):
                        self.options['verbose']=int(val)

        
        def update_options(self, **options):
                """
                update the option dictionary, for each pair of the form
                ``key = value``. For a list of available options and their default values,
                see the doc of :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`.
                """
                
                for k in options.keys():
                        self.set_option(k,options[k])
                #erase 'tol' information if more specific options are provided
                if 'feastol' in options:
                        self.set_option('feastol',options['feastol'])
                if 'abstol' in options:
                        self.set_option('abstol',options['abstol'])
                if 'reltol' in options:
                        self.set_option('reltol',options['reltol'])
                        
                
        def _eliminate_useless_variables(self):
                """
                Removes from the problem the variables that do not
                appear in any constraint or in the objective function.
                """
                for var in self.variables.values():
                        found=False
                        for cons in self.constraints.keys():
                                if isinstance(self.constraints[cons].Exp1,AffinExp):
                                        if var in self.constraints[cons].Exp1.factors:
                                                found=True
                                        if var in self.constraints[cons].Exp2.factors:
                                                found=True
                                        if not self.constraints[cons].Exp3 is None:
                                                if var in self.constraints[cons].Exp3.factors:
                                                        found=True
                                elif isinstance(self.constraints[cons].Exp1,QuadExp):
                                        if var in self.constraints[cons].Exp1.aff.factors:
                                                found=True
                                        for ij in self.constraints[cons].Exp1.quad:
                                                if var in ij:
                                                        found=True
                                elif isinstance(self.constraints[cons].Exp1,LogSumExp):
                                        if var in self.constraints[cons].Exp1.Exp.factors:
                                                found=True
                        if not self.objective[1] is None:
                                if isinstance(self.objective[1],AffinExp):
                                        if var in self.objective[1].factors:
                                                found=True
                                elif isinstance(self.objective[1],QuadExp):
                                        if var in self.objective[1].aff.factors:
                                                found=True
                                        for ij in self.objective[1].quad:
                                                if var in ij:
                                                        found=True
                                elif isinstance(self.objective[1],LogSumExp):
                                        if var in self.objective[1].Exp.factors.keys():
                                                found=True
                        if not found:
                                self.remove_variable(var.name)
                                if self.options['verbose']>0:
                                        print('variable '+var.namer+' was useless and has been removed')

        """
        ----------------------------------------------------------------
        --                TOOLS TO CREATE AN INSTANCE                 --
        ----------------------------------------------------------------
        """

        def add_variable(self,name,size=1, vtype = 'continuous' ):
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
                             
                             Can be either
                             
                                * an ``int`` *n* , in which case the variable is a **vector of dimension n**
                                * or a ``tuple`` *(n,m)*, and the variable is a **n x m-matrix**.
                
                :type size: int or tuple.
                :param vtype: variable :attr:`type <picos.Variable.vtype>`. 
                              Can be:
                                * ``'continuous'`` (default),
                                * ``'binary'``: 0/1 variable
                                * ``'integer'``: integer valued variable
                                * ``'symmetric'``: symmetric matrix
                                * ``'semicont'``: 0 or continuous variable satisfying its bounds
                                * ``'semiint'``: 0 or integer variable satisfying its bounds
                :type vtype: str.
                :returns: An instance of the class :class:`Variable <picos.Variable>`.
                """

                if name in self.variables:
                        raise Exception('this variable already exists')
                if isinstance(size,int):
                        size=(size,1)
                if len(size)==1:
                        size=(size[0],1)

                if '[' in name and ']' in name:#list or dict of variables
                        lisname=name[:name.index('[')]
                        ind=name[name.index('[')+1:name.index(']')]
                        if lisname in self.listOfVars:
                                oldn=self.listOfVars[lisname]['numvars']
                                self.listOfVars[lisname]['numvars']+=1
                                if size<>self.listOfVars[lisname]['size']:
                                        self.listOfVars[lisname]['size']='different'
                                if vtype<>self.listOfVars[lisname]['vtype']:
                                        self.listOfVars[lisname]['vtype']='different'
                                if self.listOfVars[lisname]['type']=='list' and ind<>str(oldn):
                                        self.listOfVars[lisname]['type']='dict'
                        else:
                                self.listOfVars[lisname]={'numvars':1,'size':size,'vtype':vtype}
                                if ind=='0':
                                        self.listOfVars[lisname]['type']='list'
                                else:
                                        self.listOfVars[lisname]['type']='dict'
                
                countvar=self.countVar
                numbervar=self.numberOfVars
                
                if vtype=='symmetric':
                        if size[0]!=size[1]:
                                raise ValueError('symmetric variables must be square')
                        s0=size[0]
                        self.numberOfVars+=s0*(s0+1)/2
                else:
                        self.numberOfVars+=size[0]*size[1]
                self.varIndices.append(self.countVar)
                self.countVar+=1
                
                #svec operation
                idmat=_svecm1_identity(vtype,size)
                
                self.variables[name]=Variable(name,
                                        size,
                                        countvar,
                                        numbervar,
                                        vtype=vtype)
                return self.variables[name]
        
        
        def remove_variable(self,name):
                """
                Removes the variable ``name`` from the problem.
                :param name: name of the variable to remove.
                :type name: str.
                """
                if '[' in name and ']' in name:#list or dict of variables
                        lisname=name[:name.index('[')]
                        if lisname in self.listOfVars:
                                del self.listOfVars[lisname] #not a complete list of vars anymore
                if name not in self.variables.keys():
                        raise Exception('variable does not exist. Maybe you tried to remove some item x[i] of the variable x ?')
                Id=self.variables[name].Id
                self.countVar-=1
                sz=self.variables[name].size
                self.numberOfVars-=sz[0]*sz[1]
                self.varIndices.remove(Id)
                del self.variables[name]
                self._recomputeStartEndIndices()
        
        def _recomputeStartEndIndices(self):
                ind=0
                for i in self.varIndices:
                        nam=self.get_varName(i)
                        self.variables[nam].startIndex=ind
                        ind+=self.variables[nam].size[0]*self.variables[nam].size[1]
                        self.variables[nam].endIndex=ind

        def add_constraint(self,cons, key=None):
                """Adds a constraint in the problem.
                
                :param cons: The constraint to be added.
                :type cons: :class:`Constraint<picos.Constraint>``
                :param key: Optional parameter to describe the constraint with a key string.
                :type key: str.
                """
                cons.key=key
                if not key is None:
                        self.longestkey=max(self.longestkey,len(key))
                self.constraints.append(cons)
                self.consNumbering.append(self.countCons)
                self.countCons+=1
                if cons.typeOfConstraint[:3]=='lin':
                        self.numberAffConstraints+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                elif cons.typeOfConstraint[2:]=='cone':
                        self.numberConeVars+=(cons.Exp1.size[0]*cons.Exp1.size[1])+1
                        self.numberConeConstraints+=1
                        if cons.typeOfConstraint[:2]=='RS':
                                self.numberConeVars+=1
                elif cons.typeOfConstraint=='lse':
                        self.numberLSEVars+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                        self.numberLSEConstraints+=1
                elif cons.typeOfConstraint=='quad':
                        self.numberQuadConstraints+=1
                        self.numberQuadNNZ+=cons.Exp1.nnz()
                elif cons.typeOfConstraint[:3]=='sdp':
                        self.numberSDPConstraints+=1
                        self.numberSDPVars+=(cons.Exp1.size[0]*(cons.Exp1.size[0]+1))/2
                

        def add_list_of_constraints(self,lst,it=None,indices=None,key=None):
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
                firstCons=self.countCons
                for ks in lst:
                        self.add_constraint(ks)
                        self.consNumbering.pop()
                self.consNumbering.append(range(firstCons,self.countCons))
                lastCons=self.countCons-1
                if key is None:
                        key=''
                else:
                        self.longestkey=max(self.longestkey,len(key))
                if it is None:
                        strlis='['+str(len(lst))+' constraints (first: '+lst[0].constring()+')]\n'
                else:
                        strlis=' for all '
                        if len(it)>1:
                                strlis+='('                        
                        for x in it:
                                if isinstance(x,tuple):
                                        strlis+=x[0]
                                else:
                                        strlis+=x
                                strlis+=','
                        strlis=strlis[:-1] #remvove the last comma
                        if len(it)>1:
                                strlis+=')'
                        if not indices is None:
                                strlis+=' in '+indices
                        if isinstance(it,tuple) and len(it)==2 and isinstance(it[1],int):
                                it=(it,)
                        if isinstance(it,list):
                                it=tuple(it)
                        if not isinstance(it,tuple):
                                it=(it,)
                        lstr=[l.constring() for l in lst if '(first:' not in l.constring()]
                        try:
                                indstr=putIndices(lstr,it)
                                strlis=indstr+strlis+'\n'
                        except Exception as ex:
                                strlis='['+str(len(lst))+' constraints (first: '+lst[0].constring()+')]\n'
                self.groupsOfConstraints[firstCons]=[lastCons,strlis,key]
         

        def get_valued_variable(self,name):
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
                exp=self.get_variable(name)
                if isinstance(exp,list):
                        for i in xrange(len(exp)):
                                exp[i]=exp[i].eval()
                elif isinstance(exp,dict):
                        for i in exp:
                                exp[i]=exp[i].eval()
                else:
                        exp=exp.eval()
                return exp
                

        def get_variable(self,name):
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
                var=name
                if var in self.listOfVars.keys():
                        if self.listOfVars[var]['type']=='dict':
                                rvar={}
                        else:
                                rvar=[0]*self.listOfVars[var]['numvars']
                        seenKeys=[]
                        for ind in [vname[len(var)+1:-1] for vname in self.variables.keys() if \
                                 (vname[:len(var)] ==var and vname[len(var)]=='[')]:
                                if ind.isdigit():
                                        key=int(ind)
                                        if key not in seenKeys:
                                                seenKeys.append(key)
                                        else:
                                                key=ind
                                elif ',' in ind:
                                        isplit=ind.split(',')
                                        if isplit[0].startswith('('):
                                                isplit[0]=isplit[0][1:]
                                        if isplit[-1].endswith(')'):
                                                isplit[-1]=isplit[-1][:-1]
                                        if all([i.isdigit() for i in isplit]):
                                                key=tuple([int(i) for i in isplit])
                                                if key not in seenKeys:
                                                        seenKeys.append(key)
                                                else:
                                                        key=ind
                                        else:
                                                key=ind
                                else:
                                        try:
                                                key=float(ind)
                                        except ValueError:
                                                key=ind
                                rvar[key]=self.variables[var+'['+ind+']']
                        return rvar
                else:
                        return self.variables[var]
                        


        def get_constraint(self,ind):
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
                  〈 |1| | x[i] 〉 < y[i] for all i in [5]
                  y > |0|
                ---------------------
                >>> prob.get_constraint(1)                              #2d constraint (numbered from 0)
                # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #
                >>> prob.get_constraint((0,3))                          #4th consraint from the 1st group
                # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #
                >>> prob.get_constraint((1,))                           #unique constraint of the 2d 'group'
                # (5x1)-affine constraint: y > |0| #
                >>> prob.get_constraint((0,))                           #list of constraints of the 1st group #doctest: +NORMALIZE_WHITESPACE
                [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
                 # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #,
                 # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
                 # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
                 # (1x1)-affine constraint: 〈 |1| | x[4] 〉 < y[4] #]
                >>> prob.get_constraint(5)                              #6th constraint
                # (5x1)-affine constraint: y > |0| #
                
                """
                indtuple=ind
                if isinstance(indtuple,int):
                        return self.constraints[indtuple]
                lsind=self.consNumbering            
                if not( isinstance(indtuple,tuple) or isinstance(indtuple,list)) or (
                                len(indtuple)==0):
                        raise Exception('ind must be an int or a nonempty tuple')
                
                for k in indtuple:
                        if not isinstance(lsind,list):
                                if k==0:
                                        break
                                else:
                                        raise Exception('too many indices')
                        if k>=len(lsind):
                                raise Exception('index is too large')
                        lsind=lsind[k]
                if isinstance(lsind,list):
                                return [self.constraints[i] for i in lsind]
                return self.constraints[lsind]
                
        
        def remove_constraint(self,ind):
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
                [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
                 # (1x1)-affine constraint: 〈 |1| | x[1] 〉 < y[1] #,
                 # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
                 # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
                 # (4x1)-affine constraint: y > |0| #,
                 # (2x1)-affine constraint: x[0] < |2.0| #,
                 # (2x1)-affine constraint: x[1] < |2.0| #,
                 # (2x1)-affine constraint: x[2] < |2.0| #,
                 # (2x1)-affine constraint: x[3] < |1| #]
                >>> prob.remove_constraint(1)                           #2d constraint (numbered from 0) deleted
                >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
                [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #, 
                 # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
                 # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
                 # (4x1)-affine constraint: y > |0| #,
                 # (2x1)-affine constraint: x[0] < |2.0| #,
                 # (2x1)-affine constraint: x[1] < |2.0| #,
                 # (2x1)-affine constraint: x[2] < |2.0| #,
                 # (2x1)-affine constraint: x[3] < |1| #]
                >>> prob.remove_constraint((1,))                        #2d 'group' of constraint deleted, i.e. the single constraint y>|0|
                >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
                [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
                 # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
                 # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
                 # (2x1)-affine constraint: x[0] < |2.0| #,
                 # (2x1)-affine constraint: x[1] < |2.0| #,
                 # (2x1)-affine constraint: x[2] < |2.0| #,
                 # (2x1)-affine constraint: x[3] < |1| #]
                >>> prob.remove_constraint((2,))                        #3d 'group' of constraint deleted, (originally the 4th group, i.e. x[3]<|1|)
                >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
                [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
                 # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
                 # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
                 # (2x1)-affine constraint: x[0] < |2.0| #,
                 # (2x1)-affine constraint: x[1] < |2.0| #,
                 # (2x1)-affine constraint: x[2] < |2.0| #]
                >>> prob.remove_constraint((1,1))                       #2d constraint of the 2d group (originally the 3rd group), i.e. x[1]<|2|
                >>> prob.constraints #doctest: +NORMALIZE_WHITESPACE
                [# (1x1)-affine constraint: 〈 |1| | x[0] 〉 < y[0] #,
                 # (1x1)-affine constraint: 〈 |1| | x[2] 〉 < y[2] #,
                 # (1x1)-affine constraint: 〈 |1| | x[3] 〉 < y[3] #,
                 # (2x1)-affine constraint: x[0] < |2.0| #,
                 # (2x1)-affine constraint: x[2] < |2.0| #]

                
                """
                if isinstance(ind,int):
                        del self.constraints[ind]
                        self.countCons -=1
                        if ind in self.consNumbering: #single added constraint
                                self.consNumbering.remove(ind)
                                start=ind
                                self.consNumbering=offset_in_lil(self.consNumbering,1,ind)
                        else: #a constraint within a group of constraints
                                for i,l in enumerate(self.consNumbering):
                                        if isinstance(l,list) and ind in l:
                                                start=l[0]
                                                self.consNumbering[i].remove(ind)
                                self.consNumbering=offset_in_lil(self.consNumbering,1,ind)
                                goc=self.groupsOfConstraints[start]
                                self.groupsOfConstraints[start] = [ goc[0]-1,
                                                                goc[1]+'{-1cons}',
                                                                goc[2]]
                        #offset in subsequent goc
                        for stidx in self.groupsOfConstraints:
                                if stidx>start:
                                        goc=self.groupsOfConstraints[stidx]
                                        del self.groupsOfConstraints[stidx]
                                        goc[0]=goc[0]-1
                                        self.groupsOfConstraints[stidx-1] = goc
                        return

                indtuple=ind
                lsind=self.consNumbering                
                for k in indtuple:
                        if not isinstance(lsind,list):
                                if k==0:
                                        break
                                else:
                                        raise Exception('too many indices')
                        if k>=len(lsind):
                                raise Exception('index is too large')
                        lsind=lsind[k]
                #now, lsind must be the index or list of indices to remove
                if isinstance(lsind,list) and lsind in self.consNumbering: #a list of constraints
                        for ind in lsind:
                                del self.constraints[ind]
                        self.countCons -= len(lsind)
                        self.consNumbering.remove(lsind)
                        start=lsind[0]
                        self.consNumbering=offset_in_lil(self.consNumbering,len(lsind),start)
                        #offset in subsequent goc
                        del self.groupsOfConstraints[start]
                        for stidx in self.groupsOfConstraints:
                                if stidx>start:
                                        goc=self.groupsOfConstraints[stidx]
                                        del self.groupsOfConstraints[stidx]
                                        goc[0]=goc[0]-len(lsind)
                                        self.groupsOfConstraints[stidx-len(lsind)] = goc
                elif isinstance(lsind,int):
                        self.remove_constraint(lsind)
                else:
                        raise Exception('lsind does not coincide with a single constraint or a group of constraints')
                        
                
        def _eval_all(self):
                """
                Returns the big vector with all variable values,
                in the order induced by sorted(self.variables.keys()).
                """
                xx=cvx.matrix([],(0,1))
                for v in sorted(self.variables.keys()):
                        xx=cvx.matrix([xx,self.variables[v].value[:]])
                return xx

        """
        ----------------------------------------------------------------
        --                BUILD THE VARIABLES FOR A SOLVER            --
        ----------------------------------------------------------------
        """        

        '''GUROBi does not work with our version of Python ? 
        def makeGUROBI_Instance(self):
                """
                defines the variables gurobi_Instance and grbvar
                """
                self._set_as_current_problem()                
                self.options['solver'] = 'GUROBI'
                m = Model()
                
                grb_type = {        'continuous' : GRB.CONTINUOUS, 
                                'binary' : GRB.BINARY, 
                                'integer' : GRB.INTEGER, 
                                'semicont' : GRB.SEMICONT, 
                                'semiint' : GRB.SEMIINT }
                
                # variables
                for kvar in self.variables.keys():
                        variable = self.variables[kvar]
                        # a vector
                        objective = self.objective[1].factors
                        if variable.size[1] == 1:
                                # objective vector
                                if kvar in objective.keys():
                                        vectorObjective = objective[kvar][0]
                                else:
                                        vectorObjective = []
                                        for i in range(variable.size[0]):
                                                vectorObjective.append[0]
                                
                                for i in range(variable.size[0]):
                                        #lb (optional): Lower bound for new variable.
                                        #ub (optional): Upper bound for new variable.
                                        #obj (optional): Objective coefficient for new variable.
                                        #vtype (optional): Variable type for new variable (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT).
                                        #name (optional): Name for new variable.
                                        #column (optional): Column object that indicates the set of constraints in which the new variable participates, and the associated coefficients. 
                                        newvar = m.addVar(obj = vectorObjective[i], vtype = grb_type[variable.vtype])
                                        self.grbvar[kvar].append(n)
                        # not a vector
                        else:
                                raise ValueError("the variable is not a vector, not implemented yet")
                m.update()
                
                for constr in self.constraints:
                        if constr.typeOfConstraint == 'lin<':
                                sense = GRB.LESS_EQUAL
                        elif constr.typeOfConstraint == 'lin>':
                                sense = GRB.GREATER_EQUAL
                        elif constr.typeOfConstraint == 'lin=':
                                sense = GRB.EQUAL
                        else:
                                raise ValueError('Impossible linear constraint')
                        
                        # decompose vector in i relations
                        for i in range(constr.Exp1.size[0]):
                                lhsvar = []
                                lhsparam = []
                                rhsvar = []
                                rhsparam = []
                                # left
                                for kvar in constr.Exp1.factors.keys():
                                        lhsvar.append(self.grbvar[kvar])
                                        lhsparam.append(constr.Exp1.factors[kvar][i,:])
                                # right
                                for kvar in constr.Exp2.factors.keys():
                                        rhsvar.append(self.grbvar[kvar])
                                        rhsparam.append(constr.Exp1.factors[kvar][i,:])
                                #adding the constraint
                                lhs = LinExpr(lhsparam, lhsvar)
                                rhs = LinExpr(rhsparam, rhsvar)
                                m.addConstr(lhs, sense, rhs)
        '''
                
        def is_continuous(self):
                """ Returns ``True`` if there are only continuous variables"""
                for kvar in self.variables.keys():
                        if self.variables[kvar].vtype != 'continuous':
                                return False
                return True
                
        def _make_cplex_instance(self):
                """
                Defines the variables cplex_Instance and cplexvar,
                used by the cplex solver.
                """
                try:
                        import cplex
                except:
                        Exception('cplex library not found')
                
                
                #only change the objective coefficients
                if self.options['onlyChangeObjective']:
                        if self.cplex_Instance is None:
                                raise Exception('option is only available when cplex_Instance has been defined before')
                        newobj=self.objective[1]
                        #redefine cplex_Instance
                        coefs=[]
                        for s,v in self.variables.iteritems():
                                if v in newobj.factors:
                                        fac=newobj.factors[v]
                                        if not isinstance(fac,cvx.matrix):
                                                fac=cvx.matrix(fac)
                                        sv=v.startIndex
                                        for jj,vv in enumerate(fac):
                                                coefs.append((jj+sv,vv))
                                else:
                                        for i in xrange(v.startIndex,v.endIndex):
                                                coefs.append((i,0))
                        self.cplex_Instance.objective.set_linear(coefs)
                        return
                
                c = cplex.Cplex()
                import itertools
                
                sense_opt = self.objective[0]
                if sense_opt == 'max':
                        c.objective.set_sense(c.objective.sense.maximize)
                elif sense_opt == 'min':
                        c.objective.set_sense(c.objective.sense.minimize)
                #else:
                #        raise ValueError('feasibility problems not implemented ?')
                
                self.options['solver'] = 'cplex'
                
                cplex_type = {  'continuous' : c.variables.type.continuous, 
                                'binary' : c.variables.type.binary, 
                                'integer' : c.variables.type.integer, 
                                'semicont' : c.variables.type.semi_continuous, 
                                'semiint' : c.variables.type.semi_integer }                
                
                #create new variable and quad constraints to handle socp
                tmplhs=[]
                tmprhs=[]
                icone =0
                newcons={}
                if self.numberConeConstraints > 0 :
                        for constrKey,constr in enumerate(self.constraints):
                                if constr.typeOfConstraint[2:]=='cone':
                                        if icone == 0: #first conic constraint
                                                if '__noconstant__' in self.variables:
                                                        noconstant=self.get_variable('__noconstant__')
                                                else:
                                                        noconstant=self.add_variable(
                                                                '__noconstant__',1)
                                                newcons['noconstant']=(
                                                        noconstant>0)
                                if constr.typeOfConstraint=='SOcone':
                                        if '__tmplhs[{0}]__'.format(icone) in self.variables:
                                                self.remove_variable('__tmplhs[{0}]__'.format(icone))
                                        if '__tmprhs[{0}]__'.format(icone) in self.variables:
                                                self.remove_variable('__tmprhs[{0}]__'.format(icone))
                                        tmplhs.append(self.add_variable(
                                                '__tmplhs[{0}]__'.format(icone),
                                                constr.Exp1.size))
                                        tmprhs.append(self.add_variable(
                                                '__tmprhs[{0}]__'.format(icone),
                                                1))
                                        #v_cons is 0/1/-1 to avoid constants in cone (problem with duals)
                                        v_cons = cvx.matrix( [np.sign(constr.Exp1.constant[i])
                                                                        if constr.Exp1[i].isconstant() else 0
                                                                        for i in range(constr.Exp1.size[0]*constr.Exp1.size[1])],
                                                                        constr.Exp1.size)
                                        #lhs and rhs of the cone constraint
                                        newcons['tmp_lhs_{0}'.format(icone)]=(
                                                        constr.Exp1+v_cons*noconstant == tmplhs[icone])
                                        newcons['tmp_rhs_{0}'.format(icone)]=(
                                                        constr.Exp2-noconstant == tmprhs[icone])
                                        #conic constraints
                                        newcons['tmp_conesign_{0}'.format(icone)]=(
                                                        tmprhs[icone]>0)
                                        newcons['tmp_conequad_{0}'.format(icone)]=(
                                        -tmprhs[icone]**2+(tmplhs[icone]|tmplhs[icone])<0)
                                        icone+=1
                                if constr.typeOfConstraint=='RScone':
                                        if '__tmplhs[{0}]__'.format(icone) in self.variables:
                                                self.remove_variable('__tmplhs[{0}]__'.format(icone))
                                        if '__tmprhs[{0}]__'.format(icone) in self.variables:
                                                self.remove_variable('__tmprhs[{0}]__'.format(icone))
                                        tmplhs.append(self.add_variable(
                                                '__tmplhs[{0}]__'.format(icone),
                                                (constr.Exp1.size[0]*constr.Exp1.size[1])+1
                                                ))
                                        tmprhs.append(self.add_variable(
                                                '__tmprhs[{0}]__'.format(icone),
                                                1))
                                        #v_cons is 0/1/-1 to avoid constants in cone (problem with duals)
                                        expcat = ((2*constr.Exp1[:]) // (constr.Exp2-constr.Exp3))
                                        v_cons = cvx.matrix( [np.sign(expcat.constant[i])
                                                                        if expcat[i].isconstant() else 0
                                                                        for i in range(expcat.size[0]*expcat.size[1])],
                                                                        expcat.size)
                                        
                                        #lhs and rhs of the cone constraint
                                        newcons['tmp_lhs_{0}'.format(icone)]=(
                                        (2*constr.Exp1[:] // (constr.Exp2-constr.Exp3)) + v_cons*noconstant == tmplhs[icone])
                                        newcons['tmp_rhs_{0}'.format(icone)]=(
                                                constr.Exp2+constr.Exp3 - noconstant == tmprhs[icone])
                                        #conic constraints
                                        newcons['tmp_conesign_{0}'.format(icone)]=(
                                                        tmprhs[icone]>0)
                                        newcons['tmp_conequad_{0}'.format(icone)]=(
                                        -tmprhs[icone]**2+(tmplhs[icone]|tmplhs[icone])<0)
                                        icone+=1
                                
                
                
                if self.options['verbose']>1:
                        limitbar=self.numberOfVars
                        prog = ProgressBar(0,limitbar, None, mode='fixed')
                        oldprog = str(prog)
                        if self.options['verbose']>0:
                                print('Creating variables...')
                                print
                
                #variables
                
                colnames=['']*self.numberOfVars
                obj=[0]*self.numberOfVars
                types=['C']*self.numberOfVars
                
                #specify bounds later, in constraints
                ub=[cplex.infinity]*self.numberOfVars
                lb=[-cplex.infinity]*self.numberOfVars
                
                if self.objective[1] is None:
                        objective = {}
                elif isinstance(self.objective[1],QuadExp):
                        objective = self.objective[1].aff.factors
                elif isinstance(self.objective[1],AffinExp):
                        objective = self.objective[1].factors
                
                for kvar,variable in self.variables.iteritems():
                        sj=variable.startIndex
                        if objective.has_key(variable):
                                vectorObjective = objective[variable]
                        else:
                                vectorObjective = [0]*(variable.size[0]*variable.size[1])
                        for k in range(variable.size[0]*variable.size[1]):
                                colnames[sj+k]=kvar+'_'+str(k)
                                obj[sj+k]=vectorObjective[k]
                                types[sj+k]=cplex_type[variable.vtype]
                                
                                if self.options['verbose']>1:
                                        #<--display progress
                                        prog.increment_amount()
                                        if oldprog != str(prog):
                                                print prog, "\r",
                                                sys.stdout.flush()
                                                oldprog=str(prog)
                                        #-->
                
                if self.options['verbose']>1:
                        prog.update_amount(limitbar)
                        print prog, "\r",
                        print
                
                
                #quad part of the objective
                quad_terms = []
                if isinstance(self.objective[1],QuadExp):
                        qd=self.objective[1].quad
                        for i,j in qd:
                                fact=qd[i,j]
                                si=i.startIndex
                                sj=j.startIndex
                                if (j,i) in qd: #quad stores x'*A1*y + y'*A2*x
                                        if si<sj:
                                                fact+=qd[j,i].T
                                        elif si>sj:
                                                fact=cvx.sparse([0])
                                        elif si==sj:
                                                pass
                                quad_terms += zip(fact.I+si,fact.J+sj,2*fact.V)

                #constraints
                
                #progress bar
                if self.options['verbose']>0:
                        print
                        print('adding constraints...')
                        print 
                if self.options['verbose']>1:
                        limitbar= (self.numberAffConstraints +
                                   self.numberQuadConstraints +
                                   len(newcons))
                        prog = ProgressBar(0,limitbar, None, mode='fixed')
                        oldprog = str(prog)
                
                rows=[]
                cols=[]
                vals=[]
                rhs=[]
                senses= ''
                
                ql=[]
                qq=[]
                qc=[]
                
                boundcons={} #dictionary of i,j,b,v for bound constraints
                
                
                #join all constraints
                def join_iter(it1,it2):
                        for i in it1: yield i
                        for i in it2: yield i
                        
                allcons = join_iter(enumerate(self.constraints),
                                    newcons.iteritems())
                
                irow=0
                for constrKey,constr in allcons:
                       
                        if constr.typeOfConstraint[:3] == 'lin':
                                #init of boundcons[key]
                                boundcons[constrKey]=[]
                                
                                #parse the (i,j,v) triple
                                ijv=[]
                                for var,fact in (constr.Exp1-constr.Exp2).factors.iteritems():
                                        if type(fact)!=cvx.base.spmatrix:
                                                fact = cvx.sparse(fact)
                                        sj=var.startIndex
                                        ijv.extend(zip( fact.I,fact.J+sj,fact.V))
                                ijvs=sorted(ijv)
                                
                                itojv={}
                                lasti=-1
                                for (i,j,v) in ijvs:
                                        if i==lasti:
                                                itojv[i].append((j,v))
                                        else:
                                                lasti=i
                                                itojv[i]=[(j,v)]
                                
                                #constant term
                                szcons = constr.Exp1.size[0]*constr.Exp1.size[1]
                                rhstmp = cvx.matrix(0.,(szcons,1))
                                constant1 = constr.Exp1.constant #None or a 1*1 matrix
                                constant2 = constr.Exp2.constant
                                if not constant1 is None:
                                        rhstmp = rhstmp-constant1
                                if not constant2 is None:
                                        rhstmp = rhstmp+constant2
                                                                
                                for i,jv in itojv.iteritems():
                                        r=rhstmp[i]
                                        if len(jv)==1:
                                                #BOUND
                                                j,v=jv[0]
                                                b=r/float(v)
                                                if v>0:
                                                        if constr.typeOfConstraint[:4] in ['lin<','lin=']:
                                                                if b<ub[j]:
                                                                        ub[j]=b
                                                        if constr.typeOfConstraint[:4] in ['lin>','lin=']:
                                                                if b>lb[j]:
                                                                        lb[j]=b
                                                else:#v<0
                                                        if constr.typeOfConstraint[:4] in ['lin<','lin=']:
                                                                if b>lb[j]:
                                                                        lb[j]=b
                                                        if constr.typeOfConstraint[:4] in ['lin>','lin=']:
                                                                if b<ub[j]:
                                                                        ub[j]=b
                                                if constr.typeOfConstraint[3]=='=': 
                                                        b='='
                                                boundcons[constrKey].append((i,j,b,v))
                                        else:
                                                if constr.typeOfConstraint[:4] == 'lin<':
                                                        senses += "L" # lower
                                                elif constr.typeOfConstraint[:4] == 'lin>':
                                                        senses += "G" # greater
                                                elif constr.typeOfConstraint[:4] == 'lin=':
                                                        senses += "E" # equal
                                                
                                                rows.extend([irow]*len(jv))
                                                cols.extend([j for j,v in jv])
                                                vals.extend([v for j,v in jv])
                                                rhs.append(r)
                                                irow+=1
                                                
                                        if self.options['verbose']>1:
                                                #<--display progress
                                                prog.increment_amount()
                                                if oldprog != str(prog):
                                                        print prog, "\r",
                                                        sys.stdout.flush()
                                                        oldprog=str(prog)
                                                #-->                                                
                        
                        elif constr.typeOfConstraint == 'quad':
                                #quad part
                                qind1,qind2,qval=[],[],[]
                                qd=constr.Exp1.quad
                                for i,j in qd:
                                        fact=qd[i,j]
                                        si=i.startIndex
                                        sj=j.startIndex
                                        if (j,i) in qd: #quad stores x'*A1*y + y'*A2*x
                                                if si<sj:
                                                        fact+=qd[j,i].T
                                                elif si>sj:
                                                        fact=cvx.sparse([0])
                                                elif si==sj:
                                                        pass
                                        qind1.extend(fact.I+si)
                                        qind2.extend(fact.J+sj)
                                        qval.extend(fact.V)
                                q_exp=cplex.SparseTriple(ind1 = qind1,
                                                         ind2 = qind2,
                                                         val = qval)
                                #lin part
                                lind,lval=[],[]
                                af=constr.Exp1.aff.factors
                                for var in af:
                                        lind.extend(af[var].J)
                                        lval.extend(af[var].V)
                                l_exp=cplex.SparsePair(ind = lind, val = lval)
                                
                                #constant
                                qcs=0.
                                if not(constr.Exp1.aff.constant is None):
                                        qcs = - constr.Exp1.aff.constant[0]
                                
                                ql+= [l_exp]
                                qq+= [q_exp]
                                qc+= [qcs]
                                
                                if self.options['verbose']>1:
                                        #<--display progress
                                        prog.increment_amount()
                                        if oldprog != str(prog):
                                                print prog, "\r",
                                                sys.stdout.flush()
                                                oldprog=str(prog)
                                        #-->
                                
                        elif constr.typeOfConstraint[2:] == 'cone':
                                pass  #will be handled in the newcons dictionary
                                
                        else:
                                raise Exception('type of constraint not handled (yet ?) for cplex:{0}'.format(
                                        constr.typeOfConstraint))
                                
                      

                if self.options['verbose']>1:
                        prog.update_amount(limitbar)
                        print prog, "\r",
                        print
                
                if self.options['verbose']>0:
                        print
                        print('Passing to cplex...')
                c.variables.add(obj = obj, ub = ub, lb=lb, names = colnames,types=types)
                if len(quad_terms)>0:
                        c.objective.set_quadratic_coefficients(quad_terms)
                c.linear_constraints.add(rhs = rhs, senses = senses)
                if len(rows)>0:
                        c.linear_constraints.set_coefficients(zip(rows, cols, vals))
                for lp,qp,qcs in zip(ql,qq,qc):
                        c.quadratic_constraints.add(lin_expr = lp,
                                                    quad_expr = qp,
                                                    rhs = qcs,
                                                    sense = "L")
                
                
                tp=self.type
                if tp == 'LP':
                        c.set_problem_type(c.problem_type.LP)
                elif tp == 'MIP':
                        c.set_problem_type(c.problem_type.MILP)
                elif tp in ('QCQP','SOCP','Mixed (SOCP+quad)'):
                        c.set_problem_type(c.problem_type.QCP)
                elif tp in ('MIQCP','MISOCP','Mixed (MISOCP+quad)'):
                        c.set_problem_type(c.problem_type.MIQCP)
                elif tp == 'QP':
                        c.set_problem_type(c.problem_type.QP)
                elif tp == 'MIQP':
                        c.set_problem_type(c.problem_type.MIQP)
                else:
                        raise Exception('unhandled type of problem')
                
                
                
                self.cplex_Instance = c
                if self.options['verbose']>0:
                        print('CPLEX INSTANCE created')
                self.cplex_boundcons=boundcons
                
                #remove temporary variables for cone constraints
                for v in tmplhs:
                        self.remove_variable(v.name)
                for v in tmprhs:
                        self.remove_variable(v.name)                       
                if 'noconstant' in newcons:
                        self.remove_variable(noconstant.name)

                
        def _make_cvxopt_instance(self,aff_part_of_quad=True,cone_as_quad=False):
                """
                defines the variables in self.cvxoptVars, used by the cvxopt solver
                """
                ss=self.numberOfVars
                #initial values                
                self.cvxoptVars['A']=cvx.spmatrix([],[],[],(0,ss),tc='d')
                self.cvxoptVars['b']=cvx.matrix([],(0,1),tc='d')
                self.cvxoptVars['Gl']=cvx.spmatrix([],[],[],(0,ss),tc='d')
                self.cvxoptVars['hl']=cvx.matrix([],(0,1),tc='d')
                self.cvxoptVars['Gq']=[]
                self.cvxoptVars['hq']=[]
                self.cvxoptVars['Gs']=[]
                self.cvxoptVars['hs']=[]
                self.cvxoptVars['quadcons']=[]
                #objective
                if isinstance(self.objective[1],QuadExp):
                        self.cvxoptVars['quadcons'].append(('_obj',-1))
                        objexp=self.objective[1].aff
                elif isinstance(self.objective[1],LogSumExp):
                        objexp=self.objective[1].Exp
                else:
                        objexp=self.objective[1]
                if self.numberLSEConstraints==0:
                        if self.objective[0]=='find':
                                self.cvxoptVars['c']=cvx.matrix(0,(ss,1),tc='d')
                        elif self.objective[0]=='min':
                                (c,constantInObjective)=self._makeGandh(objexp)
                                self.cvxoptVars['c']=cvx.matrix(c,tc='d').T
                        elif self.objective[0]=='max':
                                (c,constantInObjective)=self._makeGandh(objexp)
                                self.cvxoptVars['c']=-cvx.matrix(c,tc='d').T
                else:
                        if self.objective[0]=='find':
                                self.cvxoptVars['F']=cvx.matrix(0,(1,ss),tc='d')
                                self.cvxoptVars['K']=[0]
                        else:
                                (F,g)=self._makeGandh(objexp)
                                self.cvxoptVars['K']=[F.size[0]]
                                if self.objective[0]=='min':
                                        self.cvxoptVars['F']=cvx.matrix(F,tc='d')
                                        self.cvxoptVars['g']=cvx.matrix(g,tc='d')
                                elif self.objective[0]=='max':
                                        self.cvxoptVars['F']=-cvx.matrix(F,tc='d')
                                        self.cvxoptVars['g']=-cvx.matrix(g,tc='d')
                
                if not(aff_part_of_quad) and isinstance(self.objective[1],QuadExp):
                        self.cvxoptVars['c']=cvx.matrix(0,(ss,1),tc='d')

                if self.options['verbose']>1:
                        limitbar=self.numberAffConstraints + self.numberConeConstraints + self.numberQuadConstraints + self.numberLSEConstraints + self.numberSDPConstraints
                        prog = ProgressBar(0,limitbar, None, mode='fixed')
                        oldprog = str(prog)
                
                #constraints                
                for k in range(len(self.constraints)):
                        #linear constraints                        
                        if self.constraints[k].typeOfConstraint[:3]=='lin':
                                sense=self.constraints[k].typeOfConstraint[3]
                                (G_lhs,h_lhs)=self._makeGandh(self.constraints[k].Exp1)
                                (G_rhs,h_rhs)=self._makeGandh(self.constraints[k].Exp2)
                                if sense=='=':
                                        self.cvxoptVars['A']=cvx.sparse([self.cvxoptVars['A'],G_lhs-G_rhs])
                                        self.cvxoptVars['b']=cvx.matrix([self.cvxoptVars['b'],h_rhs-h_lhs])
                                elif sense=='<':
                                        self.cvxoptVars['Gl']=cvx.sparse([self.cvxoptVars['Gl'],G_lhs-G_rhs])
                                        self.cvxoptVars['hl']=cvx.matrix([self.cvxoptVars['hl'],h_rhs-h_lhs])
                                elif sense=='>':
                                        self.cvxoptVars['Gl']=cvx.sparse([self.cvxoptVars['Gl'],G_rhs-G_lhs])
                                        self.cvxoptVars['hl']=cvx.matrix([self.cvxoptVars['hl'],h_lhs-h_rhs])
                                else:
                                        raise NameError('unexpected case')
                        elif self.constraints[k].typeOfConstraint=='SOcone':
                                if not(cone_as_quad):
                                        (A,b)=self._makeGandh(self.constraints[k].Exp1)
                                        (c,d)=self._makeGandh(self.constraints[k].Exp2)
                                        self.cvxoptVars['Gq'].append(cvx.sparse([-c,-A]))
                                        self.cvxoptVars['hq'].append(cvx.matrix([d,b]))
                                else:
                                        self.cvxoptVars['quadcons'].append(
                                                (k,self.cvxoptVars['Gl'].size[0]))
                                        if aff_part_of_quad:
                                                raise Exception('cone_as_quad + aff_part_of_quad')
                        elif self.constraints[k].typeOfConstraint=='RScone':
                                if not(cone_as_quad):
                                        (A,b)=self._makeGandh(self.constraints[k].Exp1)
                                        (c1,d1)=self._makeGandh(self.constraints[k].Exp2)
                                        (c2,d2)=self._makeGandh(self.constraints[k].Exp3)
                                        self.cvxoptVars['Gq'].append(cvx.sparse([-c1-c2,-2*A,c2-c1]))
                                        self.cvxoptVars['hq'].append(cvx.matrix([d1+d2,2*b,d1-d2]))
                                else:
                                        self.cvxoptVars['quadcons'].append(
                                                (k,self.cvxoptVars['Gl'].size[0]))
                                        if aff_part_of_quad:
                                                raise Exception('cone_as_quad + aff_part_of_quad')
                        elif self.constraints[k].typeOfConstraint=='lse':
                                (F,g)=self._makeGandh(self.constraints[k].Exp1)
                                self.cvxoptVars['F']=cvx.sparse([self.cvxoptVars['F'],F])
                                self.cvxoptVars['g']=cvx.matrix([self.cvxoptVars['g'],g])
                                self.cvxoptVars['K'].append(F.size[0])
                        elif self.constraints[k].typeOfConstraint=='quad':
                                self.cvxoptVars['quadcons'].append((k,self.cvxoptVars['Gl'].size[0]))
                                if aff_part_of_quad:
                                        #quadratic part handled later
                                        (G_lhs,h_lhs)=self._makeGandh(self.constraints[k].Exp1.aff)
                                        self.cvxoptVars['Gl']=cvx.sparse([self.cvxoptVars['Gl'],G_lhs])
                                        self.cvxoptVars['hl']=cvx.matrix([self.cvxoptVars['hl'],-h_lhs])
                        elif self.constraints[k].typeOfConstraint[:3]=='sdp':
                                sense=self.constraints[k].typeOfConstraint[3]
                                (G_lhs,h_lhs)=self._makeGandh(self.constraints[k].Exp1)
                                (G_rhs,h_rhs)=self._makeGandh(self.constraints[k].Exp2)
                                if sense=='<':
                                        self.cvxoptVars['Gs'].append(G_lhs-G_rhs)
                                        self.cvxoptVars['hs'].append(h_rhs-h_lhs)
                                elif sense=='>':
                                        self.cvxoptVars['Gs'].append(G_rhs-G_lhs)
                                        self.cvxoptVars['hs'].append(h_lhs-h_rhs)
                                else:
                                        raise NameError('unexpected case')
                        else:
                                raise NameError('unexpected case')
                        if self.options['verbose']>1:
                                #<--display progress
                                prog.increment_amount()
                                if oldprog != str(prog):
                                        print prog, "\r",
                                        sys.stdout.flush()
                                        oldprog=str(prog)
                                #-->
                        
                #reshape hs matrices as square matrices
                #for m in self.cvxoptVars['hs']:
                #        n=int(np.sqrt(len(m)))
                #        m.size=(n,n)
                   
                if self.options['verbose']>1:
                        prog.update_amount(limitbar)
                        print prog, "\r",
                        sys.stdout.flush()
                        print

        #-----------
        #mosek tool
        #-----------
        
        # Define a stream printer to grab output from MOSEK
        def _streamprinter(self,text):
                sys.stdout.write(text)
                sys.stdout.flush()

        def _make_mosek_instance(self):
                """
                defines the variables msk_env and msk_task used by the solver mosek.
                """
                if self.options['verbose']>0:
                        print('build mosek instance')
                
                try:
                        import mosek
                except ImportError:
                        print('mosek library not found')

                #only change the objective coefficients
                if self.options['onlyChangeObjective']:
                        if self.msk_task is None:
                                raise Exception('option is only available when msk_task has been defined before')
                        newobj=self.objective[1]
                        (cobj,constantInObjective)=self._makeGandh(newobj)
                        self.cvxoptVars['c']=cvx.matrix(cobj,tc='d').T
                        
                        for j in range(len(self.cvxoptVars['c'])):
                        # Set the linear term c_j in the objective.
                                self.msk_task.putcj(j,self.cvxoptVars['c'][j])
                        return
                                
                # Make a MOSEK environment
                env = mosek.Env ()
                # Attach a printer to the environment
                if self.options['verbose']>=1:
                        env.set_Stream (mosek.streamtype.log, self._streamprinter)
                # Create a task
                task = env.Task(0,0)
                # Attach a printer to the task
                if self.options['verbose']>=1:
                        task.set_Stream (mosek.streamtype.log, self._streamprinter)                                
                                
                                
                                
                #patch for quadratic problems with a single var
                if self.numberOfVars==1 and self.numberQuadConstraints>0:
                        if '_ptch_' not in self.variables:
                                ptch=self.add_variable('_ptch_',1)
                        else:
                                ptch=self.get_variable('_ptch_')
                        self.add_constraint( ptch>0 )                                
                                
                                
                # Give MOSEK an estimate of the size of the input data.
                # This is done to increase the speed of inputting data.                                
                                
                self._make_cvxopt_instance()
                NUMVAR = self.numberOfVars+int(sum([Gk.size[0] for Gk in self.cvxoptVars['Gq']]))
                NUMCON = self.numberAffConstraints+int(sum([Gk.size[0] for Gk in self.cvxoptVars['Gq']]))
                NUMCONE = self.numberConeConstraints
                NUMANZ= len(self.cvxoptVars['A'].I)+len(self.cvxoptVars['Gl'].I)
                NUMQNZ= self.numberQuadNNZ

                if bool(self.cvxoptVars['Gs']) or bool(self.cvxoptVars['F']):
                        raise Exception('SDP or GP constraints are not implemented in mosek')

                # Append 'NUMCON' empty constraints.
                # The constraints will initially have no bounds.
                task.append(mosek.accmode.con,NUMCON)
                #Append 'NUMVAR' variables.
                # The variables will initially be fixed at zero (x=0).
                task.append(mosek.accmode.var,NUMVAR)

                #specifies the integer variables
                for k in self.variables:
                        if self.variables[k].vtype=='binary':
                                raise Exception('not implemented yet')
                        elif self.variables[k].vtype=='integer':
                                for i in xrange(self.variables[k].startIndex,self.variables[k].endIndex):
                                        task.putvartype(i,mosek.variabletype.type_int)

                for j in range(NUMVAR):
                        # Set the linear term c_j in the objective.
                        if j< self.numberOfVars:
                                if self.objective[0]=='max':         #max is handled directly by MOSEK,
                                                                #revert to initial value        
                                        task.putcj(j,-self.cvxoptVars['c'][j])
                                else:
                                        task.putcj(j,self.cvxoptVars['c'][j])
                        #make the variable free
                        task.putbound(mosek.accmode.var,j,mosek.boundkey.fr,0.,0.)

                #equality constraints:
                Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                ijvs=sorted(zip(Ai,Aj,Av))
                del Ai,Aj,Av
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                iaff=0
                for i,jv in itojv.iteritems():
                        J=[jvk[0] for jvk in jv]
                        V=[jvk[1] for jvk in jv]
                        if len(J)==1:
                                #fixed variable
                                b=self.cvxoptVars['b'][i]/V[0]
                                task.putbound(mosek.accmode.var,J[0],mosek.boundkey.fx,b,b)
                        else:
                        
                                #affine inequality
                                b=self.cvxoptVars['b'][i]
                                task.putaijlist([iaff]*len(J),J,V)
                                task.putbound(mosek.accmode.con,iaff,mosek.boundkey.fx,
                                                b,b)
                                iaff+=1

                #inequality constraints:
                Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                ijvs=sorted(zip(Gli,Glj,Glv))
                del Gli,Glj,Glv
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                        
                for i,jv in itojv.iteritems():
                        J=[jvk[0] for jvk in jv]
                        V=[jvk[1] for jvk in jv]
                        if len(J)==1 and (not (i in [t[1] for t in self.cvxoptVars['quadcons']])):
                                #bounded variable
                                bk,bl,bu=task.getbound(mosek.accmode.var,J[0])
                                b=self.cvxoptVars['hl'][i]/V[0]
                                if V[0]>0:
                                        #less than
                                        bu=min(b,bu)
                                if V[0]<0:
                                        #greater than
                                        bl=max(b,bl)
                                if bu==bl:
                                        task.putbound(mosek.accmode.var,J[0],mosek.boundkey.fx,bl,bu)
                                elif bl>bu:
                                        raise Exception('unfeasible bound for var '+str(J[0]))
                                else:
                                        if bl<-INFINITY:
                                                if bu>INFINITY:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.fr,bl,bu)
                                                else:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.up,bl,bu)
                                        else:
                                                if bu>INFINITY:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.lo,bl,bu)
                                                else:
                                                        task.putbound(mosek.accmode.var,
                                                        J[0],mosek.boundkey.ra,bl,bu)
                        else:
                                #affine inequality
                                b=self.cvxoptVars['hl'][i]
                                task.putaijlist([iaff]*len(J),J,V)
                                task.putbound(mosek.accmode.con,iaff,mosek.boundkey.up,-INFINITY,b)
                                if i in [t[1] for t in self.cvxoptVars['quadcons']]:
                                        #affine part of a quadratic constraint
                                        qcons= [qc for (qc,l) in self.cvxoptVars['quadcons'] if l==i][0]
                                        qconsindex=self.cvxoptVars['quadcons'].index((qcons,i))
                                        self.cvxoptVars['quadcons'][qconsindex]=(qcons,iaff)
                                        #we replace the line number in Gl by the index of the MOSEK constraint
                                iaff+=1
                
                #conic constraints:
                icone=self.numberOfVars
                for k in range(NUMCONE):
                        #Gk x + sk = hk
                        istart=icone
                        for i in range(self.cvxoptVars['Gq'][k].size[0]):
                                J=list(self.cvxoptVars['Gq'][k][i,:].J)
                                V=list(self.cvxoptVars['Gq'][k][i,:].V)
                                h=self.cvxoptVars['hq'][k][i]
                                J.append(icone)
                                V.append(1)
                                task.putaijlist([iaff]*(1+len(J)),J,V)
                                task.putbound(mosek.accmode.con,iaff,mosek.boundkey.fx,h,h)
                                iaff+=1
                                icone+=1
                        iend=icone
                        #sk in quadratic cone
                        task.appendcone(mosek.conetype.quad, 0.0, range(istart,iend))

                #quadratic constraints:
                task.putmaxnumqnz(NUMQNZ)
                for (k,iaff) in self.cvxoptVars['quadcons']:
                        subI=[]
                        subJ=[]
                        subV=[]
                        if k=='_obj':
                                qexpr=self.objective[1]
                        else:
                                qexpr=self.constraints[k].Exp1

                        for i,j in qexpr.quad:
                                si,ei=i.startIndex,i.endIndex
                                sj,ej=j.startIndex,j.endIndex
                                Qij=qexpr.quad[i,j]
                                if not isinstance(Qij,cvx.spmatrix):
                                        Qij=cvx.sparse(Qij)
                                if si==sj:#put A+A' on the diag
                                        sI=list((Qij+Qij.T).I+si)
                                        sJ=list((Qij+Qij.T).J+sj)
                                        sV=list((Qij+Qij.T).V)
                                        for u in range(len(sI)-1,-1,-1):
                                                #remove when j>i
                                                if sJ[u]>sI[u]:
                                                        del sI[u]
                                                        del sJ[u]
                                                        del sV[u]
                                elif si>=ej: #add A in the lower triang
                                        sI=list(Qij.I+si)
                                        sJ=list(Qij.J+sj)
                                        sV=list(Qij.V)
                                else: #add B' in the lower triang
                                        sI=list((Qij.T).I+sj)
                                        sJ=list((Qij.T).J+si)
                                        sV=list((Qij.T).V)
                                subI.extend(sI)
                                subJ.extend(sJ)
                                subV.extend(sV)
                        
                        if k=='_obj':
                                task.putqobj(subI,subJ,subV)
                        else:
                                task.putqconk(iaff,subI,subJ,subV)
                #objective sense
                if self.objective[0]=='max':
                        task.putobjsense(mosek.objsense.maximize)
                else:
                        task.putobjsense(mosek.objsense.minimize)
                
                self.msk_env=env
                self.msk_task=task

                if self.options['verbose']>0:
                        print('mosek instance built')


        def _make_zibopt(self):
                """
                Defines the variables scip_solver, scip_vars and scip_obj,
                used by the zibopt solver.
                """
                try:
                        from zibopt import scip
                except:
                        raise Exception('scip library not found')
                
                scip_solver = scip.solver(quiet=not(self.options['verbose']))
                
                self._make_cvxopt_instance(aff_part_of_quad=False,cone_as_quad=True)
                
                if bool(self.cvxoptVars['Gs']) or bool(self.cvxoptVars['F']) or bool(self.cvxoptVars['Gq']):
                        raise Exception('SDP, SOCP, or GP constraints are not implemented in zibopt')
                                
                #max handled directly by scip
                if self.objective[0]=='max':
                        self.cvxoptVars['c']=-self.cvxoptVars['c']
                
                zib_types={ 'continuous':scip.CONTINUOUS,
                            'integer'   :scip.INTEGER,
                            'binary'    :scip.BINARY,
                           }
                types=[0]*self.cvxoptVars['A'].size[1]
                for var in self.variables.keys():
                                si=self.variables[var].startIndex
                                ei=self.variables[var].endIndex
                                vtype=self.variables[var].vtype
                                try:
                                        types[si:ei]=[zib_types[vtype]]*(ei-si)
                                except:
                                        raise Exception('this vtype is not handled by scip: '+str(vtype))
                
                x=[]
                INFINITYZO = 1e10
                for i in range(self.cvxoptVars['A'].size[1]):
                    if not(self.cvxoptVars['c'] is None):
                        x.append(scip_solver.variable(types[i],
                                lower=-INFINITYZO,
                                upper=INFINITYZO,
                                coefficient=self.cvxoptVars['c'][i])
                            )
                    else:
                        x.append(scip_solver.variable(types[i],
                                lower=-INFINITYZO,
                                upper=INFINITYZO
                                ))
                
                #equalities
                Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                ijvs=sorted(zip(Ai,Aj,Av))
                del Ai,Aj,Av
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                        
                for i,jv in itojv.iteritems():
                        exp=0
                        for term in jv:
                                exp+= term[1]*x[term[0]]
                        scip_solver += exp == self.cvxoptVars['b'][i]
                        
                #inequalities
                Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                ijvs=sorted(zip(Gli,Glj,Glv))
                del Gli,Glj,Glv
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                        
                for i,jv in itojv.iteritems():
                        exp=0
                        for term in jv:
                                exp+= term[1]*x[term[0]]
                        scip_solver += exp <= self.cvxoptVars['hl'][i]

                
                ###
                #quadratic constraints (including SOC constraints)
                for (k,iaff) in self.cvxoptVars['quadcons']:
                        subI=[]
                        subJ=[]
                        subV=[]
                        if k=='_obj':
                                x.append(scip_solver.variable(
                                        zib_types['continuous'],
                                        lower=-INFINITYZO,
                                        upper=INFINITYZO
                                        ))
                                qexpr=self.objective[1]
                        else:
                                if self.constraints[k].typeOfConstraint=='quad':
                                        qexpr=self.constraints[k].Exp1
                                if self.constraints[k].typeOfConstraint=='SOcone':
                                        qexpr=(self.constraints[k].Exp1|self.constraints[k].Exp1)-(
                                                self.constraints[k].Exp2*self.constraints[k].Exp2)
                                        (e2x,e2c)=self._makeGandh(self.constraints[k].Exp2)
                                        exp=e2c[0]
                                        for j,v in zip(e2x.J,e2x.V):
                                                exp+=v*x[j]
                                        scip_solver += exp >=0
                                if self.constraints[k].typeOfConstraint=='RScone':
                                        qexpr=(self.constraints[k].Exp1|self.constraints[k].Exp1)-(
                                                self.constraints[k].Exp2*self.constraints[k].Exp3)
                                        (e2x,e2c)=self._makeGandh(self.constraints[k].Exp2)
                                        exp=e2c[0]
                                        for j,v in zip(e2x.J,e2x.V):
                                                exp+=v*x[j]
                                        scip_solver += exp >=0

                        qd=0
                        for i,j in qexpr.quad:
                                si,ei=i.startIndex,i.endIndex
                                sj,ej=j.startIndex,j.endIndex
                                Qij=qexpr.quad[i,j]
                                if not isinstance(Qij,cvx.spmatrix):
                                        Qij=cvx.sparse(Qij)
                                for ii,jj,vv in zip(Qij.I,Qij.J,Qij.V):
                                        qd+=vv*x[ii+si]*x[jj+sj]
                        
                        if not(qexpr.aff is None):
                                for v,fac in qexpr.aff.factors.iteritems():
                                        if not isinstance(fac,cvx.spmatrix):
                                                fac=cvx.sparse(fac)
                                        sv=v.startIndex
                                        for jj,vv in zip(fac.J,fac.V):
                                                qd+=vv*x[jj+sv]
                                if not(qexpr.aff.constant is None):
                                        qd+=qexpr.aff.constant[0]
                        
                        if k=='_obj':
                                if self.objective[0]=='max':
                                        scip_solver += (x[-1]-qd) <= 0
                                else:
                                        scip_solver += (qd-x[-1]) <= 0
                                self.scip_obj = x[-1]
                        else:
                                scip_solver += qd <= 0
                ###
                
                self.scip_solver=scip_solver
                self.scip_vars=x
                
                
                
                

                
        """
        -----------------------------------------------
        --                CALL THE SOLVER            --
        -----------------------------------------------
        """        

        def solve(self, **options):
                """
                Solves the problem.
                
                Once the problem has been solved, the optimal variables
                can be obtained thanks to the property :attr:`value <picos.Expression.value>`
                of the class :class:`Expression<picos.Expression>`.
                The optimal dual variables can be accessed by the method
                :func:`picos.Constraint.dual` of the class
                :class:`Constraint<picos.Constraint>`.
                
                :keyword options: A list of options to update before
                                  the call to the solver. In particular, 
                                  the solver can
                                  be specified here,
                                  under the form ``key = value``.
                                  See the list of available options
                                  in the doc of 
                                  :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`
                :returns: A dictionary **sol** which contains the information
                            returned by the solver.
                
                """
                if options is None: options={}
                self.update_options(**options)
                if self.options['solver'] is None:
                        self.solver_selection()
                        

                #self._eliminate_useless_variables()

                if isinstance(self.objective[1],GeneralFun):
                        return self._sqpsolve(options)
                
                #WARNING: Bug with cvxopt-mosek ?
                if (self.options['solver']=='CVXOPT' #obolete name, use lower case
                    or self.options['solver']=='cvxopt-mosek'
                    or self.options['solver']=='smcp'
                    or self.options['solver']=='cvxopt'):

                        primals,duals,obj,sol=self._cvxopt_solve()
                        
                # For cplex
                elif (self.options['solver']=='cplex'):
                        
                        primals,duals,obj,sol=self._cplex_solve()

                # for mosek
                elif (self.options['solver']=='MSK' #obsolete value, use lower case
                        or self.options['solver']=='mosek'):
                        
                        primals,duals,obj,sol=self._mosek_solve()

                
                elif (self.options['solver']=='zibopt'):
                        
                        primals,duals,obj,sol=self._zibopt_solve()

                else:
                        raise Exception('unknown solver')                       
                                        
                for k in primals.keys():
                        if not primals[k] is None:
                                self.set_var_value(k,primals[k],optimalvar=True)
                                
                if 'noduals' in self.options and self.options['noduals']:
                        pass
                else:
                        for i,d in enumerate(duals):
                                self.constraints[i].set_dualVar(d)
                if obj=='toEval' and not(self.objective[1] is None):
                        obj=self.objective[1].eval()[0]
                sol['obj']=obj
                self.status=sol['status']
                return sol

                
        def _cvxopt_solve(self):
                """
                Solves a problem with the cvxopt solver.
                """
                
                #--------------------#
                # makes the instance #
                #--------------------#
                
                if self.options['onlyChangeObjective']:
                        if self.cvxoptVars['c'] is None:
                                raise Exception('option is only available when cvxoptVars has been defined before')
                        newobj=self.objective[1]
                        (cobj,constantInObjective)=self._makeGandh(newobj)
                        self.cvxoptVars['c']=cvx.matrix(cobj,tc='d').T
                else:
                        self._make_cvxopt_instance()
                #--------------------#        
                #  sets the options  #
                #--------------------#
                import cvxopt.solvers
                cvx.solvers.options['maxiters']=self.options['maxit']
                cvx.solvers.options['abstol']=self.options['abstol']
                cvx.solvers.options['feastol']=self.options['feastol']
                cvx.solvers.options['reltol']=self.options['reltol']
                cvx.solvers.options['show_progress']=bool(self.options['verbose']>0)
                try:
                        import smcp.solvers
                        smcp.solvers.options['maxiters']=self.options['maxit']
                        smcp.solvers.options['abstol']=self.options['abstol']
                        smcp.solvers.options['feastol']=self.options['feastol']
                        smcp.solvers.options['reltol']=self.options['reltol']
                        smcp.solvers.options['show_progress']=bool(self.options['verbose']>0)
                except:
                        #smcp is not available
                        pass
                
                if self.options['solver'].upper()=='CVXOPT':
                        currentsolver=None
                elif self.options['solver']=='cvxopt-mosek':
                        currentsolver='mosek'
                elif self.options['solver']=='smcp':
                        currentsolver='smcp'
                #-------------------------------#
                #  runs the appropriate solver  #
                #-------------------------------#
                if  self.numberQuadConstraints>0:#(QC)QP
                        probtype='QCQP'
                        raise Exception('Please convert the quadratic constraints as cone constraints '+
                                                'with the function convert_quad_to_socp().')
                elif self.numberLSEConstraints>0:#GP
                        if len(self.cvxoptVars['Gq'])+len(self.cvxoptVars['Gs'])>0:
                                raise Exception('cone constraints + LSE not implemented')
                        probtype='GP'
                        if self.options['verbose']>0:
                                print '-----------------------------------'
                                print '         cvxopt GP solver'
                                print '-----------------------------------'
                        sol=cvx.solvers.gp(self.cvxoptVars['K'],
                                                self.cvxoptVars['F'],self.cvxoptVars['g'],
                                                self.cvxoptVars['Gl'],self.cvxoptVars['hl'],
                                                self.cvxoptVars['A'],self.cvxoptVars['b'])
                #changes to adapt the problem for the conelp interface:
                elif currentsolver=='mosek':
                        if len(self.cvxoptVars['Gs'])>0:
                                raise Exception('CVXOPT does not handle SDP with MOSEK')                            
                        if len(self.cvxoptVars['Gq'])+len(self.cvxoptVars['Gs']):
                                if self.options['verbose']>0:
                                        print '------------------------------------------'
                                        print '  mosek LP solver interfaced by cvxopt'
                                        print '------------------------------------------'
                                sol=cvx.solvers.lp(self.cvxoptVars['c'],
                                                self.cvxoptVars['Gl'],self.cvxoptVars['hl'],
                                                self.cvxoptVars['A'],self.cvxoptVars['b'],
                                                solver=currentsolver)
                                probtype='LP'
                        else:
                                if self.options['verbose']>0:
                                        print '-------------------------------------------'
                                        print '  mosek SOCP solver interfaced by cvxopt'
                                        print '-------------------------------------------'
                                sol=cvx.solvers.socp(self.cvxoptVars['c'],
                                                        self.cvxoptVars['Gl'],self.cvxoptVars['hl'],
                                                        self.cvxoptVars['Gq'],self.cvxoptVars['hq'],
                                                        self.cvxoptVars['A'],self.cvxoptVars['b'],
                                                        solver=currentsolver)
                                probtype='SOCP'
                else:
                        dims={}
                        dims['s']=[int(np.sqrt(Gsi.size[0])) for Gsi in self.cvxoptVars['Gs']]
                        dims['l']=self.cvxoptVars['Gl'].size[0]
                        dims['q']=[Gqi.size[0] for Gqi in self.cvxoptVars['Gq']]
                        G=self.cvxoptVars['Gl']
                        h=self.cvxoptVars['hl']
                        # handle the equalities as 2 ineq for smcp
                        if currentsolver=='smcp':
                                if self.cvxoptVars['A'].size[0]>0:
                                       G=cvx.sparse([G,self.cvxoptVars['A']]) 
                                       G=cvx.sparse([G,-self.cvxoptVars['A']])
                                       h=cvx.matrix([h,self.cvxoptVars['b']])
                                       h=cvx.matrix([h,-self.cvxoptVars['b']])
                                       dims['l']+=(2*self.cvxoptVars['A'].size[0])

                        for i in range(len(dims['q'])):
                                G=cvx.sparse([G,self.cvxoptVars['Gq'][i]])
                                h=cvx.matrix([h,self.cvxoptVars['hq'][i]])

                                         
                        for i in range(len(dims['s'])):
                                G=cvx.sparse([G,self.cvxoptVars['Gs'][i]])
                                h=cvx.matrix([h,self.cvxoptVars['hs'][i]])

                        #Remove the lines in A and b corresponding to 0==0        
                        JP=list(set(self.cvxoptVars['A'].I))
                        IP=range(len(JP))
                        VP=[1]*len(JP)
                        
                        idx_0eq0 = [i for i in range(self.cvxoptVars['A'].size[0]) if i not in JP]
                        
                        #is there a constraint of the form 0==a(a not 0) ?
                        if any([b for (i,b) in enumerate(self.cvxoptVars['b']) if i not in JP]):
                                raise Exception('infeasible constraint of the form 0=a')
                        P=cvx.spmatrix(VP,IP,JP,(len(IP),self.cvxoptVars['A'].size[0]))
                        self.cvxoptVars['A']=P*self.cvxoptVars['A']
                        self.cvxoptVars['b']=P*self.cvxoptVars['b']
                        
                                
                        if currentsolver=='smcp':
                                try:
                                        import smcp
                                except:
                                        raise Exception('library smcp not found')
                                if self.options['smcp_feas']:
                                        sol=smcp.solvers.conelp(self.cvxoptVars['c'],
                                                        G,h,dims,feas=self.options['smcp_feas'])
                                else:
                                        sol=smcp.solvers.conelp(self.cvxoptVars['c'],
                                                        G,h,dims)
                        else:

                                if self.options['verbose']>0:
                                        print '--------------------------'
                                        print '  cvxopt CONELP solver'
                                        print '--------------------------'
                                sol=cvx.solvers.conelp(self.cvxoptVars['c'],
                                                        G,h,dims,
                                                        self.cvxoptVars['A'],
                                                        self.cvxoptVars['b'])
                        probtype='ConeLP'

                status=sol['status']
                solv=currentsolver
                if solv is None: solv='cvxopt'
                if self.options['verbose']>0:
                        print solv+' status: '+status
                                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                
                try:
                        primals={}
                        for var in self.variables.values():
                                si=var.startIndex
                                ei=var.endIndex
                                varvect=sol['x'][si:ei]
                                if var.vtype=='symmetric':
                                        varvect=svecm1(varvect) #varvect was the svec
                                                                #representation of X
                                
                                primals[var.name]=cvx.matrix(varvect, var.size)
                except Exception as ex:
                        import warnings
                        warnings.warn('error while retrieving primals')
                        primals = {}
                        if self.options['verbose']>0:
                                print('##################################')
                                print('WARNING: Primal Solution Not Found')
                                print('##################################')
                                               

                #--------------------#
                # retrieve the duals #
                #--------------------#
                duals=[]
                if 'noduals' in self.options and self.options['noduals']:
                        pass
                else:
                        
                        try:
                                printnodual=False
                                (indy,indzl,indzq,indznl,indzs)=(0,0,0,0,0)
                                if probtype=='LP' or probtype=='ConeLP':
                                        zkey='z'
                                else:
                                        zkey='zl'
                                zqkey='zq'
                                zskey='zs'
                                if probtype=='ConeLP':
                                        indzq=dims['l']
                                        zqkey='z'
                                        zskey='z'
                                        indzs=dims['l']+sum(dims['q'])
                                ykey='y'
                                if currentsolver=='smcp':
                                        nbeq=self.cvxoptVars['A'].size[0]
                                        indy=dims['l']-2*nbeq
                                        ykey='z'

                                for k in range(len(self.constraints)):
                                        #Equality
                                        if self.constraints[k].typeOfConstraint=='lin=':
                                                if not (sol[ykey] is None):
                                                        consSz=np.product(self.constraints[k].Exp1.size)
                                                        duals.append((P.T*sol[ykey])[indy:indy+consSz])
                                                        indy+=consSz
                                                        if currentsolver=='smcp':
                                                                dualm=sol[ykey][indy-consSz+nbeq:indy+nbeq]
                                                                duals[-1]-=dualm
                                                else:
                                                        printnodual=True
                                                        duals.append(None)
                                        #Inequality
                                        elif self.constraints[k].typeOfConstraint[:3]=='lin':
                                                if not (sol[zkey] is None):
                                                        consSz=np.product(self.constraints[k].Exp1.size)
                                                        if self.constraints[k].typeOfConstraint[3]=='<':
                                                                duals.append(sol[zkey][indzl:indzl+consSz])
                                                        else:
                                                                duals.append(sol[zkey][indzl:indzl+consSz])
                                                        indzl+=consSz
                                                else:
                                                        printnodual=True
                                                        duals.append(None)
                                        #SOCP constraint [Rotated or not]
                                        elif self.constraints[k].typeOfConstraint[2:]=='cone':
                                                if not (sol[zqkey] is None):
                                                        if probtype=='ConeLP':
                                                                consSz=np.product(self.constraints[k].Exp1.size)+1
                                                                duals.append(sol[zqkey][indzq:indzq+consSz])
                                                                indzq+=consSz
                                                        else:
                                                                duals.append(sol[zqkey][indzq])
                                                                indzq+=1
                                                else:
                                                        printnodual=True
                                                        duals.append(None)
                                        #SDP constraint
                                        elif self.constraints[k].typeOfConstraint[:3]=='sdp':
                                                if not (sol[zskey] is None):
                                                        if probtype=='ConeLP':
                                                                consSz=np.product(self.constraints[k].Exp1.size)
                                                                duals.append(sol[zskey][indzs:indzs+consSz])
                                                                indzs+=consSz
                                                        else:
                                                                duals.append(sol[zskey][indzs])
                                                                indzs+=1
                                                else:
                                                        printnodual=True
                                                        duals.append(None)
                                        #GP constraint
                                        elif self.constraints[k].typeOfConstraint=='lse':
                                                if not (sol['znl'] is None):
                                                        consSz=np.product(self.constraints[k].Exp1.size)
                                                        duals.append(sol['znl'][indznl:indznl+consSz])
                                                        indznl+=consSz
                                                else:
                                                        printnodual=True
                                                        duals.append(None)
                                        else:
                                                raise Exception('constraint cannot be handled')
                                        
                                if printnodual and self.options['verbose']>0:
                                        print('################################')
                                        print('WARNING: Dual Solution Not Found')
                                        print('################################')
                        
                        except Exception as ex:
                                import warnings
                                warnings.warn('error while retrieving duals')
                                duals = []
                                if self.options['verbose']>0:
                                        print('################################')
                                        print('WARNING: Dual Solution Not Found')
                                        print('################################')
                
                #-----------------#
                # objective value #
                #-----------------#
                if self.numberLSEConstraints>0:#GP
                        obj='toEval'
                else:#LP or SOCP
                        if sol['primal objective'] is None:
                                if sol['dual objective'] is None:
                                        obj=None
                                else:
                                        obj=sol['dual objective']
                        else:
                                if sol['dual objective'] is None:
                                        obj=sol['primal objective']
                                else:
                                        obj=0.5*(sol['primal objective']+sol['dual objective'])
                        
                        if self.objective[0]=='max' and not obj is None:
                                obj = -obj
                
                solt={'cvxopt_sol':sol,'status':status}
                return (primals,duals,obj,solt)
 
        
        def  _cplex_solve(self):
                """
                Solves a problem with the cvxopt solver.
                
                .. Todo::  * set solver settings
                           
                """
                #----------------------------#
                #  create the cplex instance #
                #----------------------------#
                import cplex
                self._make_cplex_instance()
                c = self.cplex_Instance
                
                if c is None:
                        raise ValueError('a cplex instance should have been created before')
                
                
                #TODO : setting parameters
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
                if not self.options['treememory'] is None:
                        c.parameters.mip.limits.treememory.set(self.options['treememory'])
                if not self.options['gaplim'] is None:
                        c.parameters.mip.tolerances.mipgap.set(self.options['gaplim'])
                #pool of solutions
                if not self.options['pool_size'] is None:
                        c.parameters.mip.limits.solutions.set(self.options['pool_size'])
                if not self.options['pool_gap'] is None:
                        c.parameters.mip.pool.relgap.set(self.options['pool_gap'])
                #verbosity
                c.parameters.barrier.display.set(min(2,self.options['verbose']))
                c.parameters.simplex.display.set(min(2,self.options['verbose']))
                if self.options['verbose']==0:
                        c.parameters.mip.display.set(0)
                        
                #convergence tolerance
                c.parameters.barrier.qcpconvergetol.set(self.options['tol'])
                c.parameters.barrier.convergetol.set(self.options['tol'])
                
                #iterations limit
                c.parameters.barrier.limits.iteration.set(self.options['maxit'])
                c.parameters.simplex.limits.iterations.set(self.options['maxit'])
                
                #lpmethod
                if not self.options['lp_root_method'] is None:
                        if self.options['lp_root_method']=='psimplex':
                                c.parameters.lpmethod.set(1)
                        elif self.options['lp_root_method']=='dsimplex':
                                c.parameters.lpmethod.set(2)
                        elif self.options['lp_root_method']=='interior':
                                c.parameters.lpmethod.set(4)
                        else:
                                raise Exception('unexpected value for lp_root_method')
                if not self.options['lp_node_method'] is None:
                        if self.options['lp_node_method']=='psimplex':
                                c.parameters.mip.strategy.subalgorithm.set(1)
                        elif self.options['lp_node_method']=='dsimplex':
                                c.parameters.mip.strategy.subalgorithm.set(2)
                        elif self.options['lp_node_method']=='interior':
                                c.parameters.mip.strategy.subalgorithm.set(4)
                        else:
                                raise Exception('unexpected value for lp_node_method')

                if not self.options['nbsol'] is None:
                        c.parameters.mip.limits.nodes.set(self.options['nbsol'])
                
                
                #--------------------#
                #  call the solver   #
                #--------------------#                
              
                if not self.options['pool_size'] is None:
                        try:
                                c.populate_solution_pool()
                        except:
                                print "Exception raised during populate"
                else:
                        try:
                                c.solve()
                        except cplex.exceptions.CplexSolverError as ex:
                                if ex.args[2] == 5002:
                                        raise Exception('Error raised during solve. Problem is nonconvex')
                                else:
                                        print "Exception raised during solve"
                                
        
                self.cplex_Instance = c
                
                # solution.get_status() returns an integer code
                if self.options['verbose']>0:
                        print "Solution status = " +str(c.solution.get_status())+":"
                        # the following line prints the corresponding string
                        print(c.solution.status[c.solution.get_status()])
                status = c.solution.status[c.solution.get_status()]
                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                
                #primals
                try:
                        obj = c.solution.get_objective_value()
                        numsol = c.solution.pool.get_num()
                        if numsol>1:
                                objvals=[]
                                for i in range(numsol):
                                        objvals.append((c.solution.pool.get_objective_value(i),i))
                                indsols=[]
                                rev=(self.objective[0]=='max')
                                for ob,ind in sorted(objvals,reverse=rev)[:self.options['pool_size']]:
                                        indsols.append(ind)
                        
                        primals = {}
                        for var in self.variables.values():
                                value = []
                                sz_var = var.size[0]*var.size[1]
                                for i in range(sz_var):
                                        name = var.name + '_' + str(i)
                                        value.append(c.solution.get_values(name))
                                                
                                primals[var.name] = cvx.matrix(value,var.size)
                        
                        if numsol>1:
                                for ii,ind in enumerate(indsols):
                                        for var in self.variables.values():
                                                value = []
                                                sz_var = var.size[0]*var.size[1]
                                                for i in range(sz_var):
                                                        name = var.name + '_' + str(i)
                                                        value.append(c.solution.pool.get_values(ind,name))
                                                
                                                primals[(ii,var.name)] = cvx.matrix(value,var.size)
                except Exception as ex:
                        import warnings
                        warnings.warn('error while retrieving primals')
                        primals = {}
                        obj = None
                        if self.options['verbose']>0:
                                print('##################################')
                                print('WARNING: Primal Solution Not Found')
                                print('##################################')
                        
                #--------------------#
                # retrieve the duals #
                #--------------------#
                #import pdb;pdb.set_trace()
                duals = [] 
                if not(self.isContinuous()) or (
                     'noduals' in self.options and self.options['noduals']):
                        pass
                else:
                        try:
                                pos_cplex = 0 #row position in the cplex linear constraints
                                #basis_status = c.solution.basis.get_col_basis()
                                #>0 and <0
                                pos_conevar = self.numberOfVars+1 #plus 1 for the __noconstant__ variable 
                                seen_bounded_vars = []
                                for k,constr in enumerate(self.constraints):
                                        if constr.typeOfConstraint[:3] == 'lin':
                                                dim = constr.Exp1.size[0] * constr.Exp1.size[1]
                                                dim = dim - len(self.cplex_boundcons[k])
                                                dual_lines = range(pos_cplex, pos_cplex + dim)
                                                if len(dual_lines)==0:
                                                        dual_values = []
                                                else:
                                                        dual_values = c.solution.get_dual_values(dual_lines)
                                                for (i,j,b,v) in self.cplex_boundcons[k]:
                                                        xj = c.solution.get_values(j)
                                                        if ((b=='=') or (xj == b)) and (j not in seen_bounded_vars):
                                                                #does j appear in another equality constraint ?
                                                                if b!='=':
                                                                       boundsj=[b0 for k0 in range(len(self.constraints))
                                                                                       for (i0,j0,b0,v0) in self.cplex_boundcons[k0]
                                                                                       if j0==j]
                                                                       if '=' in boundsj:
                                                                               dual_values.insert(i,0.) #dual will be set later, only for the equality case
                                                                               continue
                                                                else: #equality
                                                                        seen_bounded_vars.append(j)
                                                                        du=c.solution.get_reduced_costs(j)/v
                                                                        dual_values.insert(i,du)
                                                                        continue
                                                                #what kind of inequality ?
                                                                du=c.solution.get_reduced_costs(j)
                                                                if (((v>0 and constr.typeOfConstraint[3]=='<') or
                                                                    (v<0 and constr.typeOfConstraint[3]=='>')) and
                                                                    du>0):#upper bound
                                                                        seen_bounded_vars.append(j)
                                                                        dual_values.insert(i,du/abs(v))
                                                                elif (((v>0 and constr.typeOfConstraint[3]=='>') or
                                                                     (v<0 and constr.typeOfConstraint[3]=='<')) and
                                                                     du<0):#lower bound
                                                                        seen_bounded_vars.append(j)
                                                                        dual_values.insert(i,-du/abs(v))
                                                                else:
                                                                        dual_values.insert(i,0.) #unactive constraint
                                                        else:
                                                                dual_values.insert(i,0.)
                                                pos_cplex += dim
                                                duals.append(cvx.matrix(dual_values))
                                                
                                        elif constr.typeOfConstraint == 'SOcone':
                                                szcons = constr.Exp1.size[0]*constr.Exp1.size[1]
                                                dual_cols = range(pos_conevar,pos_conevar+szcons+1)
                                                dual_values = c.solution.get_reduced_costs(dual_cols)
                                                duals.append(np.sign(dual_values[-1]) * cvx.matrix(
                                                                [dual_values[-1]]+dual_values[:-1]))
                                                pos_conevar += szcons+1
                                        
                                        elif constr.typeOfConstraint == 'RScone':
                                                szcons = constr.Exp1.size[0]*constr.Exp1.size[1]
                                                dual_cols = range(pos_conevar,pos_conevar+szcons+2)
                                                dual_values = c.solution.get_reduced_costs(dual_cols)
                                                duals.append(np.sign(dual_values[-1]) * cvx.matrix(
                                                                [dual_values[-1]]+dual_values[:-1]))
                                                pos_conevar += szcons+2
                                        
                                        else:
                                                if self.options['verbose']>0:
                                                        print 'duals for this type of constraint not supported yet'
                                                duals.append(None)

                        except Exception as ex:
                                import warnings
                                warnings.warn('error while retrieving duals')
                                duals = []
                #-----------------#
                # return statement#
                #-----------------#             
                
                sol = {'cplex_solution':c.solution,'status':status}
                return (primals,duals,obj,sol)
                

        def _mosek_solve(self):
                """
                Solves the problem with mosek
                
                .. Todo:: * Solver settings
                          * Dans dual() et tuto, expliquer pour RS cones
                """
                #----------------------------#
                #  create the mosek instance #
                #----------------------------#
                import mosek
                self._make_mosek_instance()
                task=self.msk_task

                #TODO : test timelimit, automatic optionmosek
                #---------------------#
                #  setting parameters #
                #---------------------# 
                
                #tolerance (conic + LP interior points)
                task.putdouparam(mosek.dparam.intpnt_tol_dfeas,self.options['tol'])
                task.putdouparam(mosek.dparam.intpnt_tol_pfeas,self.options['tol'])
                task.putdouparam(mosek.dparam.intpnt_tol_mu_red,self.options['tol'])
                task.putdouparam(mosek.dparam.intpnt_tol_rel_gap,self.options['tol'])
                
                task.putdouparam(mosek.dparam.intpnt_co_tol_dfeas,self.options['tol'])
                task.putdouparam(mosek.dparam.intpnt_co_tol_pfeas,self.options['tol'])
                task.putdouparam(mosek.dparam.intpnt_co_tol_mu_red,self.options['tol'])
                task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap,self.options['tol'])
                
                #tolerance (interior points)
                task.putdouparam(mosek.dparam.mio_tol_rel_gap,self.options['gaplim'])
                
                
                #maxiters
                task.putintparam(mosek.iparam.intpnt_max_iterations,self.options['maxit'])
                task.putintparam(mosek.iparam.sim_max_iterations,self.options['maxit'])
                
                #lpmethod
                if not self.options['lp_node_method'] is None:
                        if self.options['lp_node_method']=='interior':
                                task.putintparam(mosek.iparam.mio_node_optimizer,mosek.optimizertype.intpnt)
                        elif self.options['lp_node_method']=='psimplex':
                                task.putintparam(mosek.iparam.mio_node_optimizer,mosek.optimizertype.primal_simplex)
                        elif self.options['lp_node_method']=='dsimplex':
                                task.putintparam(mosek.iparam.mio_node_optimizer,mosek.optimizertype.dual_simplex)
                        else:
                                raise Exception('unexpected value for option lp_node_method')
                if not self.options['lp_root_method'] is None:
                        if self.options['lp_root_method']=='interior':
                                task.putintparam(mosek.iparam.mio_root_optimizer,mosek.optimizertype.intpnt)
                                if self.type=='LP':
                                        task.putintparam(mosek.iparam.optimizer,mosek.optimizertype.intpnt)
                        elif self.options['lp_root_method']=='psimplex':
                                task.putintparam(mosek.iparam.mio_root_optimizer,mosek.optimizertype.primal_simplex)
                                if self.type=='LP':
                                        task.putintparam(mosek.iparam.optimizer,mosek.optimizertype.primal_simplex)
                        elif self.options['lp_root_method']=='dsimplex':
                                task.putintparam(mosek.iparam.mio_root_optimizer,mosek.optimizertype.dual_simplex)
                                if self.type=='LP':
                                        task.putintparam(mosek.iparam.optimizer,mosek.optimizertype.dual_simplex)
                        else:
                                raise Exception('unexpected value for option lp_root_method')
                
                if not self.options['timelimit'] is None:
                        task.putdouparam(mosek.dparam.mio_max_time,self.options['timelimit'])
                        task.putdouparam(mosek.dparam.optimizer_max_time,self.options['timelimit'])
                        #task.putdouparam(mosek.dparam.mio_max_time_aprx_opt,self.options['timelimit'])
                else:
                        task.putdouparam(mosek.dparam.mio_max_time,-1.0)
                        task.putdouparam(mosek.dparam.optimizer_max_time,-1.0)
                        #task.putdouparam(mosek.dparam.mio_max_time_aprx_opt,-1.0)
                
                if not self.options['nbsol'] is None:
                        task.putintparam(mosek.iparam.mio_max_num_solutions,self.options['nbsol'])
                #--------------------#
                #  call the solver   #
                #--------------------# 
                #optimize
                try:
                        task.optimize()
                except mosek.Error as ex:
                        #catch non-convexity exception
                        if str(ex)=='(0) ' and self.numberQuadConstraints>0:
                                raise Exception('Error raised during solve. Problem nonconvex ?')
                        if str(ex).startswith('(1550)') and (
                           self.type in ('Mixed (SOCP+quad)','Mixed (MISOCP+quad)')):
                                raise Exception('Please convert the problem as an socp '+
                                                'with the function convert_quad_to_socp().')
                        else:
                                import pdb;pdb.set_trace()
                                print "Error raised during solve"
                                
                                
                
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)
                prosta = []
                solsta = []

                if self.is_continuous():
                        if not(self.options['lp_root_method'] is None) and (
                          self.options['lp_root_method'].endswith('simplex')):
                                soltype=mosek.soltype.bas
                        else:
                                soltype=mosek.soltype.itr
                        intg=False
                else:
                        soltype=mosek.soltype.itg
                        intg=True

                [prosta,solsta] = task.getsolutionstatus(soltype)
                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                
                #PRIMAL VARIABLES
                primals={}
                if self.options['verbose']>0:
                        print 'Solution status is ' +repr(solsta)
                status = repr(solsta)
                try:
                        # Output a solution
                        xx = np.zeros(self.numberOfVars, float)
                        task.getsolutionslice(soltype,
                                mosek.solitem.xx, 0,self.numberOfVars, xx)
                        obj = task.getprimalobj(soltype)
                        for var in self.variables.keys():
                                si=self.variables[var].startIndex
                                ei=self.variables[var].endIndex
                                varvect=xx[si:ei]
                                primals[var]=cvx.matrix(varvect, self.variables[var].size)
                except Exception as ex:
                        primals={}
                        obj=None
                        import warnings
                        warnings.warn('error while retrieving the primals')
                        if self.options['verbose']>0:
                                print('##################################')
                                print('WARNING: Primal Solution Not Found')
                                print('##################################')

                #--------------------#
                # retrieve the duals #
                #--------------------#
                duals=[]
                if intg or ('noduals' in self.options and self.options['noduals']):
                        pass
                else:
                        try:
                                idvarcone=self.numberOfVars #index of variables in cone
                                ideq=0 #index of equality constraint in cvxoptVars['A']
                                idconeq=0 #index of equality constraint in mosekcons (without fixed vars)
                                idin=0 #index of inequality constraint in cvxoptVars['Gl']
                                idconin=len([1 for ida in range(self.cvxoptVars['A'].size[0])
                                                if len(self.cvxoptVars['A'][ida,:].J)>1])
                                        #index of inequality constraint in mosekcons (without fixed vars)
                                idcone=0 #number of seen cones
                                Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                                
                                #indices for ineq constraints
                                ijvs=sorted(zip(Gli,Glj,Glv),reverse=True)
                                if len(ijvs)>0:
                                        (ik,jk,vk)=ijvs.pop()
                                        curik=-1
                                        delNext=False
                                del Gli,Glj,Glv
                                
                                #indices for eq constraints
                                Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                                ijls=sorted(zip(Ai,Aj,Av),reverse=True)
                                if len(ijls)>0:
                                        (il,jl,vl)=ijls.pop()
                                        curil=-1
                                        delNext=False
                                del Ai,Aj,Av
                                
                                seen_bounded_vars = []
                                
                                #now we parse the constraints
                                for k in range(len(self.constraints)):
                                        #conic constraint
                                        if self.constraints[k].typeOfConstraint[2:]=='cone':
                                                szcone=self.cvxoptVars['Gq'][idcone].size[0]
                                                v=np.zeros(szcone,float)
                                                task.getsolutionslice(soltype,mosek.solitem.snx,
                                                                idvarcone,idvarcone+szcone,v)
                                                duals.append(cvx.matrix(v))
                                                idvarcone+=szcone
                                                idcone+=1
                                        elif self.constraints[k].typeOfConstraint=='lin=':
                                                szcons=int(np.product(self.constraints[k].Exp1.size))
                                                fxd=[]
                                                while il<ideq+szcons:
                                                        if il!=curil:
                                                                fxd.append((il-ideq,jl,vl))
                                                                curil=il
                                                                delNext=True
                                                        elif delNext:
                                                                del fxd[-1]
                                                                delNext=False
                                                        try:
                                                                (il,jl,vl)=ijls.pop()
                                                        except IndexError:
                                                                break
                                                
                                                v=np.zeros(szcons-len(fxd),float)
                                                if len(v)>0:
                                                        task.getsolutionslice(soltype,mosek.solitem.y,
                                                                idconeq,idconeq+szcons-len(fxd),v)
                                                v=v.tolist()
                                                for (l,var,coef) in fxd: #dual of fixed var constraints
                                                        duu=np.zeros(1,float)
                                                        dul=np.zeros(1,float)
                                                        task.getsolutionslice(soltype,mosek.solitem.sux,
                                                                                var,var+1,duu)
                                                        task.getsolutionslice(soltype,mosek.solitem.slx,
                                                                                var,var+1,dul)
                                                        if (var not in seen_bounded_vars):
                                                                v.insert(l,(dul[0]-duu[0])/coef)
                                                                seen_bounded_vars.append(var)
                                                        else:
                                                                v.insert(l,0.)
                                                duals.append(cvx.matrix(v))
                                                ideq+=szcons
                                                idconeq+=(szcons-len(fxd))
                                        elif self.constraints[k].typeOfConstraint[:3]=='lin':#inequality
                                                szcons=int(np.product(self.constraints[k].Exp1.size))
                                                fxd=[]
                                                while ik<idin+szcons:
                                                        if ik!=curik:
                                                                fxd.append((ik-idin,jk,vk))
                                                                curik=ik
                                                                delNext=True
                                                        elif delNext:
                                                                del fxd[-1]
                                                                delNext=False
                                                        try:
                                                                (ik,jk,vk)=ijvs.pop()
                                                        except IndexError:
                                                                break
                                                                                        
                                                #for k in range(szcons):
                                                        #if len(self.cvxoptVars['Gl'][idin+k,:].J)==1:
                                                                #fxd.append((k,
                                                                        #self.cvxoptVars['Gl'][idin+k,:].J[0],
                                                                        #self.cvxoptVars['Gl'][idin+k,:].V[0]))
                                                v=np.zeros(szcons-len(fxd),float)
                                                if len(v)>0:
                                                        task.getsolutionslice(soltype,mosek.solitem.y,
                                                                idconin,idconin+szcons-len(fxd),v)
                                                v=v.tolist()
                                                for (l,var,coef) in fxd: #dual of simple bound constraints
                                                        du=np.zeros(1,float)
                                                        bound=self.cvxoptVars['hl'][idin+l]/coef
                                                        bk,bl,bu=task.getbound(mosek.accmode.var,var)
                                                        duu=np.zeros(1,float)
                                                        dul=np.zeros(1,float)
                                                        task.getsolutionslice(soltype,mosek.solitem.sux,
                                                                                var,var+1,duu)
                                                        task.getsolutionslice(soltype,mosek.solitem.slx,
                                                                                var,var+1,dul)
                                                        
                                                        if coef>0: #upper bound
                                                                if bound==bu and (var not in seen_bounded_vars) and(
                                                                    abs(duu[0])>1e-8
                                                                    and abs(dul[0])<1e-5
                                                                    and abs(duu[0])>abs(dul[0])): #active bound:
                                                                        v.insert(l,-duu[0]/coef)
                                                                        seen_bounded_vars.append(var)
                                                                else:
                                                                        v.insert(l,0.) #inactive bound, or active already seen
                                                        else:   #lower bound
                                                                if bound==bl and (var not in seen_bounded_vars) and(
                                                                    abs(dul[0])>1e-8
                                                                    and abs(duu[0])<1e-5
                                                                    and abs(dul[0])>abs(duu[0])): #active bound
                                                                        v.insert(l,dul[0]/coef)
                                                                        seen_bounded_vars.append(var)
                                                                else:
                                                                        v.insert(l,0.) #inactive bound, or active already seen
                                                duals.append(cvx.matrix(v))
                                                idin+=szcons
                                                idconin+=(szcons-len(fxd))
                                        else:
                                                        if self.options['verbose']>0:
                                                                print('dual for this constraint is not handled yet')
                                                        duals.append(None)
                        except Exception as ex:
                                import warnings
                                warnings.warn('error while retrieving the duals')
                                duals = []
                #-----------------#
                # return statement#
                #-----------------#  
                #OBJECTIVE
                sol = {'mosek_task':task,'status':status}
                
                #delete the patch variable for quad prog with 1 var
                if '_ptch_' in self.variables:
                        self.remove_variable('_ptch_')
                        del primals['_ptch_']
                
                return (primals,duals,obj,sol)                
                
        def _zibopt_solve(self):
                """
                Solves the problem with the zib optimization suite
                
                .. Todo:: * solver parameters
                """
                #-----------------------------#
                #  create the zibopt instance #
                #-----------------------------#
                if self.options['onlyChangeObjective']:
                        if self.scip_solver is None:
                                raise Exception('option is only available when scip_solver has been defined before')
                        #define scip_obj
                        newobj=self.objective[1]
                        x=self.scip_vars
                        ob=0
                        
                        if isinstance(newobj,QuadExp):
                                for i,j in newobj.quad:
                                        si,ei=i.startIndex,i.endIndex
                                        sj,ej=j.startIndex,j.endIndex
                                        Qij=newobj.quad[i,j]
                                        if not isinstance(Qij,cvx.spmatrix):
                                                Qij=cvx.sparse(Qij)
                                        for ii,jj,vv in zip(Qij.I,Qij.J,Qij.V):
                                                ob+=vv*x[ii+si]*x[jj+sj]
                                newobj=newobj.aff
                                        
                        if not(newobj is None):
                                for v,fac in newobj.factors.iteritems():
                                        if not isinstance(fac,cvx.spmatrix):
                                                fac=cvx.sparse(fac)
                                        sv=v.startIndex
                                        for jj,vv in zip(fac.J,fac.V):
                                                ob+=vv*x[jj+sv]
                                if not(newobj.constant is None):
                                        ob+=newobj.constant[0]
                        self.scip_obj = ob
                        
                else:
                        self._make_zibopt()
                
                timelimit=10000000.
                gaplim=0.
                nbsol=-1
                if not self.options['timelimit'] is None:
                        timelimit=self.options['timelimit']
                if not self.options['gaplim'] is None:        
                        gaplim=self.options['gaplim']
                if not self.options['nbsol'] is None:
                        nbsol=self.options['nbsol']
                
                #--------------------#
                #  call the solver   #
                #--------------------#
                
                if self.objective[0]=='max':
                        if self.scip_obj is None:
                                sol=self.scip_solver.maximize(time=timelimit,
                                                        gap=gaplim,
                                                        nsol=nbsol)
                        else:#quadratic obj
                                sol=self.scip_solver.maximize(time=timelimit,
                                                        gap=gaplim,
                                                        nsol=nbsol,
                                                        objective=self.scip_obj)
                else:
                        if self.scip_obj is None:
                                sol=self.scip_solver.minimize(time=timelimit,
                                                        gap=gaplim,
                                                        nsol=nbsol)
                        else:#quadratic obj
                                sol=self.scip_solver.minimize(time=timelimit,
                                                        gap=gaplim,
                                                        nsol=nbsol,
                                                        objective=self.scip_obj)
                if sol.optimal:
                        status='optimal'
                elif sol.infeasible:
                        status='infeasible'
                elif sol.unbounded:
                        status='unbounded'
                elif sol.inforunbd:
                        status='infeasible or unbounded'
                else:
                        status='unknown'
                        
                if self.options['verbose']>0:
                        print 'zibopt solution status: '+status
                
                #----------------------#
                # retrieve the primals #
                #----------------------#
                try:
                        obj=sol.objective
                        val=sol.values()
                        primals={}
                        for var in self.variables.keys():
                                si=self.variables[var].startIndex
                                ei=self.variables[var].endIndex
                                varvect=self.scip_vars[si:ei]
                                primals[var]=cvx.matrix([val[v] for v in varvect],
                                        self.variables[var].size)
                except Exception as ex:
                        primals={}
                        obj = None
                        import warnings
                        warnings.warn('error while retrieving the primals')
                        if self.options['verbose']>0:
                                print('##################################')
                                print('WARNING: Primal Solution Not Found')
                                print('##################################')

                #----------------------#
                # retrieve the duals #
                #----------------------#
                
                #not available by python-zibopt (yet ? )
                duals = []
                #------------------#
                # return statement #
                #------------------#  
                
                solt={}
                solt['zibopt_sol']=sol
                solt['status']=status
                return (primals,duals,obj,solt)
                
                
        def _sqpsolve(self,options):
                """
                Solves the problem by sequential Quadratic Programming.
                """
                import copy
                for v in self.variables:
                        if self.variables[v].value is None:
                                self.set_var_value(v,cvx.uniform(self.variables[v].size))
                #lower the display level for mosek                
                self.options['verbose']-=1
                oldvar=self._eval_all()
                subprob=copy.deepcopy(self)
                if self.options['verbose']>0:
                        print('solve by SQP method with proximal convexity enforcement')
                        print('it:     crit\t\tproxF\tstep')
                        print('---------------------------------------')
                converged=False
                k=1
                while not converged:
                        obj,grad,hess=self.objective[1].fun(self.objective[1].Exp.eval())
                        diffExp=self.objective[1].Exp-self.objective[1].Exp.eval()
                        quadobj0=obj+grad.T*diffExp+0.5*diffExp.T*hess*diffExp
                        proxF=self.options['step_sqp']
                        #VARIANT IN CONSTRAINTS: DO NOT FORCE CONVEXITY...
                        #for v in subprob.variables.keys():
                        #        x=subprob.get_varExp(v)
                        #        x0=self.get_variable(v).eval()
                        #        subprob.add_constraint((x-x0).T*(x-x0)<0.5)
                        solFound=False
                        while (not solFound):
                                if self.objective[0]=='max':
                                        quadobj=quadobj0-proxF*abs(diffExp)**2 #(proximal + force convexity)
                                else:
                                        quadobj=quadobj0+proxF*abs(diffExp)**2 #(proximal + force convexity)
                                subprob=copy.deepcopy(self)                                
                                subprob.set_objective(self.objective[0],quadobj)
                                if self.options['harmonic_steps'] and k>1:
                                        for v in subprob.variables.keys():
                                                x=subprob.get_varExp(v)
                                                x0=self.get_variable(v).eval()
                                                subprob.add_constraint((x-x0).T*(x-x0)<(10./float(k-1)))
                                try:
                                        sol=subprob.solve()
                                        solFound=True
                                except Exception as ex:
                                        if str(ex)[:6]=='(1296)': #function not convex
                                                proxF*=(1+cvx.uniform(1))
                                        else:
                                                #reinit the initial verbosity
                                                self.options['verbose']+=1
                                                raise
                        if proxF>=100*self.options['step_sqp']:
                                #reinit the initial verbosity
                                self.options['verbose']+=1
                                raise Exception('function not convex before proxF reached 100 times the initial value')

                        for v in subprob.variables:
                                self.set_var_value(v,subprob.get_valued_variable(v))
                        newvar=self._eval_all()
                        step=np.linalg.norm(newvar-oldvar)
                        if isinstance(step,cvx.matrix):
                                step=step[0]
                        oldvar=newvar
                        if self.options['verbose']>0:
                                if k==1:
                                        print('  {0}:         --- \t{1:6.3f} {2:10.4e}'.format(k,proxF,step))
                                else:
                                        print('  {0}:   {1:16.9e} {2:6.3f} {3:10.4e}'.format(k,obj,proxF,step))
                        k+=1
                        #have we converged ?
                        if step<self.options['tol']:
                                converged=True
                        if k>self.options['maxit']:
                                converged=True
                                print('Warning: no convergence after {0} iterations'.format(k))

                #reinit the initial verbosity
                self.options['verbose']+=1
                sol['lastStep']=step
                return sol
                
        def what_type(self):
                
                iv= [v for v in self.variables.values() if v.vtype not in ('continuous','symmetric') ]
                #continuous problem
                if len(iv)==0:
                        #general convex
                        if not(self.objective[1] is None) and isinstance(self.objective[1],GeneralFun):
                                return 'general-obj'
                        #GP
                        if self.numberLSEConstraints>0:
                                if (self.numberConeConstraints ==0
                                and self.numberQuadConstraints == 0
                                and self.numberSDPConstraints == 0):
                                        return 'GP'
                                else:
                                        return 'unknown type'
                        #SDP
                        if self.numberSDPConstraints>0:
                                if (self.numberConeConstraints ==0
                                and self.numberQuadConstraints == 0):
                                        return 'SDP'
                                elif self.numberQuadConstraints == 0:
                                        return 'ConeP'
                                else:
                                        return 'Mixed (SDP+quad)'
                        #SOCP
                        if self.numberConeConstraints > 0:
                                if self.numberQuadConstraints == 0:
                                        return 'SOCP'
                                else:
                                        return 'Mixed (SOCP+quad)'

                        #quadratic problem
                        if self.numberQuadConstraints>0:
                                if any([cs.typeOfConstraint=='quad' for cs in self.constraints]):
                                        return 'QCQP'
                                else:
                                        return 'QP'
                                
                        return 'LP'
                else:
                        if not(self.objective[1] is None) and isinstance(self.objective[1],GeneralFun):
                                return 'unknown type'
                        if self.numberLSEConstraints>0:
                                return 'unknown type'
                        if self.numberSDPConstraints>0:
                                return 'unknown type'
                        if self.numberConeConstraints > 0:
                                if self.numberQuadConstraints == 0:
                                        return 'MISOCP'
                                else:
                                        return 'Mixed (MISOCP+quad)'
                        if self.numberQuadConstraints>0:
                                if any([cs.typeOfConstraint=='quad' for cs in self.constraints]):
                                        return 'MIQCP'
                                else:
                                        return 'MIQP'
                        return 'MIP' #(or simply IP)
                        
                        
        def set_type(self,value):
                raise AttributeError('type is not writable')
        
        def del_type(self):
                raise AttributeError('type is not writable')
        
        type=property(what_type,set_type,del_type)
        """Type of Optimization Problem ('LP', 'MIP', 'SOCP', 'QCQP',...)"""
        
        
        def solver_selection(self):
                """Selects an appropriate solver for this problem
                and sets the option ``'solver'``.
                """
                tp=self.type
                if tp == 'LP':
                        order=['cplex','mosek','zibopt','cvxopt','smcp']
                elif tp in ('QCQP,QP'):
                        order=['cplex','mosek','zibopt']
                elif tp == 'SOCP':
                        order=['mosek','cplex','cvxopt','smcp']
                elif tp == 'SDP':
                        order=['cvxopt','smcp']
                elif tp == 'ConeP':
                        order=['cvxopt','smcp']
                elif tp == 'GP':
                        order=['cvxopt']
                elif tp == 'general-obj':
                        order=['cplex','mosek','zibopt','cvxopt']
                elif tp in ('MIP','MIQCP','MISOCP','MIQP'):
                        order=['cplex','mosek','zibopt']
                elif tp in ('Mixed (SOCP+quad)','Mixed (MISOCP+quad)'):
                        order=['cplex']
                else:
                        raise Exception('no solver available for problem of type {0}'.format(tp))
                avs=available_solvers()
                for sol in order:
                        if sol in avs:
                                self.set_option('solver',sol)
                                return
                #mosek can solve this as an socp
                if tp=='Mixed (MISOCP+quad)' and 'mosek' in avs:
                        raise Exception('You should first transform the problem into an socp '+
                                        'with the function convert_quad_to_socp() '+
                                        'and then solve the problem with mosek.')
                        
                #cvxopt can solve this as an socp
                if tp in ('Mixed (SOCP+quad)','Mixed (SDP+quad)','QCQP','QP'):
                        raise Exception('No solver available in this form. '+
                                        'You should first transform the problem into an socp '+
                                        'with the function convert_quad_to_socp().')
                #not found
                raise Exception('no solver available for problem of type {0}'.format(tp))
                
                
        def write_to_file(self,filename,writer='picos'):
                """
                This function writes the problem to a file.
                
                :param filename: The name of the file where the problem will be saved. The
                                 extension of the file (if provided) indicates the format
                                 of the export:
                                 
                                        * ``'.lp'``: `LP format <http://docs.mosek.com/6.0/pyapi/node022.html>`_
                                          . This format handles only linear constraints, unless the writer ``'cplex'``
                                          is used, and the file is saved in the extended
                                          `cplex LP format <http://pic.dhe.ibm.com/infocenter/cplexzos/v12r4/index.jsp?topic=%2Fcom.ibm.cplex.zos.help%2Fhomepages%2Freffileformatscplex.html>`_
                                          
                                        * ``'.mps'``: `MPS format <http://docs.mosek.com/6.0/pyapi/node021.html>`_
                                          (recquires mosek or cplex).
                                          
                                        * ``'.opf'``: `OPF format <http://docs.mosek.com/6.0/pyapi/node023.html>`_
                                          (recquires mosek).

                                        * ``'.dat-s'``: `sparse SDPA format <http://sdpa.indsys.chuo-u.ac.jp/sdpa/download.html#sdpa>`_
                                          This format is suitable to save semidefinite programs (SDP).
                                        
                :type filename: str.
                :param writer: The default writer is ``picos``, which has its own *LP* and
                               *sparse SDPA* write functions. If cplex or mosek is installed,
                               the user can pass the option ``writer=cplex`` or
                               ``writer=mosek``, and the write function of this solver
                               will be used.                               
                :type writer: str.
                """
                if self.numberLSEConstraints:
                        raise Exception('gp are not supported')
                if not(self.objective[1] is None) and isinstance(self.objective[1],GeneralFun):
                        raise Exception('general-obj are not supported')       
                
                #automatic extension recognition
                if not(filename[-4:] in ('.mps','.opf') or
                       filename[-3:]=='.lp' or
                       filename[-6:]=='.dat-s' or
                       filename[-7:]=='.dat-sx'):
                        if writer=='mosek':
                                if (self.numberSDPConstraints >0):
                                        raise Exception('no sdp with mosek')
                                if (self.numberConeConstraints + 
                                    self.numberQuadConstraints) ==0:
                                        filename+='.lp'
                                else:
                                        filename+='.mps'
                        elif writer=='cplex':
                                if (self.numberSDPConstraints >0):
                                        raise Exception('no sdp with cplex')
                                else:
                                        filename+='.lp'
                        elif writer=='picos':
                                if (self.numberQuadConstraints >0):
                                        raise Exception('no quad constraints in sdpa format.'+
                                                ' Try to convert to socp with the function convert_quad_to_socp().')
                                if (self.numberConeConstraints + 
                                    self.numberSDPConstraints) ==0:
                                        filename+='.lp'
                                else:
                                        filename+='.dat-s'
                        else:
                                raise Exception('unexpected writer')

                #writer selection [obsolete, since we now give picos as default]
                if writer is None:
                        avs=available_solvers()
                        if filename[-4:]=='.mps':
                                if 'mosek' in avs:
                                        writer='mosek'
                                else:
                                        raise Exception('no mps writer available')
                        elif filename[-4:]=='.opf':
                                if 'mosek' in avs:
                                        writer='mosek'
                                else:
                                        raise Exception('no opf writer available')
                        elif filename[-3:]=='.lp':
                                if not(self.cplex_Instance is None):
                                        writer='cplex'
                                elif not(self.msk_task is None) and (self.numberConeConstraints + 
                                                                self.numberQuadConstraints) ==0:
                                        writer='mosek'
                                elif 'cplex' in avs:
                                        writer='cplex'
                                elif 'mosek' in avs and (self.numberConeConstraints + 
                                                                self.numberQuadConstraints) ==0:
                                        writer='mosek'
                                else:
                                        writer='picos'
                        elif filename[-6:]=='.dat-s':
                                writer='picos'
                        elif filename[-7:]=='.dat-sx':
                                writer='picos'
                        else:
                                raise Exception('unexpected file extension')
                
                  
                if writer == 'cplex':
                        if self.cplex_Instance is None:
                                self._make_cplex_instance()
                        self.cplex_Instance.write(filename)
                elif writer == 'mosek':
                        if self.msk_task is None:
                                self._make_mosek_instance()
                        self.msk_task.writedata(filename)
                elif writer == 'picos':
                        if filename[-3:]=='.lp':
                                self._write_lp(filename)
                        elif filename[-6:]=='.dat-s':
                                self._write_sdpa(filename)
                        elif filename[-7:]=='.dat-sx':
                                self._write_sdpa(filename,True)
                        else:
                                raise Exception('unexpected file extension')
                else:
                        raise Exception('unknown writer')
        

        def _write_lp(self,filename):
                """
                writes problem in  lp format
                """
                #add extension
                if filename[-3:]!='.lp':
                        filename+='.lp'
                #check lp compatibility
                if (self.numberConeConstraints + 
                    self.numberQuadConstraints +
                    self.numberLSEConstraints  +
                    self.numberSDPConstraints) > 0:
                        raise Exception('the picos LP writer only accepts (MI)LP')
                #open file
                f = open(filename,'w')
                f.write("\\* file "+filename+" generated by picos*\\\n")
                #cvxoptVars
                if not any(self.cvxoptVars.values()):
                        self._make_cvxopt_instance()
                #variable names
                varnames={}
                for name,v in self.variables.iteritems():
                        j=0
                        k=0
                        for i in xrange(v.startIndex,v.endIndex):
                                if v.size==(1,1):
                                        varnames[i]=name
                                elif v.size[1]==1:
                                        varnames[i]=name+'('+str(j)+')'
                                        j+=1
                                else:
                                        varnames[i]=name+'('+str(j)+','+str(k)+')'
                                        j+=1
                                        if j==v.size[0]:
                                                k+=1
                                                j=0
                                varnames[i]=varnames[i].replace('[','(')
                                varnames[i]=varnames[i].replace(']',')')
                #affexpr writer
                def affexp_writer(name,indices,coefs):
                        s=''
                        s+=name
                        s+=' : '
                        start=True
                        for (i,v) in zip(indices,coefs):
                                if v>0 and not(start):
                                        s+='+ '
                                s+="%.12g" % v
                                s+=' '
                                s+=varnames[i]
                                #not the first term anymore
                                start = False
                        if not(indices):
                                s+='0.0 '
                                s+=varnames[0]
                        return s
                
                print 'writing problem in '+filename+'...'
                
                #objective
                if self.objective[0]=='max':
                        f.write("Maximize\n")
                        #max handled directly
                        self.cvxoptVars['c']=-self.cvxoptVars['c']
                else:
                        f.write("Minimize\n")
                I=cvx.sparse(self.cvxoptVars['c']).I
                V=cvx.sparse(self.cvxoptVars['c']).V
                f.write(affexp_writer('obj',I,V))
                f.write('\n')
                
                f.write("Subject To\n")
                bounds={}
                #equality constraints:
                Ai,Aj,Av=( self.cvxoptVars['A'].I,self.cvxoptVars['A'].J,self.cvxoptVars['A'].V)
                ijvs=sorted(zip(Ai,Aj,Av))
                del Ai,Aj,Av
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                ieq=0
                for i,jv in itojv.iteritems():
                        J=[jvk[0] for jvk in jv]
                        V=[jvk[1] for jvk in jv]
                        if len(J)==1:
                                #fixed variable
                                b=self.cvxoptVars['b'][i]/V[0]
                                bounds[J[0]]=(b,b)
                        else:
                                #affine equality
                                b=self.cvxoptVars['b'][i]
                                f.write(affexp_writer('eq'+str(ieq),J,V))
                                f.write(' = ')
                                f.write("%.12g" % b)
                                f.write('\n')
                                ieq+=1
                
                
                #inequality constraints:
                Gli,Glj,Glv=( self.cvxoptVars['Gl'].I,self.cvxoptVars['Gl'].J,self.cvxoptVars['Gl'].V)
                ijvs=sorted(zip(Gli,Glj,Glv))
                del Gli,Glj,Glv
                itojv={}
                lasti=-1
                for (i,j,v) in ijvs:
                        if i==lasti:
                                itojv[i].append((j,v))
                        else:
                                lasti=i
                                itojv[i]=[(j,v)]
                iaff=0
                for i,jv in itojv.iteritems():
                        J=[jvk[0] for jvk in jv]
                        V=[jvk[1] for jvk in jv]
                        if len(J)==1 and (not (i in [t[1] for t in self.cvxoptVars['quadcons']])):
                                #bounded variable
                                if J[0] in bounds:
                                        bl,bu=bounds[J[0]]
                                else:
                                        bl,bu=-INFINITY,INFINITY
                                b=self.cvxoptVars['hl'][i]/V[0]
                                if V[0]>0:
                                        #less than
                                        bu=min(b,bu)
                                if V[0]<0:
                                        #greater than
                                        bl=max(b,bl)
                                bounds[J[0]]=(bl,bu)
                        else:
                                #affine inequality
                                b=self.cvxoptVars['hl'][i]
                                f.write(affexp_writer('in'+str(iaff),J,V))
                                f.write(' <= ')
                                f.write("%.12g" % b)
                                f.write('\n')
                                iaff+=1

                #bounds
                f.write("Bounds\n")
                for i in xrange(self.numberOfVars):
                        if i in bounds:
                                bl,bu=bounds[i]
                        else:
                                bl,bu=-INFINITY,INFINITY
                        if bl == -INFINITY and bu == INFINITY:
                                f.write(varnames[i]+' free')
                        elif bl == bu:
                                f.write(varnames[i]+(" = %.12g" % bl))
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

                #general integers
                f.write("Generals\n")
                for name,v in self.variables.iteritems():
                        if v.vtype=='integer':
                                for i in xrange(v.startIndex,v.endIndex):
                                        f.write(varnames[i]+'\n')
                        if v.vtype=='semiint' or v.vtype=='semicont':
                                raise Exception('semiint and semicont variables not handled by this LP writer')
                #binary variables
                f.write("Binaries\n")
                for name,v in self.variables.iteritems():
                        if v.vtype=='binary':
                                for i in xrange(v.startIndex,v.endIndex):
                                        f.write(varnames[i]+'\n')
                print 'done.'
                f.close()

        
        def _write_sdpa(self,filename,extended=False):
                """
                write a problem to sdpa format
                """
                
                #--------------------#
                # makes the instance #
                #--------------------#
                if not any(self.cvxoptVars.values()):
                        self._make_cvxopt_instance()
                
                
                dims={}
                dims['s']=[int(np.sqrt(Gsi.size[0])) for Gsi in self.cvxoptVars['Gs']]
                dims['l']=self.cvxoptVars['Gl'].size[0]
                dims['q']=[Gqi.size[0] for Gqi in self.cvxoptVars['Gq']]
                G=self.cvxoptVars['Gl']
                h=self.cvxoptVars['hl']
                
                # handle the equalities as 2 ineq for smcp
                if self.cvxoptVars['A'].size[0]>0:
                        G=cvx.sparse([G,self.cvxoptVars['A']]) 
                        G=cvx.sparse([G,-self.cvxoptVars['A']])
                        h=cvx.matrix([h,self.cvxoptVars['b']])
                        h=cvx.matrix([h,-self.cvxoptVars['b']])
                        dims['l']+=(2*self.cvxoptVars['A'].size[0])

                for i in range(len(dims['q'])):
                        G=cvx.sparse([G,self.cvxoptVars['Gq'][i]])
                        h=cvx.matrix([h,self.cvxoptVars['hq'][i]])

                                        
                for i in range(len(dims['s'])):
                        G=cvx.sparse([G,self.cvxoptVars['Gs'][i]])
                        h=cvx.matrix([h,self.cvxoptVars['hs'][i]])

                #Remove the lines in A and b corresponding to 0==0        
                JP=list(set(self.cvxoptVars['A'].I))
                IP=range(len(JP))
                VP=[1]*len(JP)
                
                idx_0eq0 = [i for i in range(self.cvxoptVars['A'].size[0]) if i not in JP]
                
                #is there a constraint of the form 0==a(a not 0) ?
                if any([b for (i,b) in enumerate(self.cvxoptVars['b']) if i not in JP]):
                        raise Exception('infeasible constraint of the form 0=a')
                
                P=cvx.spmatrix(VP,IP,JP,(len(IP),self.cvxoptVars['A'].size[0]))
                self.cvxoptVars['A']=P*self.cvxoptVars['A']
                self.cvxoptVars['b']=P*self.cvxoptVars['b']
                c = self.cvxoptVars['c']
                
                #-----------------------------------------------------------#
                # make A,B,and blockstruct.                                 #
                # This code is a modification of the conelp function in smcp#
                #-----------------------------------------------------------#
                from cvxopt import matrix,sparse,spdiag,spmatrix
                
                Nl = dims['l']
                Nq = dims['q']
                Ns = dims['s']
                if not Nl: Nl = 0

                nblocks = Nl + len(Nq) + len(Ns)

                P_n = Nl+sum(Nq)+sum(Ns)
                P_m = G.size[1]

                P_A = {}
                P_b = -c
                P_blockstruct = []
                if Nl: P_blockstruct.append(-Nl)
                if extended:
                        for i in Nq: P_blockstruct.append(i*1j)
                else:
                        for i in Nq: P_blockstruct.append(i)
                for i in Ns: P_blockstruct.append(i)

                def tril(X): #lower triangular part
                        I=[]
                        J=[]
                        V=[]
                        for i,j,v in zip(X.I,X.J,X.V):
                                if j<=i:
                                        I.append(i)
                                        J.append(j)
                                        V.append(v)
                        return cvx.spmatrix(V,I,J,X.size)
                                                
                def ind2sub(n,ind): #transform index in col major order into
                                    #a pair of matrix indices
                        I=[]
                        J=[]
                        for i in ind:
                                I.append(i%n)
                                J.append(i//n)
                        return I,J
                        
                
                
                for k in range(P_m+1):
                        if not k==0:
                                v = sparse(G[:,k-1])
                        else:
                                v = +sparse(h)
                        B = []

                        ptr = 0
                        # lin. constraints
                        if Nl:
                                u = v[:Nl]
                                I = u.I
                                B.append(spmatrix(u.V,I,I,(Nl,Nl)))
                                ptr += Nl

                        # SOC constraints
                        for i in xrange(len(Nq)):
                                nq = Nq[i]
                                u0 = v[ptr]
                                u1 = v[ptr+1:ptr+nq]
                                tmp = spmatrix(u1.V,[nq-1 for j in xrange(len(u1))],u1.I,(nq,nq))
                                if not u0 == 0.0:
                                        tmp += spmatrix(u0,xrange(nq),xrange(nq),(nq,nq)) 
                                B.append(tmp)
                                ptr += Nq[i]

                        # SDP constraints
                        for i in xrange(len(Ns)):
                                ns = Ns[i]
                                u = v[ptr:ptr+ns**2]
                                I,J = ind2sub(ns,u.I)
                                tmp = tril(spmatrix(u.V,I,J,(ns,ns)))
                                B.append(tmp)
                                ptr += ns**2

                        #Ai = spdiag(B)
                        #P_A[:,k] = Ai[:]
                        P_A[k]=B

                
                
                #write data
                                
                #add extension
                if extended:
                        if filename[-7:]!='.dat-sx':
                                filename+='.dat-sx'
                else:
                        if filename[-6:]!='.dat-s':
                                filename+='.dat-s'
                #check lp compatibility
                if (self.numberQuadConstraints +
                    self.numberLSEConstraints) > 0:
                        raise Exception('Problem should not have quad or gp constraints. '+
                         'Try to convert the problem to an SOCP with the function convert_quad_to_socp()')
                #open file
                f = open(filename,'w')
                f.write('"file '+filename+' generated by picos"\n')
                print 'writing problem in '+filename+'...'
                f.write(str(self.numberOfVars)+' = number of vars\n')
                f.write(str(len(P_blockstruct))+' = number of blocs\n')
                #bloc structure
                f.write(str(P_blockstruct).replace('[','(').replace(']',')'))
                f.write(' = BlocStructure\n')
                #c vector (objective)
                f.write(str(list(-P_b)).replace('[','{').replace(']','}'))
                f.write('\n')
                #coefs
                from itertools import izip
                for k,Ak in P_A.iteritems():
                        for b,B in enumerate(Ak):
                                for i,j,v in izip(B.I,B.J,B.V):
                                        f.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                                                  k,b+1,i+1,j+1,-v))
                
                #binaries an integers in extended format
                if extended:
                        #general integers
                        f.write("Generals\n")
                        for name,v in self.variables.iteritems():
                                if v.vtype=='integer':
                                        for i in xrange(v.startIndex,v.endIndex):
                                                f.write(str(i+1)+'\n')
                                if v.vtype=='semiint' or v.vtype=='semicont':
                                        raise Exception('semiint and semicont variables not handled by this LP writer')
                        #binary variables
                        f.write("Binaries\n")
                        for name,v in self.variables.iteritems():
                                if v.vtype=='binary':
                                        for i in xrange(v.startIndex,v.endIndex):
                                                f.write(str(i+1)+'\n')
                
                print 'done.'
                f.close()
                
                
        def convert_quad_to_socp(self):
                for i,c in enumerate(self.constraints):
                        if c.typeOfConstraint=='quad':
                                qd=c.Exp1.quad
                                sqnorm=_quad2norm(qd)
                                self.constraints[i]=sqnorm<-c.Exp1.aff
                                self.numberQuadConstraints-=1
                                self.numberConeConstraints+=1
                                szcone=sqnorm.LR[0].size
                                self.numberConeVars+=(szcone[0]*szcone[1])+2
                if isinstance(self.objective[1],QuadExp):
                        if '_obj_' not in self.variables:
                                obj=self.add_variable('_obj_',1)
                        else:
                                obj=self.get_variable('_obj_')
                        if self.objective[0]=='min':
                                qd=self.objective[1].quad
                                aff=self.objective[1].aff
                                sqnorm=_quad2norm(qd)
                                self.add_constraint(sqnorm<obj-aff)
                                self.set_objective('min',obj)
                        else:
                                qd=(-self.objective[1]).quad
                                aff=self.objective[1].aff
                                sqnorm=_quad2norm(qd)
                                self.add_constraint(sqnorm<aff-obj)
                                self.set_objective('max',obj)
                        self.numberQuadConstraints-=1
                if self.numberQuadConstraints>0:
                        raise Exception('there should not be any quadratics left')
                self.numberQuadNNZ=0
                #reset solver instances
                self.cvxoptVars={'c':None,'A':None,'b':None,'Gl':None,
                                'hl':None,'Gq':None,'hq':None,'Gs':None,'hs':None,
                                'F':None,'g':None, 'quadcons': None}
                
                self.gurobi_Instance = None
                self.grbvar = {}
                
                self.cplex_Instance = None
                self.cplex_boundcons = None
                
                self.msk_env=None
                self.msk_task=None

                self.scip_solver = None
                self.scip_vars = None
                self.scip_obj = None
               
                                
                            
                
#----------------------------------------
#                 Obsolete functions
#----------------------------------------

        def set_varValue(self,name,value):
                self.set_var_value(name,value)
                
        def defaultOptions(self,**opt):
                self.set_all_options_to_default(opt)
                
        def set_options(self, **options):
                self.update_options( **options)

        def addConstraint(self,cons):
                self.add_constraint(cons)
       
        def isContinuous(self):
                return self.is_continuous()
                
        def makeCplex_Instance(self):
                self._make_cplex_instance()
                
        def makeCVXOPT_Instance(self):
                self._make_cvxopt_instance()
