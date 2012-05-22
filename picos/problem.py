# coding: utf-8
import cvxopt as cvx
import numpy as np
import sys
from progress_bar import ProgressBar

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
                self.constraints = {}
                """dictionary of constraints indexed by identifiers"""
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
                #TODOC ?
                
                self.longestkey=0 #for a nice display of constraints
                self.varIndices=[]
                
                self.status='unsolved'
                """status returned by the solver. The default when
                   a new problem is created is 'unsolved'.
                   TODOC mettre un exemple dans le tuto
                """
                

        def __str__(self):
                probstr='---------------------\n'               
                probstr+='optimization problem  ({0}):\n'.format(self.type)
                probstr+='{0} variables, {1} affine constraints'.format(
                                self.numberOfVars,self.numberAffConstraints)
                if self.numberConeVars>0:
                        probstr+=', {0} vars in a SO cone'.format(
                                self.numberConeVars)
                if self.numberLSEConstraints>0:
                        probstr+=', {0} vars in a LOG-SUM-EXP'.format(
                                self.numberLSEConstraints)
                if self.numberSDPConstraints>0:
                        probstr+=', {0} vars in a SD cone'.format(
                                self.numberSDPVars)
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
                Remove all constraints from the problem
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
                self.constraints = {}
                self.numberQuadNNZ=0
                if self.objective[0] is not 'find':
                        if self.objective[1] is not None:
                                expr=self.objective[1]
                                if isinstance(expr,QuadExp):
                                        self.numberQuadNNZ=expr.nnz()

        
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
                        self.numberLSEConstraints+=expr.Exp.size[0]*expr.Exp.size[1]
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
                >>> print (abs(x)**2).value
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
                
                * tol=1e-7 : optimality tolerance for the solver
                * feastol=1e-7 : feasibility tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                * abstol=1e-7 : absolute tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                * reltol=1e-6 : relative tolerance passed to `cvx.solvers.options <http://abel.ee.ucla.edu/cvxopt/userguide/coneprog.html#algorithm-parameters>`_
                * maxit=50 : maximum number of iterations
                * verbose=1 : verbosity level
                * solver=None : currently the available solvers are 'cvxopt','cplex','mosek','smcp','zibopt'.
                  ``None`` means that you let picos select a solver for you.
                * step_sqp=1 : 'first step length for the sequential quadratic programming procedure'
                * harmonic_steps=True : step at the ith step of the sqp procedure is step_sqp/i]
                * onlyChangeObjective'=False : useful when we want to recompute the solution of a
                  problem but with a different objective function. If set to *True* and a new
                  objective function has been passed to the solver, the constraints
                  of the problem will not be parsed next time :func:`solve` is called (this can lead
                  to a huge gain of time).
                * noduals=False : if True, do not retrieve the dual variables
                * nbsol=None (solver default) : maximum number of computed
                  solutions in the solution pool.
                * timelimit =None (infinity) : time limit for the solver
                * acceptableGap =None (0%) : If the time limit is reached, consider the solution as
                  acceptable if the gap is less than this value.
                * treemomory = None (solver default) : size of the buffer for the branch and bound tree,
                  in Megabytes.
                * gaplim = None (0%) : returns a solution as soon as this gap is reached.
                * pool_gap = None (0%) : keeps only the solution
                  within this gap in the pool
                                 
                                 
                .. Warning:: Not all options are handled by all solvers yet.
                
                .. Todo:: Organize above options
                          Ajouter option hotstart
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
                                 'acceptablegap'  :None,
                                 'treememory'     :None,
                                 'gaplim'         :None,
                                 'pool_gap'       :None
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

        
        def update_options(self, **options):
                """
                update the option dictionary, for each pair of the form
                ``key = value``. For a list of available options and their default values,
                see the doc of :func:`set_all_options_to_default() <picos.Problem.set_all_options_to_default>`.
                """
                
                for k in options.keys():
                        self.set_option(k,options[k])
                
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
                                #TODO manque case LSE ?
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
                TEST
                """
                """
                adds a variable in the problem,
                and returns an :class:`AffinExp <picos.AffinExp>` representing this variable.

                For example,
                
                >>> prob=pic.Problem()
                >>> x=prob.add_variable('x',3)
                >>> x
                # (3 x 1)-affine expression: x #
                
                :param name: The name of the variable.
                :type name: str.
                :param size: The size of the variable.
                             
                             Can be either
                             
                                * an ``int`` *n* , in which case the variable is a **vector of dimension n**
                                * or a ``tuple`` *(n,m)*, and the variable is a **n x m-matrix**.
                
                :type size: int or tuple.
                :param vtype: variable :attr:`type <picos.Variable.vtype>`. 
                              Can be : 'continuous', 'binary', 'integer',
                              'symmetric', 'semicont', or 'semiint'
                :type vtype: str.
                :returns: An instance of the class :class:`AffinExp <picos.AffinExp>`:
                          Affine expression representing the created variable.
                
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
                self.constraints[self.countCons]=cons
                self.consNumbering.append(self.countCons)
                self.countCons+=1
                if cons.typeOfConstraint[:3]=='lin':
                        self.numberAffConstraints+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                elif cons.typeOfConstraint[2:]=='cone':
                        self.numberConeVars+=(cons.Exp1.size[0]*cons.Exp1.size[1])
                        self.numberConeConstraints+=1
                elif cons.typeOfConstraint=='lse':
                        self.numberLSEConstraints+=(cons.Exp1.size[0]*cons.Exp1.size[1])
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
                24 variables, 9 affine constraints, 9 vars in a SO cone
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
                
                               * if ind is an ``int`` *n*, then the nth constraint (starting from 0)
                                 will be returned, where all the constraints are counted
                                 in the order where they were passed to the problem.
                                 
                               * if ind is a ``tuple`` *(k,i)*, then the ith constraint
                                 from the kth group of constraints is returned
                                 (starting from 0). By            
                                 *group of constraints*, it is meant a single constraint
                                 or a list of constraints added together with the
                                 function :func:`add_list_of_constraints() <picos.Problem.add_list_of_constraints>`.
                
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
                >>> prob.get_constraint(5)                              #6th constraint
                # (5x1)-affine constraint: y > |0| #
                
                """
                indtuple=ind
                if isinstance(indtuple,int):
                        return self.constraints[indtuple]
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
                if isinstance(lsind,list):
                                raise Exception('too few indices')
                return self.constraints[lsind]
                
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
                
                TODO: SOCP
                """
                try:
                        import cplex
                except:
                        Exception('cplex library not found')
                
                c = cplex.Cplex()
                import itertools
                
                if not self.options['timelimit'] is None:
                        import timelimitcallback
                        import time
                        timelim_cb = c.register_callback(timelimitcallback.TimeLimitCallback)
                        timelim_cb.starttime = time.time()
                        timelim_cb.timelimit = self.options['timelimit']
                        if not self.options['acceptablegap'] is None:
                                timelim_cb.acceptablegap =self.options['acceptableGap']
                        else:
                                timelim_cb.acceptablegap = 100
                        timelim_cb.aborted = 0
                        #c.parameters.tuning.timelimit.set(self.options['timelimit']) #DOES NOT WORK LIKE THIS ?
                if not self.options['treememory'] is None:
                        c.parameters.mip.limits.treememory.set(self.options['treememory'])
                if not self.options['gaplim'] is None:
                        c.parameters.mip.tolerances.mipgap.set(self.options['gaplim'])
                if not self.options['nbsol'] is None:
                        c.parameters.mip.limits.solutions.set(self.options['nbsol'])
                if not self.options['pool_gap'] is None:
                        c.parameters.mip.pool.relgap.set(self.options['pool_gap'])
                
                
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
                for constrKey,constr in self.constraints.iteritems():
                        if icone == 0: #first conic constraint
                                noconstant=self.add_variable(
                                        '__noconstant__',1)
                                newcons['noconstant']=(
                                        noconstant>0)
                        if constr.typeOfConstraint=='SOcone':
                                tmplhs.append(self.add_variable(
                                        '__tmplhs[{0}]__'.format(icone),
                                        constr.Exp1.size[0]*constr.Exp1.size[1]
                                        ))
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
                                tmplhs.append(self.add_variable(
                                        '__tmplhs[{0}]__'.format(icone),
                                        constr.Exp1.size[0]*constr.Exp1.size[1]
                                        ))
                                tmprhs.append(self.add_variable(
                                        '__tmprhs[{0}]__'.format(icone),
                                        1))
                                #v_cons is 0/1/-1 to avoid constants in cone (problem with duals)
                                expcat = ((2*constr.Exp1[:] // (constr.Exp2-constr.Exp3))
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
                        prog = ProgressBar(0,limitbar, 77, mode='fixed')
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
                        prog = ProgressBar(0,limitbar, 77, mode='fixed')
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
                
                allcons = list(self.constraints.iteritems()) + list(
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
                        elif constr.typeOfConstraint[2:] == 'cone':
                                pass  #will be handled in the newcons dictionary
                                
                        else:
                                raise Exception('type of constraint not handled (yet ?) for cplex:{0}'.format(
                                        constr.typeOfConstraint))
                                
                        
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
                elif tp in ('QCQP','SOCP'):
                        c.set_problem_type(c.problem_type.QCP)
                elif tp in ('MIQCP','MISOCP'):
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
                
                return c, self

                
        def _make_cvxopt_instance(self,aff_part_of_quad=True):
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
                        prog = ProgressBar(0,limitbar, 77, mode='fixed')
                        oldprog = str(prog)
                
                #constraints                
                for k in self.constraints.keys():
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
                                (A,b)=self._makeGandh(self.constraints[k].Exp1)
                                (c,d)=self._makeGandh(self.constraints[k].Exp2)
                                self.cvxoptVars['Gq'].append(cvx.sparse([-c,-A]))
                                self.cvxoptVars['hq'].append(cvx.matrix([d,b]))
                        elif self.constraints[k].typeOfConstraint=='RScone':
                                (A,b)=self._makeGandh(self.constraints[k].Exp1)
                                (c1,d1)=self._makeGandh(self.constraints[k].Exp2)
                                (c2,d2)=self._makeGandh(self.constraints[k].Exp3)
                                self.cvxoptVars['Gq'].append(cvx.sparse([-c1-c2,-2*A,c2-c1]))
                                self.cvxoptVars['hq'].append(cvx.matrix([d1+d2,2*b,d1-d2]))
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
                        newobj=self.options['onlyChangeObjective']
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
                        ptch=self.add_variable('_ptch_',1)
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
                
                .. TODO:: Some quadratic problems do not pass. Ask Ryan after reinstallation
                """
                try:
                        from zibopt import scip
                except:
                        raise Exception('scip library not found')
                
                scip_solver = scip.solver(quiet=not(self.options['verbose']))
                
                self._make_cvxopt_instance(aff_part_of_quad=False)
                
                if bool(self.cvxoptVars['Gs']) or bool(self.cvxoptVars['F']) or bool(self.cvxoptVars['Gq']):
                        raise Exception('SDP, SOCP, or GP constraints are not implemented in mosek')
                                
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
                for i in range(self.cvxoptVars['A'].size[1]):
                    if not(self.cvxoptVars['c'] is None):
                        x.append(scip_solver.variable(types[i],
                                lower=-INFINITY,
                                coefficient=self.cvxoptVars['c'][i])
                            )
                    else:
                        x.append(scip_solver.variable(types[i],
                                lower=-INFINITY))
                
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
                #quadratic constraints
                for (k,iaff) in self.cvxoptVars['quadcons']:
                        subI=[]
                        subJ=[]
                        subV=[]
                        if k=='_obj':
                                qexpr=self.objective[1]
                        else:
                                qexpr=self.constraints[k].Exp1

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
                                self.scip_obj = qd
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
                        
                # For cplex (only LP and MIP implemented)
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
                        #TODO:Other solvers (GUROBI, ...)
                
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
                
                .. Todo:: * handle quadratic problems
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
                if self.options['solver'].upper()=='CVXOPT':
                        currentsolver=None
                elif self.options['solver']=='cvxopt-mosek':
                        currentsolver='mosek'
                elif self.options['solver']=='smcp':
                        currentsolver='smcp'
                #-------------------------------#
                #  runs the appropriate solver  #
                #-------------------------------#
                if  self.numberQuadConstraints>0:#QCQP
                        probtype='QCQP'
                        raise Exception('CVXOPT with Quadratic constraints is not handled')
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

                                for k in self.constraints.keys():
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
                           * handle SOCP ?
                           * dual of quadratics ?
                """
                #----------------------------#
                #  create the cplex instance #
                #----------------------------#
                self._make_cplex_instance()
                c = self.cplex_Instance
                
                if c is None:
                        raise ValueError('a cplex instance should have been created before')
                
                
                #TODO : settings parameters
                
                #--------------------#
                #  call the solver   #
                #--------------------#                
              
                if not self.options['nbsol'] is None:
                        try:
                                c.populate_solution_pool()
                        except:
                                print "Exception raised during populate"
                else:
                        try:
                                c.solve()
                        except:
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
                                for ob,ind in sorted(objvals,reverse=rev)[:self.options['nbsol']]:
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
                                for k,constr in self.constraints.iteritems():
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
                                                                       boundsj=[b0 for k0 in self.constraints
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
                                                #TODO check pour RScone
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
                          * Dual vars for QP
                          * Direct handling of rotated cones
                """
                #----------------------------#
                #  create the mosek instance #
                #----------------------------#
                import mosek
                self._make_mosek_instance()
                task=self.msk_task

                #TODO : settings parameters
                
                #--------------------#
                #  call the solver   #
                #--------------------# 
                #optimize
                task.optimize()
                
                # Print a summary containing information
                # about the solution for debugging purposes
                task.solutionsummary(mosek.streamtype.msg)
                prosta = []
                solsta = []

                if self.is_continuous():
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
                                for k in self.constraints.keys():
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
                
                .. Todo:: * dual variables ?
                          * solver parameters
                """
                #-----------------------------#
                #  create the zibopt instance #
                #-----------------------------#
                self._make_zibopt()
                
                timelimit=10000000.
                gaplim=0.
                nbsol=-1
                if not self.options['timelimit'] is None:
                        timelimit=self.options['timelimit']
                if not self.options['gaplim'] is None:        
                        gaplim=self.options['gaplim']
                
                #if fact, nbsol is a limit on the number of feasible nodes visited
                #if not self.options['nbsol'] is None:
                #        nbsol=self.options['nbsol']
                
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
                                        return 'SDP + quadratic constraint (not handled -- need to reformulate)'
                        #SOCP
                        if self.numberConeConstraints > 0:
                                if self.numberQuadConstraints == 0:
                                        return 'SOCP'
                                else:
                                        return 'SOCP + quadratic constraint (not handled -- need to reformulate)'

                        #quadratic problem
                        if self.numberQuadConstraints>0:
                                if any([cs.typeOfConstraint=='quad' for cs in self.constraints.values()]):
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
                                        return 'integer SOCP + QP (not handled)'
                        if self.numberQuadConstraints>0:
                                if any([cs.typeOfConstraint=='quad' for cs in self.constraints.values()]):
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
        #TODO example in the tuto ?                                        

        def solver_selection(self):
                """Selects an appropriate solver for this problem
                and sets the option ``'solver'``.
                """
                tp=self.type
                if tp == 'LP':
                        order=['cplex','mosek','zibopt','cvxopt','smcp']
                elif tp == 'QCQP': #add cplex,zibopt ?
                        order=['mosek','zibopt','cvxopt']
                elif tp == 'SOCP':
                        order=['mosek','cvxopt','smcp']
                elif tp == 'SDP':
                        order=['cvxopt','smcp']
                elif tp == 'ConeP': #add cplex ?
                        order=['cvxopt','smcp']
                elif tp == 'GP': #add cplex ?
                        order=['cvxopt']
                elif tp == 'general-obj':
                        order=['mosek','zibopt','cvxopt']
                elif tp == 'MIP':
                        order=['cplex','mosek','zibopt']
                elif tp == 'MIQCP': #add cplex,zibopt ?
                        order=['mosek']
                elif tp == 'MISOCP':
                        order=['mosek']
                else:
                        raise Exception('no solver available for problem of type {0}'.format(tp))
                avs=available_solvers()
                for sol in order:
                        if sol in avs:
                                self.set_option('solver',sol)
                                return
                #not found
                raise Exception('no solver available for problem of type {0}'.format(tp))
                
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
