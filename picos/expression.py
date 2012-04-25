# coding: utf-8
import cvxopt as cvx
import numpy as np
import sys
from progress_bar import ProgressBar

from .tools import *
from .constraint import *

__all__=['Expression',
        'AffinExp',
        'Norm',
        'QuadExp',
        'GeneralFun',
        'LogSumExp'
        ]
       
#----------------------------------
#                Expression
#----------------------------------
                
class Expression(object):
        """the parent class of AffinExp, Norm, LogSumExp, QuadExp, GeneralFun"""
        def __init__(self,string,variables):
                self.string=string
                if variables is None:
                        raise Exception('unexpected case')
                self.variables=variables
                
        def eval(self):
                pass
        
        def set_value(self,value):
                raise ValueError('set_value can only be called on a simple Expression representing a variable')
                
        def del_simple_var_value(self):
                raise ValueError('del_simple_var_value can only be called on a simple Expression representing a variable')
                                
        value = property(eval,set_value,del_simple_var_value,"value of the affine expression")
        
        def is_valued(self):
                try:
                        val=self.value
                        return not(val is None)
                except Exception:
                        return False
#----------------------------------
#                 AffinExp class
#----------------------------------

class AffinExp(Expression):
        """a class for defining vectorial (or matrix) affine expressions
        *The dictionary 'factors' stores, for each variable, a
        tuple (fact,string), where 'string' is a representation of the
        linear combination, and 'fact' is the factor by which the variable
        it is multiplied, (if the variable is a matrix, then the factor
        is with respect to the column-vectorization of the variable).
        The factor is stored as a cvx matrix or cvx spmatrix.
        For example (if x is a vector variable and X is a matrix variable): 
                _the product A*x is stored as a pair 'x':A
                _the product A*X is stored as a pair 'X':blkdiag(A,...,A)
                        where the bloc diagonal matrix has a block
                        corresponding to each column of X
                _the scalar product <A,X> is stored as 'X':A[:].T
                        where A[:] is the colum-vectorization of A
        *Similarly, the 'constant' attribute stores a tuple
                ( <vectorized constant>,string )
                If the constant is 0, then 'constant' can be (None,'0')
        """
        
        def __init__(self,factors=None,constant=None,
                        size=(1,1),
                        string='0',
                        variables=None
                        ):
                if factors is None:
                        factors={}
                Expression.__init__(self,string,variables)
                self.factors=factors
                self.constant=constant
                self.size=size
                #self.string=string
                #self.variables=variables
                
        def __str__(self):
                if self.is_valued():
                        return str(self.eval())
                else:
                        return repr(self)

        def __repr__(self):
                affstr='# ({0} x {1})-affine expression: '.format(self.size[0],
                                                                self.size[1])
                affstr+=self.affstring()
                affstr+=' #'
                return affstr
                        
                

        def affstring(self):
                return self.string

        def eval(self,ind=None):
                if self.constant is None:
                        val=cvx.spmatrix([],[],[],(self.size[0]*self.size[1],1))
                else:
                        val=self.constant
                #for k in self.factors.keys():
                        #if MATH_PROG_PROBLEMS['current'] is None:
                                #raise Exception(k+' is not valued')                    
                        #if not MATH_PROG_PROBLEMS['current'].variables[k].value is None:
                                #val=val+self.factors[k]*MATH_PROG_PROBLEMS['current'].variables[k].value[:]
                        #else:
                                #raise Exception(k+' is not valued')
                for k in self.factors.keys():
                        if k not in self.variables:
                                raise Exception(k+' is an unknown variable')
                        if ind is None:
                                if not self.variables[k].value is None:
                                        val=val+self.factors[k]*self.variables[k].value[:]
                                else:
                                        raise Exception(k+' is not valued')
                        else:
                                if ind in self.variables[k].value_alt:
                                        val=val+self.factors[k]*self.variables[k].value_alt[ind][:]
                                else:
                                        raise Exception(k+' does not have a value for the index '+str(ind))
                return cvx.matrix(val,self.size)
                
        def set_value(self,value):
                #TODOC
                var = self.get_simple_variable()
                if var is None:
                        raise ValueError('set_value can only be called on a simple Expression representing a variable')
                                
                valuemat,valueString = _retrieve_matrix(value,var.size)
                if valuemat.size != var.size:
                        raise Exception('should be of size {0}'.format(var.size))
                if var.vtype == 'symmetric':
                        valuemat=svec(valuemat)
                var.value = valuemat

        def del_simple_var_value(self):
               #TODOC
                var = self.get_simple_variable()
                if var is None:
                        raise ValueError('del_simple_var_value can only be called on a simple Expression representing a variable')
                
                var.value = None
                       
        value = property(eval,set_value,del_simple_var_value,"value of the affine expression")
        
        def get_type(self):
               #TODOC
                var = self.get_simple_variable()
                if var is None:
                        raise ValueError('get_type can only be called on a simple Expression representing a variable')
                
                return var.vtype
                
        def set_type(self,value):
               #TODOC
                var = self.get_simple_variable()
                if var is None:
                        raise ValueError('set_type can only be called on a simple Expression representing a variable')
                
                var.vtype = value
                
        def del_type(self):
                raise AttributeError('attribute type cannot be deleted')
        
        vtype = property(get_type,set_type,del_type,'type for an expression representing a simple variable')
        
        def get_name(self):
               #TODOC
                var = self.get_simple_variable()
                if var is None:
                        raise ValueError('get_name can only be called on a simple Expression representing a variable')
                
                return var.name
                
        def set_name(self,value):
                raise AttributeError('attribute name is not writable')
                
        def del_name(self):
                raise AttributeError('attribute name is not writable')
                
        name = property(get_name,set_name,del_name,'name of the simple variable represented by the expression')
        
        def get_variable(self):
                #TODOC
                var = self.get_simple_variable()
                if var is None:
                        raise ValueError('get_variable can only be called on a simple Expression representing a variable')
                return var
                
        def set_variable(self,value):
                raise AttributeError('attribute variable is not writable')
                
        def del_variable(self):
                raise AttributeError('attribute variable is not writable')
                
        variable = property(get_variable,set_variable,del_variable,'Variable instance of the simple variable represented by the affine expression')                
        
        def is_valued(self, ind = None):
                
                for k in self.factors.keys():
                        if k not in self.variables:
                                raise Exception(k+' is an unknown variable')
                        if ind is None:
                                if self.variables[k].value is None:
                                        return False
                        else:
                                if ind  not in self.variables[k].value_alt:
                                        return False

                #Yes, you can call eval(ind) without any problem.
                return True
                
                
        
        def get_simple_variable(self):
                #TODOC
                #returns None or the simple variable
                
                #is there exactly one factor
                if len(self.factors) != 1:
                        return None
                
                #is there a nozero constant ?
                if not(self.constant is None):
                        if max(abs(self.constant)) != 0:
                                return None
                
                #is the factor equal to the identity matrix ?        
                mm=self.factors.values()[0] #should be identity matrix
                name=self.factors.keys()[0]
                if name not in self.variables:
                        raise Exception('unexpected variable name')
                vtype=self.variables[name].vtype
                size=self.variables[name].size
                idmat=_svecm1_identity(vtype,size)
                #test if mm==idmat
                mlist=sorted(zip(mm.I,mm.J,mm.V))
                idlist=sorted(zip(idmat.I,idmat.J,idmat.V))
                if (mlist!=idlist):
                        return None
                        
                #We have a simple variable
                return self.variables[name]
                
                
                
        
        def is0(self):
                if bool(self.constant):
                        return False
                for f in self.factors:
                        if bool(self.factors[f]):
                                return False
                return True

        def is1(self):
                if not bool(self.constant):
                        return False
                if not(self.size==(1,1) and self.constant[0]==1):
                        return False
                for f in self.factors:
                        if bool(self.factors[f]):
                                return False        
                return True

        def isconstant(self):
                for f in self.factors:
                        if bool(self.factors[f]):
                                return False
                return True

        def transpose(self):
                """Transposition"""
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                                        self.string,variables=self.variables)
                for k in selfcopy.factors:
                        bsize=selfcopy.size[0]
                        bsize2=selfcopy.size[1]
                        I0=[(i/bsize)+(i%bsize)*bsize2 for i in selfcopy.factors[k].I]
                        J=selfcopy.factors[k].J
                        V=selfcopy.factors[k].V
                        selfcopy.factors[k]=cvx.spmatrix(V,I0,J,selfcopy.factors[k].size)
                        '''old version -- not very efficient
                        newfac=cvx.spmatrix([],[],[],(0,selfcopy.factors[k].size[1]))
                        n=selfcopy.size[1]
                        m=selfcopy.size[0]
                        for j in range(m):
                                for i in range(n):
                                        newfac=cvx.sparse([newfac,selfcopy.factors[k][i*m+j,:]])
                        selfcopy.factors[k]=newfac
                        '''
                        
                if not (selfcopy.constant is None):
                        selfcopy.constant=cvx.matrix(selfcopy.constant,
                                        selfcopy.size).T[:]
                selfcopy.size=(selfcopy.size[1],selfcopy.size[0])
                if ( ('*' in selfcopy.affstring()) or ('/' in selfcopy.affstring())
                        or ('+' in selfcopy.affstring()) or ('-' in selfcopy.affstring()) ):
                        selfcopy.string='( '+selfcopy.string+' ).T'
                else:
                        selfcopy.string+='.T'
                return selfcopy
                
        def setT(self,value):
                raise AttributeError("attribute 'T' of 'AffinExp' is not writable")
        
        def delT(self):
                raise AttributeError("attribute 'T' of 'AffinExp' is not writable")
        
        T = property(transpose,setT,delT,"transposition")

        def __rmul__(self,fact):
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                                        self.string,variables=self.variables)
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=fact.eval(),fact.string
                        else:
                                raise Exception('not implemented')
                else:
                        fac,facString=_retrieve_matrix(fact,self.size[0])               
                if fac.size==(1,1) and selfcopy.size[0]<>1:
                        fac=fac[0]*cvx.spdiag([1.]*selfcopy.size[0])
                if self.size==(1,1) and fac.size[1]<>1:
                        oldstring=selfcopy.string
                        selfcopy=selfcopy.diag(fac.size[1])
                        selfcopy.string=oldstring
                if selfcopy.size[0]<>fac.size[1]:
                        raise Exception('incompatible dimensions')
                bfac=_blocdiag(fac,selfcopy.size[1])
                for k in selfcopy.factors:
                        newfac=bfac*selfcopy.factors[k]
                        selfcopy.factors[k]=newfac
                if selfcopy.constant is None:
                        newfac=None
                else:
                        newfac=bfac*selfcopy.constant
                selfcopy.constant=newfac
                selfcopy.size=(fac.size[0],selfcopy.size[1])
                #the following removes 'I' from the string when a matrix is multiplied
                #by the identity. We leave the 'I' when the factor of identity is a scalar
                if len(facString)>0:            
                        if facString[-1]=='I' and (len(facString)==1
                                 or facString[-2].isdigit() or facString[-2]=='.') and (
                                 self.size != (1,1)):
                                facString=facString[:-1]
                if len(facString)>0:
                        if ('+' in selfcopy.affstring()) or ('-' in selfcopy.affstring()):
                                selfcopy.string=facString+'*( '+selfcopy.string+' )'
                        else:
                                selfcopy.string=facString+'*'+selfcopy.string
                return selfcopy


        
        def __mul__(self,fact):
                #product of 2 affine expressions
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=fact.eval(),fact.string           
                        elif self.isconstant():
                                return fact.__rmul__(self)
                        elif self.size[0]==1 and fact.size[1]==1 and self.size[1]==fact.size[0]:
                                #quadratic expression
                                linpart=AffinExp({},constant=None,size=(1,1),variables=self.variables)
                                if not self.constant is None:
                                        linpart=linpart+self.constant.T*fact
                                if not fact.constant is None:
                                        linpart=linpart+self*fact.constant
                                if not ((fact.constant is None) or (self.constant is None)):
                                        linpart=linpart-self.constant.T*fact.constant
                                """if not ( self.constant is None or fact.constant is None):
                                        linpart.constant=self.constant.T*fact.constant
                                if not fact.constant is None:                           
                                        for k in self.factors.keys():
                                                linpart.factors[k]=fact.constant.T*self.factors[k]
                                if not self.constant is None:
                                        for k in fact.factors.keys():
                                                if k in linpart.factors.keys():
                                                        linpart.factors[k]+=self.constant.T*fact.factors[k]
                                                else:
                                                        linpart.factors[k]=self.constant.T*fact.factors[k]
                                """
                                quadpart={}
                                for i in self.factors.keys():
                                        for j in fact.factors.keys():
                                                quadpart[i,j]=self.factors[i].T*fact.factors[j]
                                stleft=self.affstring()
                                stright=fact.affstring()
                                if ('+' in stleft) or ('-' in stleft):
                                        if len(stleft)>3 and not(stleft[0]=='(' and stleft[-3:]==').T'):
                                                stleft='( '+stleft+' )'
                                if ('+' in stright) or ('-' in stright):
                                        stright='( '+stright+' )'                               
                                if self.size[1]==1:
                                        return QuadExp(quadpart,linpart,stleft+'*'+stright,LR=(self,fact),variables=self.variables)
                                else:
                                        return QuadExp(quadpart,linpart,stleft+'*'+stright,variables=self.variables)
                        else:
                                raise Exception('not implemented')
                elif isinstance(fact,QuadExp):
                        return QuadExp*fact
                #product with a constant
                else:
                        if self.size==(1,1): #scalar mult. of the constant
                                fac,facString=_retrieve_matrix(fact,None)
                        else: #normal matrix multiplication, we expect a size        
                                fac,facString=_retrieve_matrix(fact,self.size[1])
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                                self.string,variables=self.variables)
                if fac.size==(1,1) and selfcopy.size[1]<>1:
                        fac=fac[0]*cvx.spdiag([1.]*selfcopy.size[1])
                if self.size==(1,1) and fac.size[0]<>1:
                        oldstring=selfcopy.string
                        selfcopy=selfcopy.diag(fac.size[0])
                        selfcopy.string=oldstring
                prod=(self.T.__rmul__(fac.T)).T
                prod.size=(selfcopy.size[0],fac.size[1])
                #the following removes 'I' from the string when a matrix is multiplied
                #by the identity. We leave the 'I' when the factor of identity is a scalar
                if len(facString)>0:
                        if facString[-1]=='I' and (len(facString)==1
                                 or facString[-2].isdigit() or facString[-2]=='.') and(
                                 self.size != (1,1)):
                                facString=facString[:-1]
                if len(facString)>0:
                        if ('+' in selfcopy.affstring()) or ('-' in selfcopy.affstring()):
                                prod.string='( '+selfcopy.string+' )*'+facString
                        else:
                                prod.string=selfcopy.string+'*'+facString
                else:
                        prod.string=selfcopy.string
                return prod
        
        def __or__(self,fact):#scalar product
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                        self.string,variables=self.variables)
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=fact.eval(),fact.string
                        elif self.isconstant():
                                return fact.__ror__(self)       
                        else:
                                dotp = self.T*fact
                                dotp.string='〈 '+self.string+' | '+fact.string+' 〉'
                                return dotp
                                #raise Exception('not implemented')
                else:           
                        fac,facString=_retrieve_matrix(fact,self.size)
                if selfcopy.size<>fac.size:
                        raise Exception('incompatible dimensions')
                cfac=fac[:].T
                for k in selfcopy.factors:
                        newfac=cfac*selfcopy.factors[k]
                        selfcopy.factors[k]=newfac
                if selfcopy.constant is None:
                        newfac=None
                else:
                        newfac=cfac*selfcopy.constant
                selfcopy.constant=newfac
                selfcopy.size=(1,1)
                if facString[-1]=='I' and (len(facString)==1
                                 or facString[-2].isdigit() or facString[-2]=='.'):
                        selfcopy.string=facString[:-1]+'trace( '+selfcopy.string+' )'
                else:
                        #selfcopy.string= u'\u2329 '+selfcopy.string+' | '+facString+u' \u232a'
                        selfcopy.string='〈 '+selfcopy.string+' | '+facString+' 〉'
                        #'\xe2\x8c\xa9 '+selfcopy.string+' | '+facString+' \xe2\x8c\xaa'
                return selfcopy

        def __ror__(self,fact):
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                        self.string,variables=self.variables)
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=fact.eval(),fact.string
                        else:
                                dotp = fact.T * self
                                dotp.string='〈 '+fact.string+' | '+self.string+' 〉'
                                return dotp
                                #raise Exception('not implemented')
                else:           
                        fac,facString=_retrieve_matrix(fact,self.size)
                if selfcopy.size<>fac.size:
                        raise Exception('incompatible dimensions')
                cfac=fac[:].T
                for k in selfcopy.factors:
                        newfac=cfac*selfcopy.factors[k]
                        selfcopy.factors[k]=newfac
                if selfcopy.constant is None:
                        newfac=None
                else:
                        newfac=cfac*selfcopy.constant
                selfcopy.constant=newfac
                selfcopy.size=(1,1)
                if facString[-1]=='I' and (len(facString)==1
                                 or facString[-2].isdigit() or facString[-2]=='.'):
                        selfcopy.string=facString[:-1]+'trace( '+selfcopy.string+' )'
                else:
                        #selfcopy.string=u'\u2329 '+facString+' | '+selfcopy.string+u' \u232a'
                        selfcopy.string='〈 '+facString+' | '+selfcopy.string+' 〉'
                return selfcopy

        
        def __add__(self,term):
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                                        self.string,variables=self.variables)
                if isinstance(term,AffinExp):
                        if term.size==(1,1) and self.size<>(1,1):
                                oldstring=term.string
                                term=cvx.matrix(1.,self.size)*term.diag(self.size[1])
                                term.string='|'+oldstring+'|'
                        if self.size==(1,1) and term.size<>(1,1):
                                oldstring=self.string
                                selfone=cvx.matrix(1.,term.size)*self.diag(term.size[1])
                                selfone.string='|'+oldstring+'|'
                                return (selfone+term)
                        if term.size<>selfcopy.size:
                                raise Exception('incompatible dimension in the sum')
                        for k in term.factors.keys():
                                if k in selfcopy.factors.keys():
                                        newfac=selfcopy.factors[k]+term.factors[k]
                                        selfcopy.factors[k]=newfac
                                else:
                                        selfcopy.factors[k]=term.factors[k]
                        if selfcopy.constant is None and term.constant is None:
                                pass
                        else:
                                newCons=cvx.spmatrix([],[],[],selfcopy.size)[:]
                                if not selfcopy.constant is None:
                                        newCons=newCons+selfcopy.constant
                                if not term.constant is None:
                                        newCons=newCons+term.constant
                                selfcopy.constant=newCons
                        if term.affstring() not in ['0','','|0|','0.0','|0.0|']:
                                if term.string[0]=='-':
                                        import re                                       
                                        if ('+' not in term.string[1:]) and (
                                                '-' not in term.string[1:]):
                                                selfcopy.string=selfcopy.string+' '+term.affstring()
                                        elif (term.string[1]=='(') and (
                                                 re.search('.*\)((\[.*\])|(.T))*$',term.string) ):                                                              #a group in a (...)
                                                selfcopy.string=selfcopy.string+' '+term.affstring()
                                        else:
                                                selfcopy.string=selfcopy.string+' + ('+ \
                                                                term.affstring()+')'
                                else:
                                        selfcopy.string+=' + '+term.affstring()
                        return selfcopy
                elif isinstance(term,QuadExp):
                        if self.size<>(1,1):
                                raise Exception('LHS must be scalar')
                        expQE=QuadExp({},self,self.affstring(),variables=self.variables)
                        return expQE+term
                else: #constant term
                        term,termString=_retrieve_matrix(term,selfcopy.size)
                        return self+AffinExp({},constant=term[:],size=term.size,string=termString,variables=self.variables)

        def __radd__(self,term):
                return self.__add__(term)

        def __neg__(self):
                selfneg=(-1)*self               
                if self.string<>'':
                        if self.string[0]=='-':
                                import re
                                if ('+' not in self.string[1:]) and ('-' not in self.string[1:]):
                                        selfneg.string=self.string[1:]
                                elif (self.string[1]=='(') and (
                                   re.search('.*\)((\[.*\])|(.T))*$',self.string) ): #a group in a (...)
                                        if self.string[-1]==')':
                                                selfneg.string=self.string[2:-1] #we remove the parenthesis
                                        else:
                                                selfneg.string=self.string[1:] #we keep the parenthesis
                                else:
                                        selfneg.string='-('+self.string+')'
                        else:
                                if ('+' in self.string) or ('-' in self.string):
                                        selfneg.string='-('+self.string+')'
                                else:
                                        selfneg.string='-'+self.string
                return selfneg
                
        def __sub__(self,term):
                if isinstance(term,AffinExp) or isinstance(term,QuadExp):
                        return self+(-term)
                else: #constant term
                        term,termString=_retrieve_matrix(term,self.size)
                        return self-AffinExp({},constant=term[:],size=term.size,string=termString,variables=self.variables)

        def __rsub__(self,term):
                return term+(-self)

        def __div__(self,divisor): #division (by a scalar)
                if isinstance(divisor,AffinExp):
                        if divisor.isconstant():
                                divi,diviString=divisor.eval(),divisor.string
                        else:
                                raise Exception('not implemented')
                        if divi.size<>(1,1):
                                raise Exception('not implemented')
                        if divi[0]==0:
                                raise Exception('Division By Zero')
                        division=self * (1/divi)
                        if ('+' in self.string) or ('-' in self.string):
                                division.string = '('+ self.string + ') /' + diviString
                        else:
                                division.string =  self.string+ ' / ' + diviString
                        return division
                else : #constant term
                        divi,diviString=_retrieve_matrix(divisor,(1,1))
                        return self/AffinExp({},constant=divi[:],size=(1,1),string=diviString,variables=self.variables)

        def __rdiv__(self,divider):
                divi,diviString=_retrieve_matrix(divider,None)
                return AffinExp({},constant=divi[:],size=divi.size,string=diviString,variables=self.variables)/self
                                                

        def __getitem__(self,index):
                #TODO check negative indices
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                                self.string,variables=self.variables)
                def slicestr(sli):
                        #single element
                        if not (sli.start is None or sli.stop is None):
                                if (sli.stop==sli.start+1):
                                        return str(sli.start)
                        #single element -1
                        if (sli.start==-1 and sli.stop is None and sli.step is None):
                                return '-1'
                        ss=''
                        if not sli.start is None:
                                ss+=str(sli.start)
                        ss+=':'
                        if not sli.stop is None:
                                ss+=str(sli.stop)
                        if not sli.step is None:
                                ss+=':'
                                ss+=str(sli.step)
                        return ss
                if isinstance(index,int):
                        if index==-1: #(-1,0) does not work
                                index=slice(index,None,None)
                        else:
                                index=slice(index,index+1,None)
                if isinstance(index,slice):
                        idx=index.indices(self.size[0]*self.size[1])
                        rangeT=range(idx[0],idx[1],idx[2])
                        for k in selfcopy.factors.keys():
                                selfcopy.factors[k]=selfcopy.factors[k][rangeT,:]
                        if not selfcopy.constant is None:
                                selfcopy.constant=selfcopy.constant[rangeT]
                        selfcopy.size=(len(rangeT),1)
                        indstr=slicestr(index)
                elif isinstance(index,tuple):
                        if isinstance(index[0],int):
                                index=(slice(index[0],index[0]+1,None),index[1])
                        if isinstance(index[1],int):
                                index=(index[0],slice(index[1],index[1]+1,None))
                        idx0=index[0].indices(self.size[0])
                        idx1=index[1].indices(self.size[1])
                        rangei=range(idx0[0],idx0[1],idx0[2])
                        rangej=range(idx1[0],idx1[1],idx1[2])
                        rangeT=[]
                        for j in rangej:
                                rangei_translated=[]
                                for vi in rangei:
                                        rangei_translated.append(
                                                vi+(j*self.size[0]))
                                rangeT.extend(rangei_translated)
                        for k in selfcopy.factors.keys():
                                selfcopy.factors[k]=selfcopy.factors[k][rangeT,:]
                        if not selfcopy.constant is None:       
                                selfcopy.constant=selfcopy.constant[rangeT]
                        selfcopy.size=(len(rangei),len(rangej))
                        indstr=slicestr(index[0])+','+slicestr(index[1])
                if ('*' in selfcopy.affstring()) or ('+' in selfcopy.affstring()) or (
                        '-' in selfcopy.affstring()) or ('/' in selfcopy.affstring()):
                        selfcopy.string='( '+selfcopy.string+' )['+indstr+']'
                else:
                        selfcopy.string=selfcopy.string+'['+indstr+']'
                #check size
                if selfcopy.size[0] == 0 or selfcopy.size[1] == 0:
                        raise AttributeError('slice of zero-dimension')
                return selfcopy
                
        def __setitem__(self, key, value):
                raise AttributeError('slices of an expression are not writable')
        
        def __delitem__(self):
                raise AttributeError('slices of an expression are not writable')
        
        
        def __lt__(self,exp):
                if isinstance(exp,AffinExp):
                        if exp.size==(1,1) and self.size<>(1,1):
                                oldstring=exp.string
                                exp=cvx.matrix(1.,self.size)*exp.diag(self.size[1])
                                exp.string='|'+oldstring+'|'
                        if self.size==(1,1) and exp.size<>(1,1):
                                oldstring=self.string
                                selfone=cvx.matrix(1.,exp.size)*self.diag(exp.size[1])
                                selfone.string='|'+oldstring+'|'
                                return (selfone<exp)
                        return Constraint('lin<',None,self,exp)
                elif isinstance(exp,QuadExp):
                        if (self.isconstant() and self.size==(1,1)
                                and (not exp.LR is None) and (not exp.LR[1] is None)
                        ):
                                cst=AffinExp( factors={},constant=cvx.matrix(np.sqrt(self.eval()),(1,1)),
                                        size=(1,1),string=self.string,variables=self.variables)
                                return (Norm(cst)**2)<exp
                        elif self.size==(1,1):
                                return (-exp)<(-self)
                        else:
                                raise Exception('not implemented')
                else:                   
                        term,termString=_retrieve_matrix(exp,self.size)
                        exp2=AffinExp(factors={},constant=term[:],size=self.size,string=termString,variables=self.variables)
                        return Constraint('lin<',None,self,exp2)

        def __gt__(self,exp):
                if isinstance(exp,AffinExp):
                        if exp.size==(1,1) and self.size<>(1,1):
                                oldstring=exp.string
                                exp=cvx.matrix(1.,self.size)*exp.diag(self.size[1])
                                exp.string='|'+oldstring+'|'
                        if self.size==(1,1) and exp.size<>(1,1):
                                oldstring=self.string
                                selfone=cvx.matrix(1.,exp.size)*self.diag(exp.size[1])
                                selfone.string='|'+oldstring+'|'
                                return (selfone>exp)    
                        return Constraint('lin>',None,self,exp)
                elif isinstance(exp,QuadExp):
                        return exp<self
                else:                   
                        term,termString=_retrieve_matrix(exp,self.size)
                        exp2=AffinExp(factors={},constant=term[:],size=self.size,string=termString,variables=self.variables)
                        return Constraint('lin>',None,self,exp2)

        def __eq__(self,exp):
                if isinstance(exp,AffinExp):
                        if exp.size==(1,1) and self.size<>(1,1):
                                oldstring=exp.string
                                exp=cvx.matrix(1.,self.size)*exp.diag(self.size[1])
                                exp.string='|'+oldstring+'|'
                        if self.size==(1,1) and exp.size<>(1,1):
                                oldstring=self.string
                                selfone=cvx.matrix(1.,exp.size)*self.diag(exp.size[1])
                                selfone.string='|'+oldstring+'|'
                                return (selfone==exp)
                        return Constraint('lin=',None,self,exp)
                else:                   
                        term,termString=_retrieve_matrix(exp,self.size)
                        exp2=AffinExp(factors={},constant=term[:],size=self.size,string=termString,variables=self.variables)
                        return Constraint('lin=',None,self,exp2)

        def __abs__(self):
                return Norm(self)

        def __pow__(self,exponent):
                if (self.size==(1,1) and self.isconstant()):
                        return AffinExp(factors={},constant=self.eval()[0]**exponent,
                                size=(1,1),string='('+self.string+')**2',variables=self.variables)
                if (exponent<>2 or self.size<>(1,1)):
                        raise Exception('not implemented')
                Q=QuadExp({},
                        AffinExp(variables=self.variables),
                        None,None,variables=self.variables)
                qq = self *self
                Q.quad= qq.quad.copy()
                Q.LR=(self,None)
                if ('*' in self.affstring()) or ('+' in self.affstring()) or (
                        '-' in self.affstring()) or ('/' in self.affstring()):
                        Q.string= '('+self.affstring()+')**2'
                else:
                        Q.string= self.affstring()+'**2'
                return Q

        def diag(self,dim):
                if self.size<>(1,1):
                        raise Exception('not implemented')
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,
                                self.string,variables=self.variables)
                idx=cvx.spdiag([1.]*dim)[:].I
                for k in self.factors.keys():
                        selfcopy.factors[k]=cvx.spmatrix([],[],[],(dim**2,self.factors[k].size[1]))
                        for i in idx:
                                selfcopy.factors[k][i,:]=self.factors[k]
                selfcopy.constant=cvx.matrix(0.,(dim**2,1))
                if not self.constant is None:           
                        for i in idx:
                                selfcopy.constant[i]=self.constant[0]
                selfcopy.size=(dim,dim)
                selfcopy.string='diag('+selfcopy.string+')'
                return selfcopy

        def __and__(self,exp):
                """horizontal concatenation"""
                selfcopy=AffinExp(self.factors.copy(),self.constant,self.size,self.string,variables=self.variables)
                if isinstance(exp,AffinExp):
                        if exp.size[0]<>selfcopy.size[0]:
                                raise Exception('incompatible size for concatenation')
                        for k in list(set(exp.factors.keys()).union(set(selfcopy.factors.keys()))):
                                if (k in selfcopy.factors.keys()) and (k in exp.factors.keys()):
                                        newfac=cvx.sparse([[selfcopy.factors[k],exp.factors[k]]])
                                        selfcopy.factors[k]=newfac
                                elif k in exp.factors.keys():
                                        s1=selfcopy.size[0]*selfcopy.size[1]
                                        s2=exp.factors[k].size[1]
                                        newfac=cvx.sparse([[cvx.spmatrix([],[],[],(s1,s2)),
                                                        exp.factors[k]]])
                                        selfcopy.factors[k]=newfac
                                else:
                                        s1=exp.size[0]*exp.size[1]
                                        s2=selfcopy.factors[k].size[1]
                                        newfac=cvx.sparse([[selfcopy.factors[k],
                                                cvx.spmatrix([],[],[],(s1,s2))]])
                                        selfcopy.factors[k]=newfac
                        if selfcopy.constant is None and exp.constant is None:
                                pass
                        else:
                                s1=selfcopy.size[0]*selfcopy.size[1]
                                s2=exp.size[0]*exp.size[1]
                                if not selfcopy.constant is None:
                                        newCons=selfcopy.constant
                                else:
                                        newCons=cvx.spmatrix([],[],[],(s1,1))
                                if not exp.constant is None:
                                        newCons=cvx.sparse([[newCons,exp.constant]])
                                else:
                                        newCons=cvx.sparse([[newCons,cvx.spmatrix([],[],[],(s2,1))]])
                                selfcopy.constant=newCons
                        selfcopy.size=(exp.size[0],exp.size[1]+selfcopy.size[1])
                        sstring=selfcopy.string
                        estring=exp.string
                        if sstring[0]=='[' and sstring[-1]==']':
                                sstring=sstring[1:-1]
                        if estring[0]=='[' and estring[-1]==']':
                                estring=estring[1:-1]
                        selfcopy.string='['+sstring+','+estring+']'
                        return selfcopy
                else:
                        Exp,ExpString=_retrieve_matrix(exp,self.size[0])
                        exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                        return (self & exp2)

        def __rand__(self,exp):
                Exp,ExpString=_retrieve_matrix(exp,self.size[0])
                exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                return (exp2 & self)
                        
        def __floordiv__(self,exp):
                """vertical concatenation"""
                if isinstance(exp,AffinExp):
                        concat=(self.T & exp.T).T
                        concat.size=(exp.size[0]+self.size[0],exp.size[1])
                        sstring=self.string
                        estring=exp.string
                        if sstring[0]=='[' and sstring[-1]==']':
                                sstring=sstring[1:-1]
                        if estring[0]=='[' and estring[-1]==']':
                                estring=estring[1:-1]
                        concat.string='['+sstring+';'+estring+']'
                        return concat
                else:
                        Exp,ExpString=_retrieve_matrix(exp,self.size[1])
                        exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                        return (self // exp2)

        def __rfloordiv__(self,exp):
                Exp,ExpString=_retrieve_matrix(exp,self.size[1])
                exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                return (exp2 // self)

        def apply_function(self,fun):
                return GeneralFun(fun,self,fun())
        
        def __lshift__(self,exp):
                if self.size[0]<>self.size[1]:
                        raise Exception('both sides of << must be square')
                if isinstance(exp,AffinExp):
                        return Constraint('sdp<',None,self,exp)
                else:
                       n=self.size[0]
                       Exp,ExpString=_retrieve_matrix(exp,(n,n))
                       exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                       return (self << exp2)
                       
        def __rshift__(self,exp):
                if self.size[0]<>self.size[1]:
                        raise Exception('both sides of << must be square')
                if isinstance(exp,AffinExp):
                        return Constraint('sdp>',None,self,exp)
                else:
                       n=self.size[0]
                       Exp,ExpString=_retrieve_matrix(exp,(n,n))
                       exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString,variables=self.variables)
                       return (self >> exp2)

#---------------------------------------------
#        Class Norm and ProductOfAffinExp  
#---------------------------------------------

class Norm(Expression):
        def __init__(self,exp):
                Expression.__init__(self,'||'+exp.string+'||',exp.variables)
                self.exp=exp
        def __repr__(self):
                normstr='# norm of a ({0} x {1})- expression: ||'.format(self.exp.size[0],
                                                                self.exp.size[1])
                normstr+=self.exp.affstring()
                normstr+='||'
                normstr+=' #'
                return normstr
        
        def __str__(self):
                if self.is_valued():
                        return str(self.eval())
                else:
                        return repr(self)
                
                
        def eval(self, ind=None):
                vec=self.exp.eval(ind)
                return np.linalg.norm(vec)
             
             
        value = property(eval,Expression.set_value,Expression.del_simple_var_value,"value of the Norm expression")
        """
        sets a variable to the given value. This can be useful to check
        the value of a complicated :class:`Expression <picos.Expression>`, or to use
        a solver with a *hot start* option.
        
        :param name: name of the variable to which the value will be given
        :type name: str.
        :param value: The value for the variable. The function will try to
                        parse this variable as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`
                        of the desired size by using
                        the function :func:`_retrieve_matrix() <picos.tools._retrieve_matrix>`
                        
        **Example**
        
        >>> prob=pic.Problem()
        >>> x=prob.add_variable('x',2)
        >>> prob.set_var_value('x',[3,4])
        >>> abs(x)**2
        #quadratic expression: ||x||**2 #
        >>> (abs(x)**2).eval()
        25.0
        >>> 

        .. Todo::
        Check if doc displayed. Ajouter option hotstart
        """
                
        def __pow__(self,exponent):
                if (exponent<>2):
                        raise Exception('not implemented')
                if self.exp.isconstant():
                        Qnorm=QuadExp({},
                                AffinExp(factors={},constant=self.exp.eval(),size=(1,1),string='  ',variables=self.variables),
                                string='  ',
                                variables=self.variables)
                else:
                        Qnorm=QuadExp({},
                        AffinExp(variables=self.variables),
                        None,None,
                        variables=self.variables)
                        #Qnorm=(self.exp.T)*(self.exp)
                qq = (self.exp.T)*(self.exp)
                if isinstance(qq,AffinExp):
                        qq=QuadExp({},qq,qq.string,variables={})
                Qnorm.quad = qq.quad.copy()
                Qnorm.LR=(self.exp,None)
                #if self.exp.size<>(1,1):
                Qnorm.string='||'+self.exp.affstring()+'||**2'
                #else:
                #       Qnorm.string='('+self.exp.affstring()+')**2'
                return Qnorm

        def __lt__(self,exp):
                if isinstance(exp,AffinExp):
                        if self.exp.size<>(1,1):
                                return Constraint('SOcone',None,self.exp,exp)
                        else:
                                cons = (self.exp // -self.exp) < (exp // exp)
                                if exp.is1():
                                        cons.myconstring= '||'+self.exp.string+'|| < 1'
                                else:
                                        cons.myconstring= '||'+self.exp.string+'|| < '+exp.string
                                cons.myfullconstring='# (1x1)-SOC constraint '+cons.myconstring+' #'
                                return cons
                else:#constant          
                        term,termString=_retrieve_matrix(exp,(1,1))
                        exp1=AffinExp(factors={},constant=term,size=(1,1),string=termString,variables=self.variables)
                        return self<exp1

class LogSumExp(Expression):
        def __init__(self,exp):
                
                if not( isinstance(exp,AffinExp)):
                        term,termString=_retrieve_matrix(exp,None)
                        exp=AffinExp(factors={},constant=term,
                                        size=term.size,string=termString,
                                        variables={})
                               
                Expression.__init__(self,'LSE['+exp.string+']',exp.variables)
                self.Exp=exp
        
        def __str__(self):
                if self.is_valued():
                        return str(self.value)
                else:
                        return repr(self)
                
        def __repr__(self):
                lsestr='# log-sum-exp of an affine expression: '
                lsestr+=self.Exp.affstring()
                lsestr+=' #'
                return lsestr

        def affstring(self):
                return 'LSE['+self.Exp.affstring()+']'

        def eval(self, ind=None):
                return np.log(np.sum(np.exp(self.Exp.eval(ind))))
                
        value = property(eval,Expression.set_value,Expression.del_simple_var_value,"value of the logsumexp expression")

        def __lt__(self,exp):
                if exp<>0:
                        raise Exception('lhs must be 0')
                else:
                        return Constraint('lse',None,self.Exp,0)

class QuadExp(Expression):
        """
        quad are the quadratic factors,
        aff is the affine part of the expression,
        string is a string,
        and LR stores a factorization of the expression for norms (||x||**2 -> LR=(x,None))
                                                                and product of scalar expressions)
                                                                
        .. Todo:: quad expression of dimension>1                                                        
        """
        def __init__(self,quad,aff,string,LR=None,variables=None):
                Expression.__init__(self,string,variables)
                self.quad=quad
                self.aff=aff
                #self.string=string
                self.LR=LR

        def __str__(self):
                if self.is_valued():
                        return str(self.value)
                else:
                        return repr(self)

        def __repr__(self):
                return '#quadratic expression: '+self.string+' #'

        def eval(self, ind=None):
                if not self.aff is None:
                        val=self.aff.eval(ind)
                else:
                        val=cvx.matrix(0.,(1,1))
                if not self.LR is None:
                        ex1=self.LR[0].eval(ind)
                        if self.LR[1] is None:
                                val+=(ex1.T*ex1)
                        else:
                                if self.LR[0].size!=(1,1) or self.LR[1].size!=(1,1):
                                        raise Exception('QuadExp of size (1,1) only are implemented')
                                else:
                                        ex2=self.LR[1].eval(ind)
                                        val+=(ex1*ex2)
                                
                                
                elif not self.quad is None:
                        for i,j in self.quad:
                                if not i in self.variables:
                                        raise Exception(i+' is unknown')
                                if not j in self.variables:
                                        raise Exception(j+' is unknown')
                                if ind is None:
                                        if self.variables[i].value is None:
                                                raise Exception(i+' is not valued')
                                        if self.variables[j].value is None:
                                                raise Exception(j+' is not valued')
                                        xi=self.variables[i].value[:]
                                        xj=self.variables[j].value[:]
                                else:
                                        if ind not in self.variables[i].value_alt:
                                                raise Exception(i+' does not have a value for the index '+str(ind))
                                        if ind not in self.variables[j].value_alt:
                                                raise Exception(j+' does not have a value for the index '+str(ind))
                                        xi=self.variables[i].value_alt[ind][:]
                                        xj=self.variables[j].value_alt[ind][:]
                                val=val+xi.T*self.quad[i,j]*xj

                
                return val[0]
       
        value = property(eval,Expression.set_value,Expression.del_simple_var_value,"value of the affine expression")
       
        def nnz(self):
                nz=0
                for ij in self.quad:
                        nz+=len(self.quad[ij].I)
                return nz

        #OVERLOADS:
        #division par un scalaire

        def __mul__(self,fact):
                if isinstance(fact,AffinExp):
                        if fact.isconstant() and fact.size==(1,1):
                                import copy
                                selfcopy=QuadExp(copy.deepcopy(self.quad),copy.deepcopy(self.aff),self.string,variables=self.variables)
                                for ij in self.quad:
                                        selfcopy.quad[ij]=fact.eval()[0]*selfcopy.quad[ij]
                                selfcopy.aff=fact*selfcopy.aff
                                selfcopy.string=fact.affstring()+'*('+self.string+')'
                                if not self.LR is None:
                                        if self.LR[1] is None and (fact.eval()[0]>=0): #Norm squared
                                                selfcopy.LR=(np.sqrt(fact.eval())*self.LR[0],None)
                                        elif self.LR[1] is None and (fact.eval()[0]<0):
                                                selfcopy.LR=None
                                        else:
                                                selfcopy.LR=(fact*self.LR[0],self.LR[1])
                                return selfcopy
                        else:
                                raise Exception('not implemented')
                else: #constant term
                        fact,factString=_retrieve_matrix(fact,(1,1))
                        return self*AffinExp({},constant=fact[:],size=fact.size,string=factString,variables=self.variables)

        def __add__(self,term):
                if isinstance(term,QuadExp):
                        import copy
                        selfcopy=QuadExp(copy.deepcopy(self.quad),copy.deepcopy(self.aff),self.string,variables=self.variables)
                        for ij in self.quad:
                                if ij in term.quad.keys():
                                        selfcopy.quad[ij]=selfcopy.quad[ij]+term.quad[ij]
                        for ij in term.quad:
                                if not (ij in self.quad):
                                        selfcopy.quad[ij]=term.quad[ij]
                        selfcopy.aff=selfcopy.aff+term.aff
                        selfcopy.LR=None
                        if term.string not in ['0','']:
                                if term.string[0]=='-':
                                        import re                                       
                                        if ('+' not in term.string[1:]) and (
                                                '-' not in term.string[1:]):
                                                selfcopy.string=selfcopy.string+' '+term.string
                                        elif (term.string[1]=='(') and (
                                                 re.search('.*\)((\[.*\])|(.T))*$',term.string) ):                                                              #a group in a (...)
                                                selfcopy.string=selfcopy.string+' '+term.string
                                        else:
                                                selfcopy.string=selfcopy.string+' + ('+ \
                                                                term.string+')'
                                else:
                                        selfcopy.string+=' + '+term.string
                        return selfcopy
                elif isinstance(term,AffinExp):
                        if term.size<>(1,1):
                                raise Exception('RHS must be scalar')
                        expQE=QuadExp({},term,term.affstring(),variables=self.variables)
                        return self+expQE
                else:
                        term,termString=_retrieve_matrix(term,(1,1))
                        expAE=AffinExp(factors={},constant=term,size=term.size,string=termString,variables=self.variables)
                        return self+expAE

        def __rmul__(self,fact):
                return self*fact

        def __neg__(self):
                selfneg = (-1)*self
                if self.string[0]=='-':
                        import re
                        if ('+' not in self.string[1:]) and ('-' not in self.string[1:]):
                                selfneg.string=self.string[1:]
                        elif (self.string[1]=='(') and (
                           re.search('.*\)((\[.*\])|(.T))*$',self.string) ): #a group in a (...)
                                if self.string[-1]==')':
                                        selfneg.string=self.string[2:-1] #we remove the parenthesis
                                else:
                                        selfneg.string=self.string[1:] #we keep the parenthesis
                        else:
                                selfneg.string='-('+self.string+')'
                else:
                        if ('+' in self.string) or ('-' in self.string):
                                selfneg.string='-('+self.string+')'
                        else:
                                selfneg.string='-'+self.string
                return selfneg

        def __sub__(self,term):
                return self+(-term)

        def __rsub__(self,term):
                return term+(-self)

        def __radd__(self,term):
                return self+term

        def __lt__(self,exp):
                if isinstance(exp,QuadExp):             
                        if ((not self.LR is None) and (self.LR[1] is None)
                                and (not exp.LR is None) and (not exp.LR[1] is None)
                        ): #SOCP constraint
                                return Constraint('RScone',None,self.LR[0],exp.LR[0],exp.LR[1])
                        else:
                                return Constraint('quad',None,self-exp,0)
                if isinstance(exp,AffinExp):
                        if exp.size<>(1,1):
                                raise Exception('RHS must be scalar')
                        exp2=AffinExp(factors={},constant=cvx.matrix(1.,(1,1)),size=(1,1),string='1',variables=self.variables)
                        expQE=QuadExp({},exp,exp.affstring(),LR=(exp,exp2),variables=self.variables)
                        return self<expQE
                else:
                        term,termString=_retrieve_matrix(exp,(1,1))
                        expAE=AffinExp(factors={},constant=term,size=(1,1),string=termString,variables=self.variables)
                        return self<expAE

        def __gt__(self,exp):
                if isinstance(exp,QuadExp):
                        if (not exp.LR is None) and (exp.LR[1] is None): # a squared norm
                                return exp<self
                        return (-self)<(-exp)
                if isinstance(exp,AffinExp):
                        if exp.size<>(1,1):
                                raise Exception('RHS must be scalar')
                        if exp.isconstant():
                                cst=AffinExp( factors={},constant=cvx.matrix(np.sqrt(exp.eval()),(1,1)),
                                        size=(1,1),string=exp.string,variables=self.variables)
                                return (Norm(cst)**2)<self
                        else:
                                return (-self)<(-exp)
                else:
                        term,termString=_retrieve_matrix(exp,(1,1))
                        expAE=AffinExp(factors={},constant=term,size=(1,1),string=termString,variables=self.variables)
                        return self>expAE
                                

class GeneralFun(Expression):
        """a class storing a general scalar function,
                applied to an affine expression"""
        def __init__(self,fun,Exp,funstring):
                Expression.__init__(self,self.funstring+'( '+Exp.string+')',
                                        Exp.variables)
                self.fun=fun
                self.Exp=Exp
                self.funstring=funstring
                self.string=self.funstring+'( '+self.Exp.affstring()+' )'

        def __repr__(self):
                return '# general function '+self.string+' #'
         
        def __str__(self):
                if self.is_valued():
                        return str(self.value)
                else:
                        return repr(self)

        def eval(self,ind=None):
                val=self.Exp.eval(ind)
                o,g,h=self.fun(val)
                return o
                
        value = property(eval,Expression.set_value,Expression.del_simple_var_value,"value of the affine expression")
