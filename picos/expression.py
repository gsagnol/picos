# coding: utf-8

#-------------------------------------------------------------------
#Picos 0.1 : A pyton Interface To Conic Optimization Solvers
#Copyright (C) 2012  Guillaume Sagnol
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#For any information, please contact:
#Guillaume Sagnol
#sagnol@zib.de
#Konrad Zuse Zentrum für Informationstechnik Berlin (ZIB)
#Takustrasse 7
#D-14195 Berlin-Dahlem
#Germany 
#-------------------------------------------------------------------

import cvxopt as cvx
import numpy as np
import sys

from .tools import *
from .constraint import *

__all__=['Expression',
        'AffinExp',
        'Norm',
        'QuadExp',
        'GeneralFun',
        'LogSumExp',
        'Variable'
        ]
       
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

        def __init__(self,string):
                self.string=string
                """String representation of the expression"""


                
        def eval(self):
                pass
        
        def set_value(self,value):
                raise ValueError('set_value can only be called on a Variable')
                
        def del_simple_var_value(self):
                raise ValueError('del_simple_var_value can only be called on a Variable')
                                
        value = property(eval,set_value,del_simple_var_value,"value of the affine expression")
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
                        val=self.value
                        return not(val is None)
                except Exception:
                        return False
                        
                
#----------------------------------
#                 AffinExp class
#----------------------------------

class AffinExp(Expression):
        """A class for defining vectorial (or matrix) affine expressions.
        It derives from :class:`Expression<picos.Expression>`.
        
        **Overloaded operators**
        
                :``+``: sum            (with an affine or quadratic expression)
                :``-``: substraction   (with an affine or quadratic expression) or unitary minus
                :``*``: multiplication (by another affine expression or a scalar)
                :``/``: division       (by a scalar)
                :``|``: scalar product (with another affine expression)
                :``[.]``: slice of an affine expression
                :``abs()``: Euclidean norm (Frobenius norm for matrices)
                :``**``: exponentiation (works with arbitrary powers for constant
                           affine expressions, and only with the exponent 2
                           if the affine expression involves some variables)
                :``&``: horizontal concatenation (with another affine expression)
                :``//``: vertical concatenation (with another affine expression)
                :``<``: less **or equal** (than an affine or quadratic expression)
                :``>``: greater **or equal** (than an affine or quadratic expression)
                :``==``: is equal (to another affine expression)
                :``<<``: less than inequality in the Loewner ordering (linear matrix inequality)
                :``>>``: greater than inequality in the Loewner ordering (linear matrix inequality)
                
        """
        
        def __init__(self,factors=None,constant=None,
                        size=(1,1),
                        string='0'
                        ):
                if factors is None:
                        factors={}
                Expression.__init__(self,string)
                self.factors=factors
                """
                dictionary storing the matrix of coefficients of the linear
                part of the affine expressions. The matrices of coefficients
                are always stored with respect to vectorized variables (for the
                case of matrix variables), and are indexed by instances
                of the class :class:`Variable<picos.Variable>`.
                """
                self.constant=constant
                """constant of the affine expression,
                stored as a :func:`cvxopt sparse matrix <cvxopt:cvxopt.spmatrix>`.
                """
                self.size=size
                """size of the affine expression"""
                #self.string=string
                
        def __str__(self):
                if self.is_valued():
                        if self.size==(1,1):
                                return str(self.value[0])
                        else:
                                return str(self.value)
                else:
                        return repr(self)

        def __repr__(self):
                affstr='# ({0} x {1})-affine expression: '.format(self.size[0],
                                                                self.size[1])
                affstr+=self.affstring()
                affstr+=' #'
                return affstr
                        
        def copy(self):
                import copy
                facopy={}
                for f,m in self.factors.iteritems(): #copy matrices but not the variables (keys of the dict)
                        facopy[f]=copy.deepcopy(m)
        
                conscopy=copy.deepcopy(self.constant)
                return AffinExp(facopy,conscopy,self.size,self.string)

        def affstring(self):
                return self.string

        def eval(self,ind=None):
                if self.constant is None:
                        val=cvx.spmatrix([],[],[],(self.size[0]*self.size[1],1))
                else:
                        val=self.constant

                for k in self.factors:
                        if ind is None:
                                if not k.value is None:
                                        if k.vtype=='symmetric':
                                                val=val+self.factors[k]*svec(k.value)
                                        else:
                                                val=val+self.factors[k]*k.value[:]
                                else:
                                        raise Exception(k+' is not valued')
                        else:
                                if ind in k.value_alt:
                                        if k.vtype=='symmetric':
                                                val=val+self.factors[k]*svec(k.value_alt[ind])
                                        else:
                                                val=val+self.factors[k]*k.value_alt[ind][:]
                                else:
                                        raise Exception(k+' does not have a value for the index '+str(ind))
                return cvx.matrix(val,self.size)
                
        def set_value(self,value):
                raise ValueError('set_value can only be called on a Variable')
                
        def del_simple_var_value(self):
                raise ValueError('del_simple_var_value can only be called on a Variable')
        
        value = property(eval,set_value,del_simple_var_value,"value of the affine expression")

           
        
        def is_valued(self, ind = None):
                
                try:
                        for k in self.factors:
                                if ind is None:
                                        if k.value is None:
                                                return False
                                else:
                                        if ind  not in k.value_alt:
                                                return False
                except:
                        return False

                #Yes, you can call eval(ind) without any problem.
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
                if not(self.size==(1,1) and self.constant[0]==1):
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
                if self.is_valued() and self.size==(1,1) and int(self.value[0])==self.value[0]:
                        return int(self.value[0])
                else:
                        raise Exception('unexpected index (nonvalued, multidimensional, or noninteger)')
                
        def transpose(self):
                selfcopy=self.copy()
                
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
        """Transposition"""

        def __rmul__(self,fact):
                selfcopy=self.copy()
                
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=cvx.sparse(fact.eval()),fact.string
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
                sstring=selfcopy.affstring()
                if len(facString)>0:
                        if ('+' in sstring) or ('-' in sstring):
                                sstring='( '+sstring+' )'
                        if ('+' in facString) or ('-' in facString):
                                facString='( '+facString+' )'
                                
                        selfcopy.string=facString+'*'+sstring

                return selfcopy


        
        def __mul__(self,fact):
                """product of 2 affine expressions"""
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=cvx.sparse(fact.eval()),fact.string           
                        elif self.isconstant():
                                return fact.__rmul__(self)
                        elif self.size[0]==1 and fact.size[1]==1 and self.size[1]==fact.size[0]:
                                #quadratic expression
                                linpart=AffinExp({},constant=None,size=(1,1))
                                if not self.constant is None:
                                        linpart=linpart+self.constant.T*fact
                                if not fact.constant is None:
                                        linpart=linpart+self*fact.constant
                                if not ((fact.constant is None) or (self.constant is None)):
                                        linpart=linpart-self.constant.T*fact.constant
                                """if not ( self.constant is None or fact.constant is None):
                                        linpart.constant=self.constant.T*fact.constant
                                if not fact.constant is None:                           
                                        for k in self.factors:
                                                linpart.factors[k]=fact.constant.T*self.factors[k]
                                if not self.constant is None:
                                        for k in fact.factors:
                                                if k in linpart.factors:
                                                        linpart.factors[k]+=self.constant.T*fact.factors[k]
                                                else:
                                                        linpart.factors[k]=self.constant.T*fact.factors[k]
                                """
                                quadpart={}
                                for i in self.factors:
                                        for j in fact.factors:
                                                quadpart[i,j]=self.factors[i].T*fact.factors[j]
                                stleft=self.affstring()
                                stright=fact.affstring()
                                if ('+' in stleft) or ('-' in stleft):
                                        if len(stleft)>3 and not(stleft[0]=='(' and stleft[-3:]==').T'):
                                                stleft='( '+stleft+' )'
                                if ('+' in stright) or ('-' in stright):
                                        stright='( '+stright+' )'                               
                                if self.size[1]==1:
                                        return QuadExp(quadpart,linpart,stleft+'*'+stright,LR=(self,fact))
                                else:
                                        return QuadExp(quadpart,linpart,stleft+'*'+stright)
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
                selfcopy=self.copy()
                
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
                sstring=selfcopy.affstring()
                if len(facString)>0:
                        if ('+' in sstring) or ('-' in sstring):
                                sstring='( '+sstring+' )'
                        if ('+' in facString) or ('-' in facString):
                                facString='( '+facString+' )'
                        prod.string=sstring+'*'+facString
                else:
                        prod.string=selfcopy.string
                return prod
        
        def __or__(self,fact):#scalar product
                selfcopy=self.copy()
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=cvx.sparse(fact.constant),fact.string
                        elif self.isconstant():
                                return fact.__ror__(self)       
                        else:
                                dotp = self[:].T*fact[:]
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
                selfcopy=self.copy()
                if isinstance(fact,AffinExp):
                        if fact.isconstant():
                                fac,facString=cvx.sparse(fact.eval()),fact.string
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
                selfcopy=self.copy()
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
                        for k in term.factors:
                                if k in selfcopy.factors:
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
                        expQE=QuadExp({},self,self.affstring())
                        return expQE+term
                else: #constant term
                        term,termString=_retrieve_matrix(term,selfcopy.size)
                        return self+AffinExp({},constant=term[:],size=term.size,string=termString)

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
                        return self-AffinExp({},constant=term[:],size=term.size,string=termString)

        def __rsub__(self,term):
                return term+(-self)

        def __div__(self,divisor): #division (by a scalar)
                if isinstance(divisor,AffinExp):
                        if divisor.isconstant():
                                divi,diviString=divisor.value,divisor.string
                        else:
                                raise Exception('not implemented')
                        if divi.size<>(1,1):
                                raise Exception('not implemented')
                        divi=divi[0]
                        if divi==0:
                                raise Exception('Division By Zero')
                        division=self * (1/divi)
                        if ('+' in self.string) or ('-' in self.string):
                                division.string = '('+ self.string + ') /' + diviString
                        else:
                                division.string =  self.string+ ' / ' + diviString
                        return division
                else : #constant term
                        divi,diviString=_retrieve_matrix(divisor,(1,1))
                        return self/AffinExp({},constant=divi[:],size=(1,1),string=diviString)

        def __rdiv__(self,divider):
                divi,diviString=_retrieve_matrix(divider,None)
                return AffinExp({},constant=divi[:],size=divi.size,string=diviString)/self
                                                

        def __getitem__(self,index):
                selfcopy=self.copy()
                def indexstr(idx):
                        if isinstance(idx,int):
                                return str(idx)
                        elif isinstance(idx,Expression):
                                return idx.string
                def slicestr(sli):
                        #single element
                        if not (sli.start is None or sli.stop is None):
                                sta=sli.start
                                sto=sli.stop
                                if isinstance(sta,int):
                                        sta=new_param(str(sta),sta)
                                if isinstance(sto,int):
                                        sto=new_param(str(sto),sto)
                                if (sto.__index__()==sta.__index__()+1):
                                        return sta.string
                        #single element -1 (Expression)
                        if (isinstance(sli.start,Expression) and sli.start.__index__()==-1
                           and sli.stop is None and sli.step is None):
                                return sli.start.string
                        #single element -1
                        if (isinstance(sli.start,int) and sli.start==-1
                           and sli.stop is None and sli.step is None):
                                return '-1'
                        ss=''
                        if not sli.start is None:
                                ss+=indexstr(sli.start)
                        ss+=':'
                        if not sli.stop is None:
                                ss+=indexstr(sli.stop)
                        if not sli.step is None:
                                ss+=':'
                                ss+=indexstr(sli.step)
                        return ss
                
                if isinstance(index,Expression):
                        if index.__index__()==-1:#(-1,0) does not work
                                index=slice(index,None,None)
                        else:
                                index=slice(index,index+1,None)                                
                elif isinstance(index,int):
                        if index==-1: #(-1,0) does not work
                                index=slice(index,None,None)
                        else:
                                index=slice(index,index+1,None)
                if isinstance(index,slice):
                        idx=index.indices(self.size[0]*self.size[1])
                        rangeT=range(idx[0],idx[1],idx[2])
                        for k in selfcopy.factors:
                                selfcopy.factors[k]=selfcopy.factors[k][rangeT,:]
                        if not selfcopy.constant is None:
                                selfcopy.constant=selfcopy.constant[rangeT]
                        selfcopy.size=(len(rangeT),1)
                        indstr=slicestr(index)
                elif isinstance(index,tuple):
                        if isinstance(index[0],Expression):
                                if index[0].__index__()==-1:
                                        index=(slice(index[0],None,None),index[1])
                                else:
                                        index=(slice(index[0],index[0]+1,None),index[1])
                        elif isinstance(index[0],int):
                                if index[0]==-1:
                                        index=(slice(index[0],None,None),index[1])
                                else:
                                        index=(slice(index[0],index[0]+1,None),index[1])
                        if isinstance(index[1],Expression):
                                if index[1].__index__()==-1:
                                        index=(index[0],slice(index[1],None,None))
                                else:
                                        index=(index[0],slice(index[1],index[1]+1,None))
                        elif isinstance(index[1],int):
                                if index[1]==-1:
                                        index=(index[0],slice(index[1],None,None))
                                else:
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
                        for k in selfcopy.factors:
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
                                        size=(1,1),string=self.string)
                                return (Norm(cst)**2)<exp
                        elif self.size==(1,1):
                                return (-exp)<(-self)
                        else:
                                raise Exception('not implemented')
                else:                   
                        term,termString=_retrieve_matrix(exp,self.size)
                        exp2=AffinExp(factors={},constant=term[:],size=self.size,string=termString)
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
                        exp2=AffinExp(factors={},constant=term[:],size=self.size,string=termString)
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
                        exp2=AffinExp(factors={},constant=term[:],size=self.size,string=termString)
                        return Constraint('lin=',None,self,exp2)

        def __abs__(self):
                return Norm(self)

        def __pow__(self,exponent):
                if (self.size==(1,1) and self.isconstant()):
                        if (isinstance(exponent,AffinExp) and exponent.isconstant()):
                                exponent = exponent.value[0]
                        if isinstance(exponent,int) or isinstance(exponent,float):
                                return AffinExp(factors={},constant=self.eval()[0]**exponent,
                                        size=(1,1),string='('+self.string+')**{0}'.format(exponent))
                        else:
                                raise Exception('type of exponent not handled')
                if (exponent<>2 or self.size<>(1,1)):
                        raise Exception('not implemented')
                Q=QuadExp({},
                        AffinExp(),
                        None,None)
                qq = self *self
                Q.quad= qq.quad
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
                selfcopy=self.copy()
                idx=cvx.spdiag([1.]*dim)[:].I
                for k in self.factors:
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
                selfcopy=self.copy()
                if isinstance(exp,AffinExp):
                        if exp.size[0]<>selfcopy.size[0]:
                                raise Exception('incompatible size for concatenation')
                        for k in list(set(exp.factors.keys()).union(set(selfcopy.factors.keys()))):
                                if (k in selfcopy.factors) and (k in exp.factors):
                                        newfac=cvx.sparse([[selfcopy.factors[k],exp.factors[k]]])
                                        selfcopy.factors[k]=newfac
                                elif k in exp.factors:
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
                        exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString)
                        return (self & exp2)

        def __rand__(self,exp):
                Exp,ExpString=_retrieve_matrix(exp,self.size[0])
                exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString)
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
                        exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString)
                        return (self // exp2)

        def __rfloordiv__(self,exp):
                Exp,ExpString=_retrieve_matrix(exp,self.size[1])
                exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString)
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
                       exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString)
                       return (self << exp2)
                       
        def __rshift__(self,exp):
                if self.size[0]<>self.size[1]:
                        raise Exception('both sides of << must be square')
                if isinstance(exp,AffinExp):
                        return Constraint('sdp>',None,self,exp)
                else:
                       n=self.size[0]
                       Exp,ExpString=_retrieve_matrix(exp,(n,n))
                       exp2=AffinExp(factors={},constant=Exp[:],size=Exp.size,string=ExpString)
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
        def __init__(self,exp):
                Expression.__init__(self,'||'+exp.string+'||')
                self.exp=exp
                """The affine expression of which we take the norm"""
        def __repr__(self):
                normstr='# norm of a ({0} x {1})- expression: ||'.format(self.exp.size[0],
                                                                self.exp.size[1])
                normstr+=self.exp.affstring()
                normstr+='||'
                normstr+=' #'
                return normstr
        
        def __str__(self):
                if self.is_valued():
                        return str(self.value[0])
                else:
                        return repr(self)
                
                
        def eval(self, ind=None):
                vec=self.exp.eval(ind)
                return cvx.matrix(np.linalg.norm(vec),(1,1))
             
        
        value = property(eval,Expression.set_value,Expression.del_simple_var_value)

                
        def __pow__(self,exponent):
                if (exponent<>2):
                        raise Exception('not implemented')
                
                qq = (self.exp[:].T)*(self.exp[:])
                if isinstance(qq,AffinExp):
                        qq=QuadExp({},qq,qq.string)
                Qnorm=QuadExp(qq.quad,
                              qq.aff,
                              '||'+self.exp.affstring()+'||**2',
                              LR=(self.exp,None)
                              )
                
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
                        exp1=AffinExp(factors={},constant=term,size=(1,1),string=termString)
                        return self<exp1

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
        def __init__(self,exp):
                
                if not( isinstance(exp,AffinExp)):
                        term,termString=_retrieve_matrix(exp,None)
                        exp=AffinExp(factors={},constant=term,
                                        size=term.size,string=termString)
                               
                Expression.__init__(self,'LSE['+exp.string+']')
                self.Exp=exp
        
        def __str__(self):
                if self.is_valued():
                        return str(self.value[0])
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
                return cvx.matrix(np.log(np.sum(np.exp(self.Exp.eval(ind)))),
                                  (1,1)
                                  )
                
        value = property(eval,Expression.set_value,Expression.del_simple_var_value,"value of the logsumexp expression")

        def __lt__(self,exp):
                if exp<>0 and not(isinstance(exp,AffinExp) and exp.is0()):
                        raise Exception('rhs must be 0')
                else:
                        return Constraint('lse',None,self.Exp,0)

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
        def __init__(self,quad,aff,string,LR=None):
                Expression.__init__(self,string)
                self.quad=quad
                """dictionary of quadratic forms,
                stored as matrices representing bilinear forms
                with respect to two vectorized variables, 
                and indexed by tuples of 
                instances of the class :class:`Variable<picos.Variable>`.
                """
                self.aff=aff
                """affine expression representing the affine part of the quadratic expression""" 
                self.LR=LR
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
                return '#quadratic expression: '+self.string+' #'

        def copy(self):
                import copy
                qdcopy={}
                for ij,m in self.quad.iteritems():
                        qdcopy[ij]=copy.deepcopy(m)

                if self.aff is None:
                        affcopy=None
                else:
                        affcopy=self.aff.copy()
                       
                if self.LR is None:
                        lrcopy=None
                else:
                        if self.LR[1] is None:
                                lrcopy=(self.LR[0].copy(),None)
                        else:
                                lrcopy=(self.LR[0].copy(),self.LR[1].copy())
                return QuadExp(qdcopy,affcopy,self.string,lrcopy)
                        
                
        def eval(self, ind=None):
                if not self.LR is None:
                        ex1=self.LR[0].eval(ind)
                        if self.LR[1] is None:
                                val=(ex1.T*ex1)
                        else:
                                if self.LR[0].size!=(1,1) or self.LR[1].size!=(1,1):
                                        raise Exception('QuadExp of size (1,1) only are implemented')
                                else:
                                        ex2=self.LR[1].eval(ind)
                                        val=(ex1*ex2)

                else:
                        if not self.aff is None:
                                val=self.aff.eval(ind)
                        else:
                                val=cvx.matrix(0.,(1,1))
                                
                        for i,j in self.quad:
                                if ind is None:
                                        if i.value is None:
                                                raise Exception(i+' is not valued')
                                        if j.value is None:
                                                raise Exception(j+' is not valued')
                                        xi=i.value[:]
                                        xj=j.value[:]
                                else:
                                        if ind not in i.value_alt:
                                                raise Exception(i+' does not have a value for the index '+str(ind))
                                        if ind not in j.value_alt:
                                                raise Exception(j+' does not have a value for the index '+str(ind))
                                        xi=i.value_alt[ind][:]
                                        xj=j.value_alt[ind][:]
                                val=val+xi.T*self.quad[i,j]*xj
                
                return cvx.matrix(val,(1,1))
       
        value = property(eval,Expression.set_value,Expression.del_simple_var_value)
       
        def nnz(self):
                nz=0
                for ij in self.quad:
                        nz+=len(self.quad[ij].I)
                return nz

        
        def __mul__(self,fact):
                if isinstance(fact,AffinExp):
                        if fact.isconstant() and fact.size==(1,1):
                                selfcopy=self.copy()
                                for ij in selfcopy.quad:
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
                        return self*AffinExp({},constant=fact[:],size=fact.size,string=factString)

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
                        divi=divi[0]
                        division=self * (1/divi)
                        lstring = self.string
                        if ('+' in self.string) or ('-' in self.string):
                                lstring = '('+ self.string + ')'
                        if ('+' in diviString) or ('-' in diviString):
                                diviString = '('+ diviString + ')'
                        division.string =  lstring+ ' / ' + diviString
                        return division
                else : #constant term
                        divi,diviString=_retrieve_matrix(divisor,(1,1))
                        return self/AffinExp({},constant=divi[:],size=(1,1),string=diviString)                   
                        
        def __add__(self,term):
                if isinstance(term,QuadExp):
                        selfcopy=self.copy()
                        for ij in self.quad:
                                if ij in term.quad:
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
                        expQE=QuadExp({},term,term.affstring())
                        return self+expQE
                else:
                        term,termString=_retrieve_matrix(term,(1,1))
                        expAE=AffinExp(factors={},constant=term,size=term.size,string=termString)
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
                           and (not exp.LR is None)):
                                if (not exp.LR[1] is None):#rotated cone
                                        return Constraint('RScone',None,self.LR[0],exp.LR[0],exp.LR[1])
                                else:#simple cone
                                        return Constraint('SOcone',None,self.LR[0],exp.LR[0])
                                        
                        else:
                                return Constraint('quad',None,self-exp,0)
                if isinstance(exp,AffinExp):
                        if exp.size<>(1,1):
                                raise Exception('RHS must be scalar')
                        exp2=AffinExp(factors={},constant=cvx.matrix(1.,(1,1)),size=(1,1),string='1')
                        expQE=QuadExp({},exp,exp.affstring(),LR=(exp,exp2))
                        return self<expQE
                else:
                        term,termString=_retrieve_matrix(exp,(1,1))
                        expAE=AffinExp(factors={},constant=term,size=(1,1),string=termString)
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
                                        size=(1,1),string=exp.string+'**0.5')
                                
                                if not(self.LR is None):
                                        return (Norm(cst)**2)<self
                                else:
                                        return exp<self
                        else:
                                return (-self)<(-exp)
                else:
                        term,termString=_retrieve_matrix(exp,(1,1))
                        expAE=AffinExp(factors={},constant=term,size=(1,1),string=termString)
                        return self>expAE
                                

class GeneralFun(Expression):
        """A class storing a scalar function,
           applied to an affine expression.
           It derives from :class:`Expression<picos.Expression>`.
        """
        def __init__(self,fun,Exp,funstring):
                Expression.__init__(self,self.funstring+'( '+Exp.string+')')
                self.fun=fun
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
                self.Exp=Exp
                """The affine expression to which the function is applied"""
                self.funstring=funstring
                """a string representation of the function name"""
                #self.string=self.funstring+'( '+self.Exp.affstring()+' )'

        def __repr__(self):
                return '# general function '+self.string+' #'
         
        def __str__(self):
                if self.is_valued():
                        return str(self.value[0])
                else:
                        return repr(self)

        def eval(self,ind=None):
                val=self.Exp.eval(ind)
                o,g,h=self.fun(val)
                return cvx.matrix(o,(1,1))
                
        value = property(eval,Expression.set_value,Expression.del_simple_var_value)

        
class Variable(AffinExp):
        """This class stores a variable. It
        derives from :class:`AffinExp<picos.AffinExp>`.
        """
        
        def __init__(self,name,
                          size,
                          Id,
                          startIndex,
                          vtype = 'continuous'):
                
                ##attributes of the parent class (AffinExp)
                idmat=_svecm1_identity(vtype,size)
                AffinExp.__init__(self,factors={self:idmat},
                                       constant=None,
                                       size=size,
                                       string=name
                                       )
                
                self.name=name
                """The name of the variable (str)"""

                self.Id=Id
                """An integer index"""
                self.vtype=vtype
                """one of the following strings:
                
                     * 'continuous' (continuous variable)
                     * 'binary'     (binary 0/1 variable)
                     * 'integer'    (integer variable)
                     * 'symmetric'  (symmetric matrix variable)
                     * 'semicont'   (semicontinuous variable 
                                    [can take the value 0 or any 
                                    other admissible value])
                     * 'semiint'    (semi integer variable 
                                     [can take the value 0 or any
                                     other integer admissible value])
                """
                self.startIndex=startIndex
                """starting position in the global vector of all variables"""
                
                self.endIndex=None
                """end position in the global vector of all variables"""
                
                
                if vtype=='symmetric':
                        self.endIndex=startIndex+(size[0]*(size[0]+1))/2 #end position +1
                else:
                        self.endIndex=startIndex+size[0]*size[1] #end position +1
                
                self._value=None

                self.value_alt = {} #alternative values for solution pools
                                

                
                
        def __str__(self):
                if self.is_valued():
                        if self.size==(1,1):
                                return str(self.value[0])
                        else:
                                return str(self.value)
                else:
                        return repr(self)
                        
        def __repr__(self):
                return '# variable {0}:({1} x {2}),{3} #'.format(
                        self.name,self.size[0],self.size[1],self.vtype)
                        
                        
        def eval(self, ind = None):
                if ind is None:
                        if self._value is None:
                                raise Exception(self.name+' is not valued')
                        else:
                                return cvx.matrix(self._value)
                else:
                        if ind in self.value_alt:
                                return cvx.matrix(self.value_alt[ind])
                        else:
                                raise Exception(self.name+' does not have a value for the index '+str(ind))
    
        
        def set_value(self,value):
                valuemat,valueString = _retrieve_matrix(value,self.size)
                if self.vtype == 'symmetric':
                        valuemat=svecm1(valuemat)
                if valuemat.size != self.size:
                        raise Exception('should be of size {0}'.format(self.size))
                self._value = valuemat

        def del_var_value(self):
                var._value = None
                       
        value = property(eval,set_value,del_var_value,"value of the affine expression")
        """value of the variable. The value of a variable is
                defined in the following two situations:
                
                * The user has assigned a value to the variable,
                  by using either the present ``value`` attribute,
                  or the function
                  :func:`set_var_value()<picos.Problem.set_var_value>` of the class
                  :class:`Problem<picos.Problem>`.
                
                * The problem involving the variable has been solved,
                  and the ``value`` attribute stores the optimal value
                  of this variable.
        """