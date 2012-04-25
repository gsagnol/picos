# coding: utf-8
import cvxopt as cvx
import numpy as np
import sys
from progress_bar import ProgressBar

from .tools import *

__all__=['Constraint']

class Constraint:
        """a class for describing a constraint (see the method add_constraint)
        """

        def __init__(self,typeOfConstraint,Id,Exp1,Exp2,Exp3=None,dualVariable=None,key=None):
                from .expression import AffinExp
                self.typeOfConstraint=typeOfConstraint
                self.Exp1=Exp1
                self.Exp2=Exp2
                self.Exp3=Exp3
                self.Id=Id
                self.dualVariable=dualVariable
                self.key=None
                self.myconstring = None
                self.myfullconstring = None
                if typeOfConstraint=='RScone' and Exp3 is None:
                        raise NameError('I need a 3d expression')
                if typeOfConstraint[:3]=='lin':
                        if Exp1.size<>Exp2.size:
                                raise NameError('incoherent lhs and rhs')
                if typeOfConstraint[2:]=='cone':                        
                        if Exp2.size<>(1,1):
                                raise NameError('expression on the rhs should be scalar')
                        if not Exp3 is None:
                                if Exp3.size<>(1,1):
                                        raise NameError(
                                        'expression on the rhs should be scalar')
                if typeOfConstraint=='lse':
                        if not (Exp2==0 or Exp2.is0()):
                                raise NameError('lhs must be 0')
                        self.Exp2=AffinExp(factors={},constant=cvx.matrix(0,(1,1)),string='0',size=(1,1),variables=Exp1.variables)
                if typeOfConstraint=='quad':
                        if not (Exp2==0 or Exp2.is0()):
                                raise NameError('lhs must be 0')
                        self.Exp2=AffinExp(factors={},constant=cvx.matrix(0,(1,1)),string='0',size=(1,1),variables=Exp1.variables)
                if typeOfConstraint[:3]=='sdp':
                        if Exp1.size<>Exp2.size:
                                raise NameError('incoherent lhs and rhs')
                        if Exp1.size[0]<>Exp1.size[1]:
                                raise NameError('lhs and rhs should be square')

        def __str__(self):
                if not(self.myfullconstring is None):
                        return self.myfullconstring
                if self.typeOfConstraint[:3]=='lin':
                        constr='# ({0}x{1})-affine constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='SOcone':
                        constr='# ({0}x{1})-SOC constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='RScone':
                        constr='# ({0}x{1})-Rotated SOC constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='lse':
                        constr='# ({0}x{1})-Log-Sum-Exp constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='quad':
                        constr='#Quadratic constraint '
                if self.typeOfConstraint[:3]=='sdp':
                        constr='# ({0}x{1})-LMI constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if not self.key is None:
                        constr+='('+self.key+')'
                constr+=': '
                return constr+self.constring()+' #'
        
        def __repr__(self):
                if self.typeOfConstraint[:3]=='lin':
                        constr='# ({0}x{1})-affine constraint: '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='SOcone':
                        constr='# ({0}x{1})-SOC constraint: '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='RScone':
                        constr='# ({0}x{1})-Rotated SOC constraint: '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='lse':
                        constr='# ({0}x{1})-Log-Sum-Exp constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                if self.typeOfConstraint=='quad':
                        constr='#Quadratic constraint '
                if self.typeOfConstraint[:3]=='sdp':
                        constr='# ({0}x{1})-LMI constraint '.format(
                                                self.Exp1.size[0],self.Exp1.size[1])
                return constr+self.constring()+' #'

        def constring(self):
                if not(self.myconstring is None):
                        return self.myconstring
                if self.typeOfConstraint[:3]=='lin':
                        sense=' '+self.typeOfConstraint[-1]+' '
                        #if self.Exp2.is0():
                        #        return self.Exp1.affstring()+sense+'0'
                        #else:
                        return self.Exp1.affstring()+sense+self.Exp2.affstring()
                if self.typeOfConstraint=='SOcone':
                        if self.Exp2.is1():
                                return '||'+ self.Exp1.affstring()+'|| < 1'
                        else:
                                return '||'+ self.Exp1.affstring()+ \
                                        '|| < '+self.Exp2.affstring()
                if self.typeOfConstraint=='RScone':
                        #if self.Exp1.size==(1,1):
                        #       if self.Exp1.isconstant():
                        #               retstr=self.Exp1.affstring() # warning: no square to simplfy
                        #       else:
                        #               retstr='('+self.Exp1.affstring()+')**2'
                        if self.Exp1.size==(1,1) and self.Exp1.isconstant():
                                retstr=self.Exp1.affstring() # warning: no square to simplfy
                        else:
                                retstr= '||'+ self.Exp1.affstring()+'||^2'
                        if (self.Exp2.is1() and self.Exp3.is1()):
                                return retstr+' < 1'
                        elif self.Exp2.is1():
                                return retstr+' < '+self.Exp3.affstring()
                        elif self.Exp3.is1():
                                return retstr+' < '+self.Exp2.affstring()
                        else:
                                return retstr+' < ( '+ \
                                self.Exp2.affstring()+')( '+self.Exp3.affstring()+')'
                if self.typeOfConstraint=='lse':
                        return 'LSE[ '+self.Exp1.affstring()+' ] < 0'
                if self.typeOfConstraint=='quad':
                        return self.Exp1.string+' < 0'
                if self.typeOfConstraint[:3]=='sdp':
                        #sense=' '+self.typeOfConstraint[-1]+' '
                        if self.typeOfConstraint[-1]=='<':
                                sense = ' ≼ ' #≺,≼,⊰
                        else:
                                sense = ' ≽ ' #≻,≽,⊱
                        #if self.Exp2.is0():
                        #        return self.Exp1.affstring()+sense+'0'
                        #elif self.Exp1.is0():
                        #        return '0'+sense+self.Exp2.affstring()
                        #else:
                        return self.Exp1.affstring()+sense+self.Exp2.affstring()

        def keyconstring(self,lgstkey=None):
                constr=''               
                if not self.key is None:
                        constr+='('+self.key+')'
                if lgstkey is None:             
                        constr+=':\t'                   
                else:
                        if self.key is None:                    
                                lcur=0
                        else:
                                lcur=len(self.key)+2
                        if lgstkey==0:
                                ntabs=0
                        else:
                                ntabs=int(np.ceil((2+lgstkey)/8.0))
                        missingtabs=int(  np.ceil(((ntabs*8)-lcur)/8.0)  )
                        for i in range(missingtabs):
                                constr+='\t'
                        if lcur>0:
                                constr+=': '
                        else:
                                constr+='  '
                        constr+=self.constring()
                return constr

        def set_dualVar(self,value):
                self.dualVariable=value
        
        def dual(self):
                """returns the optimal dual variable associated to the constraint"""
                return self.dualVariable

        def slack(self):
                """returns the slack of the constraint
                (For an inequality of the type ``lhs<rhs``,
                the slack is ``rhs-lhs``, and for ``lhs<rhs``
                the slack is ``lhs-rhs``)."""
                if self.typeOfConstraint[3]=='<':
                        return self.Exp2.eval()-self.Exp1.eval()
                elif self.typeOfConstraint[3]=='>':
                        return self.Exp1.eval()-self.Exp2.eval()
                elif self.typeOfConstraint=='lin=':
                        return self.Exp1.eval()-self.Exp2.eval()
                elif self.typeOfConstraint=='SOcone':
                        return self.Exp2.eval()-(abs(self.Exp1)).eval()
                elif self.typeOfConstraint=='RScone':
                        return self.Exp2.eval()[0]*self.Exp3.eval()[0]-(abs(self.Exp1)**2).eval()
                elif self.typeOfConstraint=='lse':
                        return -lse(self.Exp1).eval()
                elif self.typeOfConstraint=='quad':
                        return -(Exp1.eval())

