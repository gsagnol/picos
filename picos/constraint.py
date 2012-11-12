# coding: utf-8

#-------------------------------------------------------------------
#Picos 0.1.1 : A pyton Interface To Conic Optimization Solvers
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

__all__=['Constraint']

class Constraint:
        """A class for describing a constraint.
        """

        def __init__(self,typeOfConstraint,Id,Exp1,Exp2,Exp3=None,dualVariable=None,key=None):
                from .expression import AffinExp
                self.typeOfConstraint=typeOfConstraint
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
                          ``Exp1 ⪳ Exp2`` or ``Exp1 ⪴ Exp2``.
                """

                self.Exp1=Exp1
                """LHS"""
                self.Exp2=Exp2
                """RHS
                   (ignored for constraints of type ``lse`` and ``quad``, where
                   ``Exp2`` is set to ``0``)
                """
                self.Exp3=Exp3
                """Second factor of the RHS for ``RScone`` constraints
                   (see :attr:`typeOfConstraint<picos.Constraint.typeOfConstraint>`).
                """
                self.Id=Id
                """An integer identifier"""
                self.dualVariable=dualVariable
                self.key=None
                """A string to give a key name to the constraint"""
                self.myconstring = None #workaround to redefine string representation
                self.myfullconstring = None #workaround to redefine complete constraint (with # ... #) string
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
                        self.Exp2=AffinExp(factors={},constant=cvx.matrix(0,(1,1)),string='0',size=(1,1))
                if typeOfConstraint=='quad':
                        if not (Exp2==0 or Exp2.is0()):
                                raise NameError('lhs must be 0')
                        self.Exp2=AffinExp(factors={},constant=cvx.matrix(0,(1,1)),string='0',size=(1,1))
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
                        constr='# ({0}x{1})-Geometric Programming constraint '.format(
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
                        constr='# ({0}x{1})-Geometric Programming constraint '.format(
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
                                retstr=self.Exp1.affstring()
                                if retstr[-5:]=='**0.5':
                                        retstr=retstr[:-5]
                                else:
                                        retstr+='**2'
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
        
        def dual_var(self):
                return self.dualVariable
                
        def del_dual(self):
                self.dualVariable=None
                
        dual=property(dual_var,set_dualVar,del_dual)
        """
        Value of the dual variable associated to this constraint
        
        See the :ref:`note on dual variables <noteduals>` in the tutorial
        for more information.
        """

        def slack_var(self):
                """returns the slack of the constraint
                (For an inequality of the type ``lhs<rhs``,
                the slack is ``rhs-lhs``, and for ``lhs>rhs``
                the slack is ``lhs-rhs``)."""
                if self.typeOfConstraint=='lse':
                        from .tools import lse
                        return -lse(self.Exp1).eval()
                elif self.typeOfConstraint[3]=='<':
                        return self.Exp2.eval()-self.Exp1.eval()
                elif self.typeOfConstraint[3]=='>':
                        return self.Exp1.eval()-self.Exp2.eval()
                elif self.typeOfConstraint=='lin=':
                        return self.Exp1.eval()-self.Exp2.eval()
                elif self.typeOfConstraint=='SOcone':
                        return self.Exp2.eval()-(abs(self.Exp1)).eval()
                elif self.typeOfConstraint=='RScone':
                        return self.Exp2.eval()[0]*self.Exp3.eval()[0]-(abs(self.Exp1)**2).eval()
                elif self.typeOfConstraint=='quad':
                        return -(self.Exp1.eval())

        def set_slack(self,value):
                raise AttributeError('slack is not writable')
        
        def del_slack(self):
                raise AttributeError('slack is not writable')
        
        slack=property(slack_var,set_slack,del_slack)
        """Value of the slack variable associated to this constraint
           (should be nonnegative/zero if the inequality/equality
           is satisfied)"""
        