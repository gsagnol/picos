import picos as pic
import cvxopt as cvx

P = cvx.matrix([[1,2-1j,3],[2+1j,4,-1-2j],[3,-1+2j,5]])
Q = cvx.matrix([[1,1-4j,3-1j],[1+4j,2,-3j],[3+1j,3j,1]])
R = cvx.matrix([[1,2-4j,8-1j],[1-5j,2,-4j],[3+1j,1+3j,1]])#not hermitian

cpl = pic.Problem()
Z = cpl.add_variable('Z',(3,3),'hermitian')
cpl.set_objective('min','I'|Z)
cpl.add_constraint(P|Z>1)
cpl.add_constraint(Q|Z>1)
cpl.add_constraint(Z>>0)

print cpl
re = cpl.to_real()
Zr = re.get_variable('Z_RE')
Zi = re.get_variable('Z_IM')

re2 = pic.Problem()
XX = re2.add_variable('XX',(6,6),'symmetric')
PP = pic.tools._cplx_mat_to_real_mat(P)
QQ = pic.tools._cplx_mat_to_real_mat(Q)
I = cvx.spdiag([1]*3)
II = pic.tools._cplx_mat_to_real_mat(I)
re2.add_constraint(PP|XX>2)
re2.add_constraint(QQ|XX>2)
re2.add_constraint(XX>>0)
re2.set_objective('min',II|XX)


re3 = pic.Problem()
X = re3.add_variable('X',(3,3),'symmetric')
Y = re3.add_variable('Y',(3,3))
I = cvx.spdiag([1]*3)
re3.add_constraint((P.real()|X)-(P.imag()|Y)>1)
re3.add_constraint((Q.real()|X)-(Q.imag()|Y)>1)
re3.add_constraint(((X & -Y)// (Y & X))>>0)
re3.set_objective('min',I|X)


c2 = pic.Problem()
Z = c2.add_variable('Z',(3,3),'hermitian')
u = c2.add_variable('u',2)
#c2.set_objective('min','I'|Z+3*u[0]+2*u[1])
c2.set_objective('min',Q*Q.H|Z)
c2.add_constraint(P|Z>1)
#c2.add_constraint(Q|Z>1)
#c2.add_constraint(P*Z+Z*P.H+ u[0]*Q*Q.H + u[1]*P*P.H >> P+P.H+Q+Q.H)
c2.add_constraint(R*Z+Z*R.H>>0)
c2.add_constraint('I'|Z==1)
#c2.add_constraint(Z>>0)


#affinity between 2 operators
import picos as pic
import cvxopt as cvx
P = cvx.normal(3,3) + 1j*cvx.normal(3,3)
Q = cvx.normal(3,3) + 1j*cvx.normal(3,3)
P = P*P.H
Q = Q*Q.H

c3 = pic.Problem()
Z = c3.add_variable('Z',(3,3),'hermitian')
c3.set_objective('max','I'|Z)
c3.add_constraint(((P & Z) // (Z & Q))>>0 )

#in fact, Z itself must not be hermitian !
c3b = pic.Problem()
Zr = c3b.add_variable('Zr',(3,3),'continuous')
Zi = c3b.add_variable('Zi',(3,3),'continuous')
c3b.set_objective('max','I'|Zr)
c3b.add_constraint(((P & (Zr+1j*Zi)) // ((Zr.T-1j*Zi.T) & Q))>>0 )
c3b.solve()

#Or with the new cplex vtype
c3f = pic.Problem()
Z = c3f.add_variable('Z',(3,3),'complex')
c3f.set_objective('max','I'|0.5*(Z+Z.H))
c3f.add_constraint(((P & Z) // (Z.H & Q))>>0 )


r3 = pic.Problem()
X = r3.add_variable('X',(3,3),'symmetric')
Y = r3.add_variable('Y',(3,3))
r3.set_objective('max','I'|X)
Pr,Pi = P.real(), P.imag()
Qr,Qi = Q.real(), Q.imag()
r3.add_constraint( ((Pr & X & -Pi & -Y)//
                    (X  & Qr& -Y  &-Qi)//
                    (Pi & Y & Pr  & X )//
                    (Y  & Qi& X   & Qr))>>0)

rl = c3.to_real()
Zr = rl.get_variable('Z_RE')
Zi = rl.get_variable('Z_IM')

r3l = c3b.to_real()
Zr = r3l.get_variable('Zr')
Zi = r3l.get_variable('Zi')


r3b = pic.Problem()
Xb = r3b.add_variable('X',(3,3))
Yb = r3b.add_variable('Y',(3,3))
r3b.set_objective('max','I'|Xb)
Pr,Pi = P.real(), P.imag()
Qr,Qi = Q.real(), Q.imag()
r3b.add_constraint( ((Pr & Xb & -Pi & -Yb)//
                   (Xb.T  & Qr& Yb.T  &-Qi)//
                   (Pi    & Yb & Pr  & Xb )//
                   (-Yb.T & Qi& Xb.T  & Qr))>>0)
                   
""" Debug
print cvx.matrix(rl.constraints[-1].Exp1.constant,(12,12))
print cvx.matrix(r3.constraints[-1].Exp1.constant,(12,12))

print cvx.matrix(r3.constraints[-1].Exp1.factors[X][:,0].(12,12))

max(rl.constraints[-1].Exp1.factors[Zr]-r3.constraints[-1].Exp1.factors[X])
max(rl.constraints[-1].Exp1.factors[Zi]-r3.constraints[-1].Exp1.factors[Y])

max(r3l.constraints[-1].Exp1.factors[Zr]-r3b.constraints[-1].Exp1.factors[Xb])
max(r3l.constraints[-1].Exp1.factors[Zi]-r3b.constraints[-1].Exp1.factors[Yb])

I,J,V = r3l.constraints[-1].Exp1.factors[Zi].I,r3l.constraints[-1].Exp1.factors[Zi].J,r3l.constraints[-1].Exp1.factors[Zi].V
Ib,Jb,Vb = r3b.constraints[-1].Exp1.factors[Yb].I,r3b.constraints[-1].Exp1.factors[Yb].J,r3b.constraints[-1].Exp1.factors[Yb].V
"""

#complex version of maxcut
c4 = pic.Problem()
Z = c4.add_variable('Z',(3,3),'hermitian')
c4.set_objective('max',P|Z)
c4.add_constraint(Z>>0)
c4.add_constraint(Z[0,0]<1)
c4.add_constraint(Z[1,1]<1)
c4.add_constraint(Z[2,2]<1)

import picos as pic
import cvxopt as cvx
P = cvx.normal(3,3) + 1j*cvx.normal(3,3)
Q = cvx.normal(3,3) + 1j*cvx.normal(3,3)
P = P*P.H
Q = Q*Q.H

c4 = pic.Problem()
Z = c4.add_variable('Z',(3,3),'hermitian')
c4.set_objective('max',P|Z)
c4.add_constraint(Z>>0)
#c4.add_constraint(pic.tools.diag_vect(Z) == 1) ...fatal error ?
c4.add_constraint(Z[0,0]==1)
c4.add_constraint(Z[1,1]==1)
c4.add_constraint(Z[2,2]==1)



#test Htranspose
Z0 = cvx.matrix([1,2+1j,0,2-1j,4,0,0,0,2],(3,3))
Z.value=Z0
PP = P[0:2,:]
print (PP*Z0).H
print (PP*Z).H


[A  X]*     [A* X ]
[X* B]   =  [X* B*]

#test scalar product
import cvxopt as cvx
import picos as pic
import numpy as np
P = pic.Problem()
X = P.add_variable('X',(3,3),'hermitian')
Z = P.add_variable('Z',(3,3),'complex')
A = cvx.normal(3,3) + 1j * cvx.normal(3,3)
B = cvx.normal(3,3) + 1j * cvx.normal(3,3)
Z.value = B
X.value = (A + A.H)
print B | X
np.trace ((A+A.H).H*B)
print X|B
np.trace (B.H * (A+A.H))

print A | Z
np.trace (B.H * A)
print Z|A
np.trace (A.H * B)

#test frob norm
import picos as pic
import cvxopt as cvx

P = cvx.matrix([[1,2-1j,3],[2+1j,4,-1-2j],[3,-1+2j,5]])
Q = cvx.matrix([[1,1-4j,3-1j],[1+4j,2,-3j],[3+1j,3j,1]])
R = cvx.matrix([[1,2-4j,8-1j],[1-5j,2,-4j],[3+1j,1+3j,1]])#not hermitian

cpl = pic.Problem()
Z = cpl.add_variable('Z',(3,3),'hermitian')
cpl.set_objective('max','I'|Z)
cpl.add_constraint((P*P.H)|Z<1)
cpl.add_constraint((Q*Q.H)|Z<1)
#cpl.add_constraint(abs(Z)<0.04)
cpl.add_constraint(abs(2*Z)<0.08)#TODO here, handle this part- case, and then directly on a herm variable
cpl.add_constraint(Z>>0)


#AND TEST equiv of the following in complex numbers (complex SOCP)
import picos as pic
P = pic.Problem()
X = P.add_variable('X',(3,2),'complex')
t = P.add_variable('t',1)
P.add_constraint(pic.norm(X,(1,2)) < t) #Basis check with 1,2
P.add_constraint('|1|(1,3)' * X == [1-1j,2+3j])
P.set_objective('min',t)
P.solve()

import picos as pic
P = pic.Problem()
Z = P.add_variable('Z',(3,1),'complex')
t = P.add_variable('t',1)
P.add_constraint(abs(Z) < t)
P.add_constraint('|1|(1,3)' * Z == 1+2j)
P.set_objective('min',t)
P.solve()
