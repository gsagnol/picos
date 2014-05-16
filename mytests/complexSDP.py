import picos as pic
import cvxopt as cvx

P = cvx.matrix([[1,2-1j,3],[2+1j,4,-1-2j],[3,-1+2j,5]])
Q = cvx.matrix([[1,1-4j,3-1j],[1+4j,2,-3j],[3+1j,3j,1]])
R = cvx.matrix([[1,2-4j,8-1j],[1-5j,2,-4j],[3+1j,1+3j,1]])

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

c3 = pic.Problem()
Z = c3.add_variable('Z',(3,3),'hermitian')
c3.set_objective('min','I'|Z)
c3.add_constraint(((P & Z) // (Z.H & Q))>>0 )

#complex version of maxcut
c4 = pic.Problem()
Z = c4.add_variable('Z',(3,3),'hermitian')
c4.set_objective('max',P|Z)
c4.add_constraint(Z>>0)
c4.add_constraint(Z[0,0]<1)
c4.add_constraint(Z[1,1]<1)
c4.add_constraint(Z[2,2]<1)


#test Htranspose
Z0 = cvx.matrix([1,2+1j,0,2-1j,4,0,0,0,2],(3,3))
Z.value=Z0
PP = P[0:2,:]
print (PP*Z0).H
print (PP*Z).H