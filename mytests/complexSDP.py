import picos as pic
import cvxopt as cvx

P = cvx.matrix([[1,2-1j,3],[2+1j,4,-1-2j],[3,-1+2j,5]])
Q = cvx.matrix([[1,1-4j,3-1j],[1+4j,2,-3j],[3+1j,3j,1]])

cpl = pic.Problem()
Z = cpl.add_variable('Z',(3,3),'hermitian')
cpl.set_objective('min','I'|Z)
cpl.add_constraint(P|Z>1)
cpl.add_constraint(Q|Z>1)
cpl.add_constraint(Z>>0)

print cpl