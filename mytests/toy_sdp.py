import picos as pic
P=pic.Problem()
X=P.add_variable('X',(2,2),'symmetric')
P.add_constraint(X[0,0]==1)
P.add_constraint(X[1,1]==2)
P.add_constraint(X>>0)
P.set_objective('max',X[0,1])
P.solve(solver='mosek')


#nonsdp with symm variable
import picos as pic
import cvxopt as cvx
P=pic.Problem()
AA=cvx.matrix([[ 1.81e-01,  7.16e-01],
               [ 3.33e-01, -2.67e+00]]).T
#cvx.normal(2,2)
A=pic.new_param('A',AA+AA.T)
I=pic.new_param('I',cvx.spdiag([1.]*2))
X=P.add_variable('X',(2,2), 'symmetric')
P.add_constraint((X*A+A*X)[:2] == I[:2])
P.add_constraint(abs(X)<1)
P.set_objective('max',X[1,1])
P.solve(solver='cplex')