
import cvxopt as cvx
import picos as pic


A=cvx.matrix([[1,0,0,0,1],
                [0,0.2,2,0,0],
                [1,0,3,2,0]])
  


P=pic.Problem()
x = P.add_variable('x',3)
P.add_constraint(1|x<2)
P.add_constraint(abs(A[0:3,:]*x)**2<2*x[0]*x[1])
P.set_objective('max',x[1]+3*x[2])
P.solve(solver='mosek')


import cvxopt as cvx
import picos as pic

A=cvx.matrix([[-1,0,0,0,1],
                [0,3,2,0,0],
                [1,0,3,2,0]])
  


P=pic.Problem()
x = P.add_variable('x',3)
P.add_constraint(1|x<2)
P.add_constraint(abs(A[0:3,:]*x)<2*x[0])
P.set_objective('max',x[1]+3*x[2])
P.solve(solver='cvxopt')