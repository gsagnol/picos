import picos as pic
import cvxopt as cvx

vect = True
NN = 10000

P = pic.Problem()

if vect:
        x = P.add_variable('x',NN)
else:
        x = [P.add_variable('x_'+str(i),1) for i in range(NN)]

P.add_constraint(x[0] > 0)
for i in range(1,NN):
        P.add_constraint(x[i] > x[i-1] +1)

P.set_objective('min',pic.sum([x[i] for i in range(NN)]))
P.solve()