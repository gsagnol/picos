import picos as pic
P = pic.Problem()
x = P.add_variable('x',7)
P.add_constraint(x[0] + x[1] +x[2] + x[3] > 1)
P.add_constraint(x[2]+x[4] > 1)
P.add_constraint(x[0]>0.55)
P.add_constraint(x[5]-x[3]<0)

P.add_constraint(x[1]>0.1)
#x.set_sparse_lower([1],[0.1])

P.add_constraint(x[4]<0.85)

#P.add_constraint(-2*x[5]<-0.4)
#P.add_constraint(-0.5*x[6]>0.25)
x.set_sparse_lower([5,6],[0.2,-8])
x.set_sparse_upper([5,6],[0.6,-0.5])


#P.set_objective('max',-(x[0] + 2*x[1] + 3*x[2] + 4*x[3]-x[6]))
P.set_objective('min',x[0] + 2*x[1] + 3*x[2] + 4*x[3]-x[6])

P.solve(solver='mosek')

[cs.dual[0] for cs in P.constraints]

#[0,3,1,4,2,3,2,2]
# ou
#[1-eps,2+eps,eps,3+eps,1+eps,2+eps,(3+eps)/2,2]