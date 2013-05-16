import pyMathProg as MP
reload(MP)

P=MP.Problem()
x=P.add_variable('x',2,vtype='integer')

P.add_constraint(x>0)
P.add_constraint(x<8)
P.add_constraint(1|x<10)

P.set_objective('min',-1|x)

sol=P.solve(solver = 'cplex', nbsol=10) # does not find all sols !!!

