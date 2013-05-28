#TEST CUTTING PLANES
solver='mosek7' #TODO something does not work anymore with mosek6...
import picos as pic
P=pic.Problem()
x=P.add_variable('x',5)

P.set_objective('max',x[0]+0.5*x[2]+x[-1]-x[1])
P.add_constraint(x>0)
y = P.add_variable('y',2)#tmp shift
P.add_constraint(abs(x[2]//x[3])<y[1])#tmp shift
P.add_constraint(1|x<3)
#P.add_constraint(abs(x)<2)
P.add_constraint(abs(x+0.5*x[2]*'e_2(5,1)')<2) #x2 has a scale factor 1.5
#P.solve(solver='cplex')

#y = P.add_variable('y',2)
P.add_constraint(x[4]<y[0])
#P.add_constraint(abs(x[2]//x[3])<y[1])
P.add_constraint(y>0)
P.add_constraint(1|y<1)
P.solve(solver=solver)




import picos as pic
P2=pic.Problem()
x=P2.add_variable('x',5)

P2.set_objective('max',x[0]+0.5*x[2]+x[-1]-x[1])
P2.add_constraint(x>0)
P2.add_constraint(1|x<3)
#P2.add_constraint(abs(x)<2)
P2.add_constraint(abs(x+0.5*x[2]*'e_2(5,1)')<2) #x2 has a scale factor 1.5
P2.solve(solver=solver)

y = P2.add_variable('y',2)
P2.add_constraint(x[4]<y[0])
P2.add_constraint(abs(x[2]//x[3])<y[1])
P2.add_constraint(y>0)
P2.add_constraint(1|y<1)
P2.solve(solver=solver)

print x,'--',P.get_valued_variable('x')
print y,'--',P2.get_valued_variable('y')

for c1,c2 in zip(P.constraints,P2.constraints):
        print c1.dual,'--',c2.dual