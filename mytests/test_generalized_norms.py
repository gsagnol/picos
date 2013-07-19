import picos as pic
P=pic.Problem()
x=P.add_variable('x',2)
P.set_objective('min',x[0]+2*x[1])
cs = P.add_constraint(pic.norm(x,-1.33333)>1,ret=True)
print P
P.solve()

#import picos as pic
#P=pic.Problem()
#x=P.add_variable('x',2)
#P.set_objective('max',x[0]+2*x[1])
#P.add_constraint(pic.norm(x,2.66667)<1)
#print P
#P.solve()

print x
p=-1.33333;[1./(2**(p/(p-1))+1)**(1./p),1./(0.5**(p/(p-1))+1)**(1./p)]

import picos as pic
p=1.7
P=pic.Problem()
x=P.add_variable('x',1)
t=P.add_variable('t',1)
P.set_objective('max',x-t)
cs = P.add_constraint(x**p<t,ret=True)
print P
P.solve()
print t
print p**(p/(1-p))

p=5./7.
P=pic.Problem()
x=P.add_variable('x',1)
t=P.add_variable('t',1)
P.set_objective('max',t-x)
cs = P.add_constraint(t<x**p,ret=True)
print P
P.solve()
print x
print (1/p)**(1/(p-1))

p = -6/5.
P=pic.Problem()
x=P.add_variable('x',1)
t=P.add_variable('t',1)
P.set_objective('min',t+x)
cs = P.add_constraint(t>x**p,ret=True)
print P
P.solve()
print x
print (-1/p)**(-1/(1-p))

