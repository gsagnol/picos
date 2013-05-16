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