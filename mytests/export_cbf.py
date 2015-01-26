import cvxopt as cvx
import picos as pic

#---------------------------------#
# First generate some data :      #
#       _ a list of 8 matrices A  #
#       _ a vector c              #
#---------------------------------#
A=[ cvx.matrix([[1,0,0,0,0],
                [0,3,0,0,0],
                [0,0,1,0,0]]),
cvx.matrix([[0,0,2,0,0],
                [0,1,0,0,0],
                [0,0,0,1,0]]),
cvx.matrix([[0,0,0,2,0],
                [4,0,0,0,0],
                [0,0,1,0,0]]),
cvx.matrix([[1,0,0,0,0],
                [0,0,2,0,0],
                [0,0,0,0,4]]),
cvx.matrix([[1,0,2,0,0],
                [0,3,0,1,2],
                [0,0,1,2,0]]),
cvx.matrix([[0,1,1,1,0],
                [0,3,0,1,0],
                [0,0,2,2,0]]),
cvx.matrix([[1,2,0,0,0],
                [0,3,3,0,5],
                [1,0,0,2,0]]),
cvx.matrix([[1,0,3,0,1],
                [0,3,2,0,0],
                [1,0,0,2,0]])
]


prob_D = pic.Problem()
AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
s=len(AA)
m=AA[0].size[0]
AA=pic.new_param('A',AA)
w = prob_D.add_variable('w',s,lower=0)
t = prob_D.add_variable('t',1)
X = prob_D.add_variable('X',(m,m),'symmetric')

#constraint and objective
prob_D.add_constraint(1|w < 1)
prob_D.add_constraint( X == pic.sum([w[i]*AA[i]*AA[i].T for i in range(s)],'i'))
prob_D.add_constraint( X >> 0)
prob_D.add_constraint(t < pic.detrootn(X))
prob_D.set_objective('max',t)
self = prob_D



P = pic.Problem()
x = P.add_variable('x',3)
Y = P.add_variable('Y',(2,2),'symmetric')
Z = P.add_variable('Z',(3,3),'symmetric')
P.add_constraint( x < 2)
P.add_constraint(Y>>0)
P.add_constraint(x[0]*cvx.matrix([[1,1],[1,1]]) << x[1]*cvx.matrix([[1,0],[0,1]]) )
M1 = cvx.matrix([[1,0],[0,1]])
M2 = cvx.matrix([[1,-1,0],[-1,1,0],[0,0,1]])
P.add_constraint( 2*x[2] + (M1|Y) + (M2|Z) == 1)
P.add_constraint(Z >> 0)
