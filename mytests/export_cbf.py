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
#AA = [cvx.sparse([1,2,3]),cvx.sparse([1,0,2]),cvx.sparse([0,0,1])]
s=len(AA)
m=AA[0].size[0]
AA=pic.new_param('A',AA)
w = prob_D.add_variable('w',s,lower=0,upper=1)
t = prob_D.add_variable('t',1)
#X = prob_D.add_variable('X',(m,m),'symmetric')

#constraint and objective
prob_D.add_constraint(1|w < 1,key='simplex')
#prob_D.add_constraint( X == pic.sum([w[i]*AA[i]*AA[i].T for i in range(s)],'i'))
#prob_D.add_constraint( X >> 0)
X = pic.sum([w[i]*AA[i]*AA[i].T for i in range(s)],'i')
prob_D.add_constraint(t < pic.detrootn(X))
prob_D.set_objective('max',t)
self = prob_D


#toy
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

#example 1
import picos as pic
import cvxopt as cvx
M0 = cvx.matrix([[2,1,0],[1,2,1],[0,1,2]])
M1 = cvx.matrix([[1,0,0],[0,1,0],[0,0,1]])
M2 = cvx.matrix([[1,1,1],[1,1,1],[1,1,1]])

P = pic.Problem()
X = P.add_variable('X',(3,3),'symmetric')
x = P.add_variable('x',3)
P.add_constraint((M1|X) + x[1] == 1)
P.add_constraint((M2|X) + x[0] + x[2] == 0.5)
P.add_constraint(X>>0)
P.add_constraint(abs((x[0]//x[2])) < x[1])
P.set_objective('min', (M0|X) + x[1])

#example 2
import picos as pic
import cvxopt as cvx
M0 = cvx.matrix([[1,0],[0,1]])
M1 = cvx.matrix([[0,1],[1,0]])
M2 = cvx.matrix([[0,1],[1,3]])
M3 = cvx.matrix([[3,1],[1,0]])
M4 = cvx.matrix([[1,0],[0,1]])

P = pic.Problem()
X = P.add_variable('X',(2,2),'symmetric')
x = P.add_variable('x',2)

P.add_constraint((M1|X) >= x[0] + x[1])
P.add_constraint(x[0]*M2+x[1]*M3-M4>>0)
P.add_constraint(X>>0)
P.set_objective('min', (M0|X) + x[0] + x[1] + 1)


#matrix sqrt
import picos as pic
import cvxopt as cvx
P = pic.Problem()
A = cvx.matrix([[1,0],[0,1]])
B = cvx.matrix([[2,1],[1,2]])
X = P.add_variable('X',(2,2),'symmetric')
P.add_constraint(X>>0)
P.add_constraint(((A & X) // (X & B))>>0)
P.set_objective('max', ('I'|X))


#MISOCP - exact A
prob_exact_A=pic.Problem()
AA=[cvx.sparse(a,tc='d') for a in A] #each AA[i].T is a 3 x 5 observation matrix
s=len(AA)
m=AA[0].size[0]
AA=pic.new_param('A',AA)

N =pic.new_param('N',20) #number of trials allowed
I =pic.new_param('I',cvx.spmatrix([1]*m,range(m),range(m),(m,m))) #identity matrix
Z=[prob_exact_A.add_variable('Z['+str(i)+']',AA[i].T.size) for i in range(s)]
n=prob_exact_A.add_variable('n',s, vtype='integer')
t=prob_exact_A.add_variable('t',s)

#define the constraints and objective function
prob_exact_A.add_list_of_constraints(
        [abs(Z[i])**2<n[i]*t[i] for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_exact_A.add_constraint(
        pic.sum(
        [AA[i]*Z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )
        == I )

prob_exact_A.add_constraint( 1|n < N )
prob_exact_A.set_objective('min',1|t)

P = prob_exact_A
#solve the problem and display the optimal design
print P
