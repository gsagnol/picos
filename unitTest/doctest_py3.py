# coding: utf-8

def cleanspace(st):
    return st.replace(' ','').replace('\n','').replace('\t','')

def cvxcomp(A,B):
    return max([abs(v) for v in A-B])

SOLVER = 'cvxopt'

#-------------#
#  print test #
#-------------#

import picos as pic
print('starting tests with picos'+str(pic.__version__))

prob = pic.Problem()
x = prob.add_variable('x',1, vtype='integer') #scalar integer variable
prob.add_constraint(x<5.2)                    #x less or equal to 5.2
prob.set_objective('max',x)                   #maximize x
assert(cleanspace(str(prob)) == cleanspace(
    '---------------------\noptimization problem  (MIP):\n1 variables, 1 affine constraints\n\nx \t: (1, 1), integer\n\n\tmaximize x\nsuch that\n  x < 5.2\n---------------------'))


#--------------------------#
#  optdes example in intro #
#--------------------------#

import numpy as np
import cvxopt as cvx

#generate data
A = [   cvx.sparse([[1 ,2 ,0 ],
                        [2 ,0 ,0 ]]),
        cvx.sparse([[0 ,2 ,2 ]]),
        cvx.sparse([[0 ,2 ,-1],
                        [-1,0 ,2 ],
                        [0 ,1 ,0 ]])
        ]
K = cvx.sparse([[1 ,1 ,1 ],
                [1 ,-5,-5]])

#size of the data
s = len(A)
m = A[0].size[0]
l = [ Ai.size[1] for Ai in A ]
r = K.size[1]

#creates a problem and the optimization variables
prob = pic.Problem()
mu = prob.add_variable('mu',s)
Z  = [prob.add_variable('Z[' + str(i) + ']', (l[i],r))
        for i in range(s)]

#convert the constants into params of the problem
A = pic.new_param('A',A)
K = pic.new_param('K',K)

#add the constraints
prob.add_constraint( pic.sum([ A[i]*Z[i] for i in range(s)], #summands
                                'i',                            #name of the index
                                '[s]'                           #set to which the index belongs
                                ) == K
                        )
prob.add_list_of_constraints( [ abs(Z[i]) < mu[i] for i in range(s)], #constraints
                                'i',                                    #index of the constraints
                                '[s]'                                   #set to which the index belongs
                                )

#sets the objective
prob.set_objective('min', 1 | mu ) # scalar product of the vector of all ones with mu

#call to the solver cvxopt
sol = prob.solve(solver=SOLVER, verbose = 0)

assert( max([abs(v) for v in (mu.value - cvx.matrix([[0.66017],[ 2.4189],[ 0.1640]]).T)]) < 1e-4)

assert(max([abs(v) for v in (prob.get_constraint(0).dual - cvx.matrix([-0.3412770157278555, 0.09164120429815878, -0.18755919557221587, -0.35241708871373845, 0.23181086079278834, 0.2589026387700825]))]) < 1e-5)

#----------------------------#
#  Some tests of the tuto    #
#----------------------------#

pairs = [(0,2), (1,4), (1,3), (3,2), (0,4),(2,4)]  #a list of pairs
A = []
b = ( [0 ,2 ,0 ,3 ],                               #a tuple of 5 lists, each of length 4
      [1 ,1 ,0 ,5 ],
      [-1,0 ,2 ,4 ],
      [0 ,0 ,-2,-1],
      [1 ,1 ,0 ,0 ]
    )
for i in range(5):
    A.append(cvx.matrix(range(i-3,i+5),(2,4)))     #A is a list of 2x4 matrices
D={'Peter': 12,
   'Bob'  : 4,
   'Betty': 7,
   'Elisa': 14
   }
prob = pic.Problem()
t = prob.add_variable('t',1) #a scalar
x = prob.add_variable('x',4) #a column vector
Y = prob.add_variable('Y',(2,4)) #a matrix
Z = []
for i in range(5):
    Z.append( prob.add_variable('Z[{0}]'.format(i),(4,2))  )# a list of 5 matrices
w={}
for p in pairs:   #a dictionary of (scalar) binary variables, indexed by our pairs
    w[p] = prob.add_variable('w[{0}]'.format(p),1 , vtype='binary')

assert( cleanspace(str(w[2,4])) == cleanspace('# variable w[(2, 4)]:(1 x 1),binary #') )
assert( cleanspace(str(Y)) == cleanspace('# variable Y:(2 x 4),continuous #') )
assert( w[2,4].vtype == 'binary')
assert( x.vtype =='continuous')
assert(x.size==(4,1))
assert( not(Z[0].is_valued()))
Z[1].value = A[0].T
assert( Z[1].is_valued())
assert( Z[2].name == 'Z[2]')

AA = pic.new_param('A',A)
Alpha = pic.new_param('alpha',12)
alpha = 12
DD = pic.new_param('D',D)
bb = pic.new_param('b',b)
x_minus_1 = x - 1
assert(cleanspace(str(x_minus_1)) == cleanspace('# (4 x 1)-affine expression: x -|1| #') )

#-----------------------------------#
# some tests with valued variables  #
#-----------------------------------#

Z[0].value = list(range(0,8))
Z[1].value = list(range(8,16))
Z[2].value = list(range(16,24))
Z[3].value = list(range(24,32))
Z[4].value = list(range(32,40))
t.value = -1
w[0, 2].value = 0
w[1, 4].value = 0
w[1, 3].value = 1
w[3, 2].value = 1
w[0, 4].value = 0
w[2, 4].value = 1
x.value = list(range(-5,-1))

Zv = [Zi.eval() for Zi in Z]
tv = t.value
wv = pic.tools.eval_dict(w)
xv = x.value

#left right multiplication
assert( cvxcomp((AA[1]*Z[0]*AA[2]).value, A[1]*Zv[0]*A[2]) < 1e-6)
#dot product
assert( cvxcomp( ( bb[2] | x ).value, bb[2].T.value * xv)  < 1e-6 )
#hadamard
assert( cvxcomp( (bb[1]^x).value, cvx.matrix([bi*xi for bi,xi in zip(bb[1].value,xv)]) )< 1e-6 )
#concatenation
RHS = cvx.matrix([[ 1.00e+00, -1.00e+00, -5.00e+00, -2.20e+01],
                  [ 1.00e+00,  0.00e+00, -4.00e+00, -1.00e+01],
                  [ 0.00e+00,  2.00e+00, -3.00e+00,  2.00e+00],
                  [ 5.00e+00,  4.00e+00, -2.00e+00,  1.40e+01],
                  [-5.00e+00, -4.00e+00, -3.00e+00, -2.00e+00]]).T
assert( cvxcomp(((bb[1] & bb[2] & x & AA[0].T*AA[0]*x) // x.T).value,RHS) < 1e-6)

#sum
assert(cvxcomp(
    pic.sum([A[i]*Z[i] for i in range(5)],'i','[5]').value,
    sum([A[i]*Zv[i] for i in range(5)])) < 1e-6
    )

#norm
assert( abs(Z[1]-2*A[0].T).value[0] == np.linalg.norm(Zv[1]-2*A[0].T,'fro'))

#quad
assert(cvxcomp( (x +2 | Z[1][:,1]).value, (xv+2).T * Zv[1][:,1]) < 1e-6)

#constring
assert(cleanspace(str(pic.sum([AA[i]*Z[i] for i in range(5)],'i','[5]') == 0))
       == cleanspace(str('# (2x2)-affine constraint: Σ_{i in [5]} A[i]*Z[i] = |0| #'))
       )


#cons slacks
assert( (abs(x) < (2|x-1)).slack[0] == 2*sum(xv-1)- np.linalg.norm(xv))
assert( (1 < (t-1)*(x[2]+x[3]) ).slack[0] == ((tv-1) * (xv[2]+xv[3])-1)[0])

#powers
assert((Z[0][4]**(2./3)).value == Zv[0][4]**(2./3))
assert( cvxcomp(
    ((1-t)**0.6666 > x[0]).slack,
    (1-tv)**0.6666 - xv[0]) < 1e-4)

assert((pic.norm(-x,'inf') < 2).slack[0] == -3)

M = prob.add_variable('M',(5,5),'symmetric')
M.value = [1+(i+j)+2*(i+j)**2-0.01*(i+j)**4 + (25 if i==j else 0) for i in range(5) for j in range(5)]
assert( cvxcomp((t < pic.detrootn(M)).slack, np.linalg.det(M.value)**(1./5) - tv[0] ) < 1e-6)

#---------------#
#  Complex SDP  #
#---------------#

P = pic.Problem()
Z = P.add_variable('Z',(3,2),'complex')

assert(cleanspace(str(Z.real))==cleanspace('# variable Z_RE:(3 x 2),continuous #'))
assert(cleanspace(str(Z.imag))==cleanspace('# variable Z_IM:(3 x 2),continuous #'))
assert(Z.vtype == 'complex')

P = cvx.matrix([ [1-1j , 2+2j  , 1    ],
                [3j   , -2j   , -1-1j],
                [1+2j, -0.5+1j, 1.5  ]
                ])
P = P * P.H

Q = cvx.matrix([ [-1-2j , 2j   , 1.5   ],
                [1+2j  ,-2j   , 2.-3j ],
                [1+2j  ,-1+1j , 1+4j  ]
                ])
Q = Q * Q.H

n=P.size[0]
P = pic.new_param('P',P)
Q = pic.new_param('Q',Q)

#create the problem in picos
F = pic.Problem()
Z = F.add_variable('Z',(n,n),'complex')

F.set_objective('max','I'|0.5*(Z+Z.H))       #('I' | Z.real) works as well
F.add_constraint(((P & Z) // (Z.H & Q))>>0 )


F.solve(solver=SOLVER,verbose = 0)
assert(abs(F.obj_value()-37.4742)<1e-4)
sol = cvx.matrix([
        [ 1.51e+01+2.21e+00j, -7.17e+00-1.22e+00j,  2.52e+00+6.87e-01j],
        [-4.88e+00+4.06e+00j,  1.00e+01-1.57e-01j,  8.33e+00+1.13e+01j],
        [-4.32e-01+2.98e-01j,  3.84e+00-3.28e+00j,  1.24e+01-2.05e+00j]]).T
        
#very coarse test because I just pasted the string repr of the solution
assert(max([abs(v)/abs(z) for v,z in zip(sol-Z.value,sol)])<0.005)

M = pic.new_param('M',Q)
n=3

P = pic.Problem()
U = P.add_variable('U',(n,n),'hermitian')
P.add_list_of_constraints([U[i,i]==1 for i in range(n)],'i')
P.add_constraint(U >> 0)

P.set_objective('min', U | M)
P.solve(solver=SOLVER,verbose=0)
solstr = """
[ 1.00e+00-j0.00e+00  9.97e-01-j7.20e-02 -9.22e-01-j3.86e-01]
[ 9.97e-01+j7.20e-02  1.00e+00-j0.00e+00 -8.92e-01-j4.51e-01]
[-9.22e-01+j3.86e-01 -8.92e-01+j4.51e-01  1.00e+00-j0.00e+00]
        """

assert(cleanspace(str(U))==cleanspace(solstr))

print('everything seems to work fine')