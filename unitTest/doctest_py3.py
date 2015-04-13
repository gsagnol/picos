import picos as pic
prob = pic.Problem()
x = prob.add_variable('x',1, vtype='integer') #scalar integer variable
prob.add_constraint(x<5.2)                    #x less or equal to 5.2
prob.set_objective('max',x)                   #maximize x
assert(str(prob) == '---------------------\noptimization problem  (MIP):\n1 variables, 1 affine constraints\n\nx \t: (1, 1), integer\n\n\tmaximize x\nsuch that\n  x < 5.2\n---------------------')


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
sol = prob.solve(solver='cvxopt', verbose = 0)

assert( max([abs(v) for v in (mu.value - cvx.matrix([[0.66017],[ 2.4189],[ 0.1640]]).T)]) < 1e-4)

assert(max([abs(v) for v in (prob.get_constraint(0).dual - cvx.matrix([-0.3412770157278555, 0.09164120429815878, -0.18755919557221587, -0.35241708871373845, 0.23181086079278834, 0.2589026387700825]))]) < 1e-5)

