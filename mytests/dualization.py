import cvxopt as cvx
import picos as pic


coneP=pic.Problem()
y=coneP.add_variable('y',3)
x=coneP.add_variable('x',3)
AAA=[cvx.matrix([[ 1.30,-0.23, 1.81, 2.27],
               [-0.23, 2.37, 2.07,-0.42],
               [ 1.81, 2.07, 5.02, 3.15],
               [ 2.27,-0.42, 3.15, 8.22]
              ]),
   cvx.matrix([[ 5.08, 1.53, 2.34, 2.92],
               [ 1.53, 2.28,-0.74, 0.46],
               [ 2.34,-0.74, 2.76, 1.87],
               [ 2.92, 0.46, 1.87, 2.06]
              ]),
   cvx.matrix([[ 4.10,-0.71, 0.86, 1.26],
               [-0.71, 1.85,-1.33, 0.30],
               [ 0.86,-1.33, 2.24,-0.07],
               [ 1.26, 0.30,-0.07, 0.82]
              ]),
   cvx.matrix([[ 0.56,-0.06, 0.32,-1.03],
               [-0.06, 2.43, 1.57,-0.22],
               [ 0.32, 1.57, 2.71, 2.10],
               [-1.03,-0.22, 2.10, 7.84]
              ])
   ]
           
AA=pic.new_param('A',AAA)
        
coneP.add_constraint(abs(x)<y[0]-0.1)
coneP.add_constraint(1|x==0.3)
coneP.add_constraint(0.1<y[1]*x[2])
coneP.add_constraint(pic.sum([x[i]*AA[i] for i in range(3)],'i') 
                 + y[2]*AA[3] >> 'I')
coneP.set_objective('min',1|y)
P=coneP
Q=P.dualize()






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
  
c = cvx.matrix([1,2,3,4,5])

socp=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
x=socp.add_variable('x',5)
socp.add_list_of_constraints(
        [abs(AA[i]*x+AA[i]*'|1|(5,1)') < '|2|(1,3)'*AA[i]*x - 1
        for i in range(s)],'i','[s]')
socp.add_constraint(1|x==3)
socp.add_constraint(x>0)
socp.set_objective('max',cc.T*x)
P=socp
Q=P.dualize()









prob_LP_c=pic.Problem()
AA=[cvx.sparse(a[:,i],tc='d').T for i in range(3) for a in A[4:]]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
u=prob_LP_c.add_variable('u',c.size)
prob_LP_c.add_list_of_constraints(
        [abs(AA[i]*u)<1 for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_LP_c.set_objective('max', cc|u)
P=prob_LP_c
Q=P.dualize()