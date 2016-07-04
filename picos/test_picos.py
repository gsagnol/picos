#generate data

from __future__ import print_function

import cvxopt as cvx
import picos as pic

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

#create the problems

#--------------------------------------#
#         D-optimal design             #
#--------------------------------------#
prob_D = pic.Problem()
AA=[cvx.sparse(a,tc='d') for a in A]
s=len(AA)
m=AA[0].size[0]
AA=pic.new_param('A',AA)
mm=pic.new_param('m',m)
L=prob_D.add_variable('L',(m,m))
V=[prob_D.add_variable('V['+str(i)+']',AA[i].T.size) for i in range(s)]
w=prob_D.add_variable('w',s)
u={}
for k in ['01','23','4.','0123','4...','01234']:
        u[k] = prob_D.add_variable('u['+k+']',1)
prob_D.add_constraint(
                pic.sum([AA[i]*V[i]
                    for i in range(s)],'i','[s]')
                 == L)
#L lower inferior
prob_D.add_list_of_constraints( [L[i,j] == 0
                                for i in range(m)
                                for j in range(i+1,m)],['i','j'],'upper triangle')
prob_D.add_list_of_constraints([abs(V[i])<(mm**0.5)*w[i]
                                for i in range(s)],'i','[s]')
prob_D.add_constraint(1|w<1)
#SOC constraints to define u['01234'] such that u['01234']**8 < t[0] t[1] t[2] t[3] t[4]
prob_D.add_constraint(u['01']**2   <L[0,0]*L[1,1])
prob_D.add_constraint(u['23']**2   <L[2,2]*L[3,3])
prob_D.add_constraint(u['4.']**2   <L[4,4])
prob_D.add_constraint(u['0123']**2 <u['01']*u['23'])
prob_D.add_constraint(u['4...']**2 <u['4.'])
prob_D.add_constraint(u['01234']**2<u['0123']*u['4...'])

prob_D.set_objective('max',u['01234'])

#--------------------------------------#
# multiresponse case c-optimality SOCP #
#--------------------------------------#
prob_multiresponse_c=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
u=prob_multiresponse_c.add_variable('u',c.size)
prob_multiresponse_c.add_list_of_constraints(
        [abs(AA[i]*u)<1 for i in range(s)], #constraints
        #[abs((AA[i]*u)//np.sqrt(3))<2 for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_multiresponse_c.set_objective('max', cc|u)
#--------------------------------------------#
# multiresponse case c-optimality, dual SOCP #
#--------------------------------------------#
prob_dual_c=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
z=[prob_dual_c.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]
mu=prob_dual_c.add_variable('mu',s)
prob_dual_c.add_list_of_constraints(
        [abs(z[i])<mu[i] for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_dual_c.add_constraint( 
        pic.sum(
        [AA[i].T*z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        == cc )
prob_dual_c.set_objective('min',1|mu)

#----------------------------------------------------------------#
# multiresponse case exacr c-optimality, Lagrangian bound MISOCP #
#----------------------------------------------------------------#
prob_exact_c=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
z=[prob_exact_c.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]
w=prob_exact_c.add_variable('w',s,vtype='integer')
t=prob_exact_c.add_variable('t',1)
prob_exact_c.add_list_of_constraints(
        [abs(z[i])<w[i] for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_exact_c.add_constraint( 
        pic.sum(
        [AA[i].T*z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        == t*cc )
        
prob_exact_c.add_constraint( 1|w < 20 )
        
prob_exact_c.set_objective('max',t)


#----------------------------------------------#
# single case c-optimality, exact-int dual MIP #
#----------------------------------------------#
prob_exact_single_c=pic.Problem()
AA=[cvx.sparse(a[:,i],tc='d').T for i in range(3) for a in A[4:]]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
z=[prob_exact_single_c.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]
w=prob_exact_single_c.add_variable('w',s,vtype='integer')
t=prob_exact_single_c.add_variable('t',1)
prob_exact_single_c.add_list_of_constraints(
        [abs(z[i])<w[i] for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_exact_single_c.add_constraint( 
        pic.sum(
        [AA[i].T*z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        == t*cc )
        
prob_exact_single_c.add_constraint( 1|w < 20 )
        
prob_exact_single_c.set_objective('max',t)

#--------------------------------------#
# multiresponse case A-optimality SOCP #
#--------------------------------------#
prob_multiresponse_A=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
m=AA[0].size[1]
AA=pic.new_param('A',AA)
U=prob_multiresponse_A.add_variable('U',(m,m))
prob_multiresponse_A.add_list_of_constraints(
        [abs(AA[i]*U)<1 for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_multiresponse_A.set_objective('max', 'I'|U)

#--------------------------------------------#
# multiresponse case A-optimality, dual SOCP #
#--------------------------------------------#
prob_dual_A=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
Z=[prob_dual_A.add_variable('Z['+str(i)+']',AA[i].size) for i in range(s)]
mu=prob_dual_A.add_variable('mu',s)
prob_dual_A.add_list_of_constraints(
        [abs(Z[i])<mu[i] for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_dual_A.add_constraint( 
        pic.sum(
        [AA[i].T*Z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        == 'I' )
prob_dual_A.set_objective('min',1|mu)

#---------------------------------------#
# single response case, c-optimality LP #
#---------------------------------------#
prob_LP_c=pic.Problem()
#AA=[cvx.sparse(a[:,i],tc='d').T for i in range(3) for a in A[4:]]
AA=[cvx.sparse(a[:,0],tc='d').T for a in A]#new version to have variable bounds
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
u=prob_LP_c.add_variable('u',c.size)
prob_LP_c.add_list_of_constraints(
        [abs(AA[i]*u)<1 for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
#prob_LP_c.set_objective('max', cc|u)
prob_LP_c.set_objective('min', cc|u)#new version to have a minimization LP

#--------------------------------------------#
# single response case, c-optimality dual LP #
#--------------------------------------------#
prob_LP_dual_c=pic.Problem()
#AA=[cvx.sparse(a[:,i],tc='d').T for i in range(3) for a in A[4:]]
AA = [cvx.sparse(a[:,0],tc='d').T for a in A] #new version to have variable bounds
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
z=[prob_LP_dual_c.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]
mu=prob_LP_dual_c.add_variable('mu',s)
prob_LP_dual_c.add_list_of_constraints(
        [abs(z[i])<mu[i] for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_LP_dual_c.add_constraint( 
        pic.sum(
        [AA[i].T*z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        == cc )
prob_LP_dual_c.set_objective('min',1|mu)

#--------------------------------------#
# multiresponse case c-optimality SDP  #
#--------------------------------------#
prob_SDP_c=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
X=prob_SDP_c.add_variable('X',(c.size[0],c.size[0]),vtype='symmetric')
prob_SDP_c.add_list_of_constraints(
        [(AA[i].T*AA[i] | X ) <1 for i in range(s)], #constraints
        'i', #index
        '[s]' #set to which the index belongs
        )
prob_SDP_c.add_constraint(X>>0)
prob_SDP_c.set_objective('max', cc.T*X*cc)

#--------------------------------------------#
# multiresponse case c-optimality, dual SDP  #
#--------------------------------------------#
prob_SDP_c_dual=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
mu=prob_SDP_c_dual.add_variable('mu',s)
prob_SDP_c_dual.add_constraint( 
        pic.sum(
        [mu[i]*AA[i].T*AA[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        >> cc*cc.T )
prob_SDP_c_dual.add_constraint(mu>0)
prob_SDP_c_dual.set_objective('min',1|mu)

#--------------------------------------#
# multiresponse case A-optimality SDP  #
#--------------------------------------#

prob_SDP_A=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
#s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
X=prob_SDP_A.add_variable('X',(c.size[0],c.size[0]),vtype='symmetric')
w=prob_SDP_A.add_variable('w',s)
prob_SDP_A.add_constraint(
        (   (pic.sum(
        [w[i]*AA[i].T*AA[i] for i in range(s)], 
        'i','[s]')                            &        'I') //
        (          'I'                        &         X))
        >> 0)
prob_SDP_A.add_constraint(w>0)
prob_SDP_A.add_constraint(1|w==1)
prob_SDP_A.set_objective('min', 'I'|X)

#-------------------------------------------#
#  multiresponse SOCP, multiple constraints #
#  (1|w_0 ... w_3) <1 , (1|w_4 ... w_7) <1  #
#-------------------------------------------#

#solve min <K,U>+b' l+ l0
                                #s.t.   ||Ai U||2< ri'l
                                #       ||A0 U||2<l0
                                #       l,l0>0
                                
prob_multiresponse_multiconstraints=pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
u=prob_multiresponse_multiconstraints.add_variable('u',c.size)
lbd=prob_multiresponse_multiconstraints.add_variable('lbd',2)
prob_multiresponse_multiconstraints.add_list_of_constraints(
        [abs(AA[i]*u)**2<lbd[0] for i in range(s//2)], #constraints
        'i', #index
        '0..3' #set to which the index belongs
        )
prob_multiresponse_multiconstraints.add_list_of_constraints(
        [abs(AA[i]*u)**2<lbd[1] for i in range(s//2,s)], #constraints
        'i', #index
        '4..7' #set to which the index belongs
        )
prob_multiresponse_multiconstraints.add_constraint(lbd>0)
prob_multiresponse_multiconstraints.add_constraint((1|lbd)<1)
prob_multiresponse_multiconstraints.set_objective('max', (cc|u))

multiconstraints_dual=pic.Problem()
alpha=multiconstraints_dual.add_variable('alpha',s)
mu=multiconstraints_dual.add_variable('mu',s)
t=multiconstraints_dual.add_variable('t',1)
z=[multiconstraints_dual.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]

multiconstraints_dual.add_constraint( 
        pic.sum(
        [AA[i].T*z[i] for i in range(s)], #summands
        'i', #index
        '[s]' #set to which the index belongs
        )  
        == cc )
        
multiconstraints_dual.add_constraint((1|mu[:4])<t)
multiconstraints_dual.add_constraint((1|mu[4:])<t)
multiconstraints_dual.add_list_of_constraints(
        [abs(z[i])**2<4*mu[i]*alpha[i]
        for i in range(s)],'i','[s]')
multiconstraints_dual.set_objective('min',(1|alpha)+t)


#test QCQP
S=cvx.matrix([[ 4e-2,  6e-3, -4e-3,    0.0 ],
             [ 6e-3,  1e-2,  0.0,     0.0 ],
             [-4e-3,  0.0,   2.5e-3,  0.0 ],
             [ 0.0,   0.0,   0.0,     0.0 ]])
pbar=cvx.matrix([.12, .10, .07, .03])
qcqp=pic.Problem()
x=qcqp.add_variable('x',4)
z=qcqp.add_variable('z',2)
qcqp.add_constraint(1|x<6)
qcqp.add_constraint(x>0)
qcqp.add_constraint(z[0]**2+2*z[1]**2-2*z[0]*z[1]+x[2]**2+0.3*x[2]*z[0]-x[3]<x[0])
pbar=pic.new_param('pbar',pbar)
S=pic.new_param('S',S)
qcqp.add_constraint(z<0)
qcqp.add_constraint((1|z)>-6)
qcqp.set_objective('min',-(pbar|x)+10*x.T*S*x+3*z[0]+z[1])

#----------

#S=cvx.matrix([[ 4e-2,  6e-3, -4e-3,    0.0 ],
             #[ 6e-3,  1e-2,  0.0,     0.0 ],
             #[-4e-3,  0.0,   2.5e-3,  0.0 ],
             #[ 0.0,   0.0,   0.0,     0.0 ]])
#pbar=cvx.matrix([.12, .10, .07, .03])
#qcqp=pic.Problem()
#x01=qcqp.add_variable('x01',2)
#x2=qcqp.add_variable('x2',1)#,'integer')
#x3=qcqp.add_variable('x3',1)
#z=qcqp.add_variable('z',2)
#qcqp.add_constraint((1|(x01//x2//x3))<6)
#qcqp.add_constraint((x01//x2//x3)>0)
#qcqp.add_constraint(z[0]**2+2*z[1]**2-2*z[0]*z[1]+x2**2+0.3*x2*z[0]-x3<x01[0])
#pbar=pic.new_param('pbar',pbar)
#S=pic.new_param('S',S)
#qcqp.add_constraint(z<0)
#qcqp.add_constraint((1|z)>-6)
#qcqp.set_objective('min',-(pbar|(x01//x2//x3))+10*(x01//x2//x3).T*S*(x01//x2//x3)+3*z[0]+z[1])
##qcqp.add_constraint(z[0]**2+z[1]**2 < 10)
#qcqp.add_constraint(abs(z)**2 < 10)
#qcqp.solve(solver='mosek')

#------------

#QP+SOCP
soqcqp=qcqp.copy()
z=soqcqp.get_variable('z') #variable handle
soqcqp.add_constraint(abs(z)<x[0])

#test MIQCQP
miqcqp=qcqp.copy()
z=miqcqp.get_variable('z') #variable handle
miqcqp.add_constraint(z[0]**2+z[1]**2 < 10)
#we add the constraint x[2] integer
x=miqcqp.get_variable('x') #handle to the variable x
i=miqcqp.add_variable('i',1,vtype='integer')
miqcqp.add_constraint(x[2]==i)

#------------#
#standard LP #
#------------#

lp = pic.Problem()
AA=[cvx.sparse(a,tc='d').T for a in A]
AA.append(cvx.sparse([0,3,0,0,0]).T)
s=len(AA)
AA=pic.new_param('A',AA)
cc=pic.new_param('c',c)
x=lp.add_variable('x',5)

lp.add_constraint(AA[5]*x < 3)
lp.add_constraint(AA[6]*x > 2)
lp.add_constraint(AA[7]*x == 2.5)


lp.add_constraint(x>0)
        
lp.set_objective('max',cc.T*x)


dualp=pic.Problem()
mu5=dualp.add_variable('mu5',3)
mu6=dualp.add_variable('mu6',3)
mu7=dualp.add_variable('mu7',3)

dualp.add_constraint(mu5>0)
dualp.add_constraint(mu6>0)
dualp.add_constraint(AA[5].T*mu5-AA[6].T*mu6+AA[7].T*mu7==cc)

dualp.set_objective('min',(2.5|mu7)-(2|mu6)+(3|mu5))

#--------------#
#standard SOCP #
#--------------#
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

dsocp=pic.Problem()
z=[dsocp.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]
lbda=dsocp.add_variable('lbda',s)
mu=dsocp.add_variable('mu',1)

dsocp.add_list_of_constraints(
        [abs(z[i]) < lbda[i] for i in range(s)],
        'i','[s]')

dsocp.add_constraint(
        cc+pic.sum([
        lbda[i]*AA[i].T*'|2|(3,1)'
        -AA[i].T*z[i]
        for i in range(s)],'i','[s]') < mu)
        
dsocp.set_objective('min',3*mu-pic.sum(
        [lbda[i]+z[i].T*AA[i]*'|1|(5,1)'
        for i in range(s)],'i','[s]')
        )

dsocp4=pic.Problem() #variant for the problem in TestSOCP4 with bounds on variables
z=[dsocp4.add_variable('z['+str(i)+']',AA[i].size[0]) for i in range(s)]
lbda=dsocp4.add_variable('lbda',s)
mu=dsocp4.add_variable('mu',1)
mu1=dsocp4.add_variable('mu1',1,lower = 0)
mu2=dsocp4.add_variable('mu2',1,lower = 0)

dsocp4.add_list_of_constraints(
        [abs(z[i]) < lbda[i] for i in range(s)],
        'i','[s]')

dsocp4.add_constraint(
        -cc+pic.sum([
        lbda[i]*AA[i].T*'|2|(3,1)'
        -AA[i].T*z[i]
        for i in range(s)],'i','[s]') < mu +mu1 * 'e_1(5,1)' -mu2 * 'e_3(5,1)')
        
dsocp4.set_objective('min',3*mu+0.58*mu1-0.59*mu2-pic.sum(
        [lbda[i]+z[i].T*AA[i]*'|1|(5,1)'
        for i in range(s)],'i','[s]')
        )
#-----------------#
#   SOCP + SDP    #
#-----------------#
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


coneQP=coneP.copy()
x=coneQP.get_variable('x')
y=coneQP.get_variable('y')
coneQP.add_constraint(y[0]**2+y[1]<2*y[2]-0.1)
coneQP.set_objective('min',(1|y)+0.6*y[0]**2)

from math import sqrt
dual_coneP=pic.Problem()
z=dual_coneP.add_variable('z',3)
w=dual_coneP.add_variable('w',1)
u=dual_coneP.add_variable('u',1)
mu=dual_coneP.add_variable('mu',1)
X=dual_coneP.add_variable('X',(4,4),'symmetric')

dual_coneP.add_constraint(abs(z)<1)
dual_coneP.add_constraint(w**2<4*u)
dual_coneP.add_constraint(X>>0)

dual_coneP.add_constraint(z[0]+mu==(AA[0]|X))
dual_coneP.add_constraint(z[1]+mu==(AA[1]|X))
dual_coneP.add_constraint(z[2]+mu-u==(AA[2]|X))

dual_coneP.add_constraint(AA[3]|X==1)

dual_coneP.set_objective('max',0.1+sqrt(0.1)*w-0.3*mu+('I'|X))

#--------------------------------------#
#       test a geometric program:
#--------------------------------------#

# min x/y+2y/x
#       x*y*y=1
#       (x,y>0)
#X=ln x, Y=ln y 
#  <=>
# exp max lse[X-Y,Y-X+ln(2)]
#       X+2Y=0

from math import log
gp=pic.Problem()
X=gp.add_variable('X',1)
Y=gp.add_variable('Y',1)
#gp.add_constraint(X+2*Y==0) #marche moins bien que les 2 LSE equivalentes ci-dessous (?)
gp.add_constraint(pic.lse(X+2*Y)<0)
gp.add_constraint(pic.lse(-X-2*Y)<0)
gp.set_objective('min',pic.lse((X-Y) & (Y-X+log(2))))


#----------------------------------------#
#  non-convex QP (bimatrix game)
#----------------------------------------#
import picos as pic
bim=pic.Problem()
AA=pic.new_param('A',[[ 1.21e+00,  5.90e-01, -1.45e+00],[-5.64e-01,  1.01e+00,  4.22e-01]])
BB=pic.new_param('B',[[ 1.30e-01, -1.59e+00,  2.06e+00],[ 1.06e+00, -5.56e-01, -1.76e-01]])
x=bim.add_variable('x',2)
y=bim.add_variable('y',3)
x.T*(AA+BB)*y
alpha=bim.add_variable('alpha',1)
beta=bim.add_variable('beta',1)
bim.add_constraint(AA*y<alpha)
bim.add_constraint(BB.T*x<beta)
bim.add_constraint(1|x==1)
bim.add_constraint(1|y==1)
bim.add_constraint(x>0)
bim.add_constraint(y>0)
bim.set_objective('max',x.T*(AA+BB)*y-alpha-beta)


avs=pic.tools.available_solvers()

def LP1Test(solver_to_test):
        print('LP1',solver_to_test)
        #1st test: c optimality single response
        primal=prob_LP_c.copy()
        try:
                primal.solve(solver=solver_to_test,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))
        
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=-14.
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
                
        try:
                z=[(cs.dual[1]-cs.dual[0]) for cs in primal.constraints]
                mu=[abs(zi) for zi in z]
        except TypeError:
                return (False,'no dual computed')
        
        zvar=prob_LP_dual_c.get_variable('z')
        muvar=prob_LP_dual_c.get_variable('mu')
        
        muvar.value=mu
        for i,zi in enumerate(z):
                zvar[i].value=zi

        dualf = prob_LP_dual_c.check_current_value_feasibility() 
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(prob_LP_dual_c.obj_value()+obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        return (True,primal.status)
        
def LP2Test(solver_to_test):
        print('LP2',solver_to_test)
        #1st test: LP in standard form
        primal=lp.copy()
        try:
                primal.solve(solver=solver_to_test,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))
        
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=12.5
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
                
        mu5 = primal.constraints[0].dual
        mu6 = primal.constraints[1].dual
        mu7 = primal.constraints[2].dual
        
        if (mu5 is None) or (mu6 is None) or (mu7 is None):
                return (False,'no dual computed')
                
        mu5var=dualp.get_variable('mu5')
        mu6var=dualp.get_variable('mu6')
        mu7var=dualp.get_variable('mu7')
        
        mu5var.value=mu5
        mu6var.value=mu6
        mu7var.value=mu7

        dualf=dualp.check_current_value_feasibility()
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(dualp.obj_value()-obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        return (True,primal.status)

def SOCP1Test(solver_to_test):
        print('SOCP1',solver_to_test)
        #first test (A optimality)
        primal=prob_multiresponse_A.copy()
        try:
                primal.solve(solver=solver_to_test,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))
        
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=1.0759874194855403
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
        
        Zvar=prob_dual_A.get_variable('Z')
        muvar=prob_dual_A.get_variable('mu')
        
        try:
                Z=[cvx.matrix(cs.dual[1:],Zvar[i].size) for i,cs in enumerate(primal.constraints)]
                mu=[cs.dual[0] for cs in primal.constraints]
        except TypeError:
                return (False,'no dual computed')
        
        muvar.value=mu
        for i,zi in enumerate(Z):
                Zvar[i].value=zi
                
        dualf=prob_dual_A.check_current_value_feasibility(tol=1e-5)
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(prob_dual_A.obj_value()-obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        return (True,primal.status)

def SOCP2Test(solver_to_test):        
        print('SOCP2',solver_to_test)
        #2d test (socp in standard form)
        primal=socp.copy()
        try:
                primal.solve(solver=solver_to_test,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))
             
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=8.921914163181004
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
        
                
        zvar=dsocp.get_variable('z')
        lbdavar=dsocp.get_variable('lbda')
        muvar=dsocp.get_variable('mu')
        try:
                z=[cvx.matrix(cs.dual[1:],zvar[i].size)
                for i,cs in enumerate(primal.constraints[:8])]
                lbda=[cs.dual[0] for cs in primal.constraints[:8]]
                mu=primal.get_constraint((1,)).dual[0]
        except TypeError:
                return (False,'no dual computed')
        
        lbdavar.value=lbda
        muvar.value=mu
        for i,zi in enumerate(z):
                zvar[i].value=zi
                
        dualf=dsocp.check_current_value_feasibility(tol=1e-5)
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(dsocp.obj_value()-obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        return (True,primal.status)

def SOCP3Test(solver_to_test):    
        print('SOCP3',solver_to_test)
        #3d test (socp with rotated cones)
        primal=prob_multiresponse_multiconstraints.copy()
        try:
                primal.solve(solver=solver_to_test,timelimit=1,maxit=100)
        except Exception as ex:
                return (False,repr(ex))
        
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=1.7997245328947509
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
        
        zvar=multiconstraints_dual.get_variable('z')
        alphavar=multiconstraints_dual.get_variable('alpha')
        muvar=multiconstraints_dual.get_variable('mu')
        tvar=multiconstraints_dual.get_variable('t')
        
        try:
                z=[2*cs.dual[1:4]
                for i,cs in enumerate(primal.constraints[:8])]
                alpha=[cs.dual[0]+cs.dual[-1] for cs in primal.constraints[:8]]
                mu   =[cs.dual[0]-cs.dual[-1] for cs in primal.constraints[:8]]
                t = primal.get_constraint((3,)).dual
        except TypeError:
                return (False,'no dual computed')
        
        alphavar.value=alpha
        muvar.value=mu
        tvar.value=t
        for i,zi in enumerate(z):
                zvar[i].value=zi
                
        dualf=multiconstraints_dual.check_current_value_feasibility(tol=1e-5)
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(multiconstraints_dual.obj_value()-obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
          
        return (True,primal.status)        

def SOCP4Test(solver_to_test,with_hardcoded_bound = True):
        print('SOCP4',solver_to_test)
        #4th test (socp in standard form, with additional variable bounds)
        primal=socp.copy()
        #solve the problem a first time to check if constaints can be added afterwards
        #try:
                #primal.solve()
        #except:
                #pass
        x = primal.get_variable('x')
        primal.add_constraint(x[1]<0.58)
        if with_hardcoded_bound:
                x.set_sparse_lower([3],[0.59])
        else:
                primal.add_constraint(x[3]>0.59)
        primal.set_objective('min',primal.objective[1])
        try:
                sol=primal.solve(solver=solver_to_test,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))

        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=8.88848803874566
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
        
        zvar=dsocp4.get_variable('z')
        lbdavar=dsocp4.get_variable('lbda')
        muvar=dsocp4.get_variable('mu')
        mu1var=dsocp4.get_variable('mu1')
        mu2var=dsocp4.get_variable('mu2')
        try:
                z=[cvx.matrix(cs.dual[1:],zvar[i].size)
                for i,cs in enumerate(primal.constraints[:8])]
                lbda=[cs.dual[0] for cs in primal.constraints[:8]]
                mu=primal.get_constraint((1,)).dual[0]
                mu1=primal.get_constraint((3,)).dual[0]
                if not(with_hardcoded_bound):
                        mu2=primal.get_constraint((4,)).dual[0]
        except TypeError:
                return (False,'no dual computed')
        
        lbdavar.value=lbda
        muvar.value=mu
        mu1var.value = mu1
        if with_hardcoded_bound:#TODO with reduced cost ?
                if solver_to_test=='cplex':
                        mu2var.value = primal.cplex_Instance.solution.get_reduced_costs(3)
                elif solver_to_test=='mosek7':
                        rc=[0]
                        try:
                                import mosek7 as mosek
                        except:
                                import mosek
                        primal.msk_task.getreducedcosts(mosek.soltype.itr,3,4,rc)
                        mu2var.value = rc[0]
                elif solver_to_test=='mosek6':
                        rc=[0]
                        import mosek
                        primal.msk_task.getreducedcosts(mosek.soltype.itr,3,4,rc)
                        mu2var.value = rc[0]
                        #mu2var.value = 10.338035946292555
                elif solver_to_test=='cvxopt':
                        mu2var.value = sol['cvxopt_sol']['z'][6]
                else: #'gurobi'
                        mu2var.value = primal.grbvar[3].RC
        else:
                mu2var.value = mu2
        
        for i,zi in enumerate(z):
                zvar[i].value=zi
        
        dualf=dsocp4.check_current_value_feasibility(tol=1e-5)
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(dsocp4.obj_value()+obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        
        return (True,primal.status)
        
def SDPTest(solver_to_test):
        print('SDP',solver_to_test)
        primal=prob_SDP_c.copy()
        try:
                primal.solve(solver=solver_to_test,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))
        
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=5.366615677650481
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
                
        muvar=prob_SDP_c_dual.get_variable('mu')
        try:
                mu=[cs.dual[0] for cs in primal.constraints[:8]]
                Z=primal.constraints[8].dual
        except TypeError:
                return (False,'no dual computed')
        if Z is None:
                return (False,'no dual computed')
                
        muvar.value=mu
        
        dualf=prob_SDP_c_dual.check_current_value_feasibility()
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        
        S = (prob_SDP_c_dual.constraints[0].Exp1-prob_SDP_c_dual.constraints[0].Exp2)
        dfinf = (abs(Z-S).value[0]/abs(S).value[0])
        if dfinf>1e-5:
                return (False,'not dual feasible|{0:1.0e}'.format(dfinf))
        dgap = abs(prob_SDP_c_dual.obj_value()-obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        return (True,primal.status)

def CONEPTest(solver_to_test):
        print('CONEP',solver_to_test)
        primal=coneP.copy()
        try:
                primal.solve(solver=solver_to_test,tol=1e-7,timelimit=1,maxit=50)
        except Exception as ex:
                return (False,repr(ex))
        
        primf = primal.check_current_value_feasibility()
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        obj=1.0159165875250857
        
        pgap = abs(primal.obj_value()-obj)/abs(obj)
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))
                
        zvar=dual_coneP.get_variable('z')
        muvar=dual_coneP.get_variable('mu')
        Xvar=dual_coneP.get_variable('X')
        uvar=dual_coneP.get_variable('u')
        wvar=dual_coneP.get_variable('w')
        
        try:
                mu=primal.constraints[1].dual
                z =primal.constraints[0].dual[1:]
                X =primal.constraints[3].dual        
                du=primal.constraints[2].dual       
                u =du[0] + du[-1]
                w = 2 * du[1]
        except TypeError:
                return (False,'no dual computed')
        
        if X is None:
                return (False,'no dual computed')
        
        muvar.value=mu
        zvar.value =z
        Xvar.value =X
        wvar.value =w
        uvar.value =u

        dualf=dual_coneP.check_current_value_feasibility()
        if not(dualf[0]):
                return (False,'not dual feasible|{0:1.0e}'.format(dualf[1]))
        dgap = abs(dual_coneP.obj_value()-obj)/abs(obj)
        if dgap >1e-5:
                return (False,'not dual optimal|{0:1.0e}'.format(dgap))
        return (True,primal.status)
        
def testOnlyPrimal(solver_to_test,primal,obj,tol=1e-6,maxit=50):
        primal2=primal.copy()
        try:
                primal2.solve(solver=solver_to_test,timelimit=1,maxit=maxit)
        except Exception as ex:
                return (False,repr(ex))

        primf = primal2.check_current_value_feasibility(tol=10*tol)
        if not(primf[0]):
                return (False,'not primal feasible|{0:1.0e}'.format(primf[1]))
        
        if obj==0:
                denom=1.
        else:
                denom=abs(obj)
        
        pgap = abs(primal2.obj_value()-obj)/denom
        if pgap>0.1:
                return (False,'failed')        
        elif pgap>1e-5:
                return (False,'not primal optimal|{0:1.0e}'.format(pgap))

        return (True,primal2.status)
                
def QCQPTest(solver_to_test):
        print('QCQP',solver_to_test)
        return testOnlyPrimal(solver_to_test,qcqp,
                                -12.433985877219854)
        
def MIXED_SOCP_QPTest(solver_to_test):
        print('MIXED SOCP QP',solver_to_test)
        return testOnlyPrimal(solver_to_test,soqcqp,
                                -6.8780810803741055)
        
def MIQCQPTest(solver_to_test):
        print('MIQCQP',solver_to_test)
        return testOnlyPrimal(solver_to_test,miqcqp,
                                -10.21427246841899,tol=1e-4,maxit=500)
         
def GPTest(solver_to_test):
        print('GP',solver_to_test)
        return testOnlyPrimal(solver_to_test,gp,
                                1.0397207708399179)
def MISOCPTest(solver_to_test):
        print('MISOCP',solver_to_test)
        return testOnlyPrimal(solver_to_test,prob_exact_c,
                                8.601831095537415,tol=1e-4,maxit=None)

def MIPTest(solver_to_test):
        print('MIP',solver_to_test)
        return testOnlyPrimal(solver_to_test,prob_exact_single_c,
                                5.48076923076923,tol=1e-4,maxit=500)
def CONEQCPTest(solver_to_test):
        print('CONEQCP',solver_to_test)
        return testOnlyPrimal(solver_to_test,coneQP,
                1.1541072108276682)

def NON_CONVEX_QPTest(solver_to_test):
        print('NONCONVEX QP',solver_to_test)
        return testOnlyPrimal(solver_to_test,bim,0.)
                
#tests with cvxopt
prob_classes = ['LP1','LP2','SOCP1',
                'SOCP2','SOCP3','SOCP4','SDP','coneP','coneQCP',
                'QCQP','Mixed_SOCP_QP','non_convex_Qp','GP',
                'MIP','MISOCP','MIQCQP']

conic_classes = ['LP1','LP2','SOCP1','SOCP2','SOCP3','SOCP4','SDP','coneP']


results={}
for solver in avs:
        #if solver == 'smcp':continue#TODO TMP
        results[solver]={}
        for pclas in prob_classes:
                results[solver][pclas]=eval(pclas.upper()+'Test')(solver)

for i in range(20): print()

print('Test of PICOS Version '+pic.__version__)
                
print('------------------------------------------------------------------')
print('----------------------  Results Summary  -------------------------')
print('------------------------------------------------------------------')
print()
#Display available solvers
print('list of available solvers:')
print('--------------------------')
for solv in avs:
        print('\t_'+solv)

print()  
        
linesep='+---------------+'+'----------+'*len(avs)
emptyln='|               |'+'          |'*len(avs)
header= '| problem class |'
for solver in avs:
        #if solver == 'smcp':continue#TODO TMP
        header+='{0:^10}|'.format(solver)


print(linesep)
print(emptyln)
print(header)
print(emptyln)
print(linesep)

for pclas in prob_classes:
        clasln='|{0:^15}|'.format(pclas)
        for solver in avs:
                #if solver == 'smcp':continue#TODO TMP
                if results[solver][pclas][0]:
                        if pclas in conic_classes:
                                clasln+='    OK*   |'
                        else:
                                clasln+='    OK    |'
                else:
                        err=results[solver][pclas][1]
                        if 'NotAppropriateSolverError' in err:
                                clasln+='          |'
                        elif 'NonConvexError' in err:
                                clasln+='          |'
                        elif 'DualizationError' in err:
                                clasln+='          |'
                        elif 'no Primals' in err:
                                clasln+='  failed  |'
                        elif 'primal' in err:
                                inf = err.split('|')[1]
                                if 'feasible' in err:
                                        clasln+='Pinf:'+inf+'|'
                                else:
                                        clasln+='gap: '+inf+'|'
                        elif 'no dual' in err:
                                clasln+='OK(nodual)|'
                        elif 'dual' in err:
                                inf = err.split('|')[1]
                                if 'feasible' in err:
                                        clasln+='Dinf:'+inf+'|'
                                else:
                                        clasln+='Dgap:'+inf+'|'
                        else:
                                clasln+='  failed  |'
        print(clasln)
        print(linesep)
        
print()
print('explanation: OK*        = test passed (optimal primal and dual variables computed).')
print('             OK         = test passed (only the primal vars are computed for this class of problems).')
print('             OK(nodual) = Optimal primal solution, but no dual variables were computed')
print('             Dinf:err   = Optimal primal solution, but dual infeasibility in the order of (err)')
print('             Dgap:err   = Optimal primal solution, but duality gap in the order of (err)')
print('             Pinf:err   = Primal solution with has an infeasibility in the order of (err)')
print('             gap: err   = Feasible but suboptimal solution (gap in the order of (err)')
print('             <blank>    = class of problem not handled by this solver')
print('             failed     = The problem was not solved')

