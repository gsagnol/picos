import picos as pic
import cvxopt as cvx
P = pic.Problem()
X = P.add_variable('X',(3,3),'hermitian')
A = cvx.normal(3,3)
#A = A + A.T
B = cvx.normal(3,3)
#B = B - B.T
C = cvx.normal(3,3)
C = C + C.T
D = cvx.normal(3,3)
D = D - D.T
exp = ((A+1j*B) | X) // ((C+1j*D) | X)


Xr = P.add_variable('X_RE',(3,3),'symmetric')
Xi = P.add_variable('X_IM',(3,3))
dc = {'X_RE':Xr, 'X_IM':Xi}

Y = cvx.normal(3,3)
Y = Y + Y.T
Z = cvx.normal(3,3)
Z = Z - Z.T

X.value = Y + 1j * Z
Xr.value = Y
Xi.value = Z

nexp = pic.tools._copy_exp_to_new_vars(exp,dc,complex=True)
nexp = pic.tools._copy_exp_to_new_vars(exp,dc,complex=False)

#TODO correct all signs
# take into account that row=vec(A) in factors[Z] means conj( <A.H,Z> )= <Z,A.H> ... or something similar ?

def trace(X):
        return sum([X[i,i] for i in range(min(X.size))])