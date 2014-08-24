#dual fidelity
D = pic.Problem()
A = D.add_variable('A',(3,3),'hermitian')
C = D.add_variable('C',(3,3),'hermitian')


U = -0.5*cvx.spdiag([1]*3)
D.add_constraint( ((A & U) // (U.T & C)) >> 0 )
D.set_objective('min', (A|P) + (C|Q))

Fr = F.to_real()
Fr.solve()

F1 = Fr.constraints[0].dual[:6,:6]
F1a = Fr.constraints[0].dual[6:,6:]
F2a = Fr.constraints[0].dual[:6,6:]
F2 = Fr.constraints[0].dual[6:,:6]

#we should have F1=F1a and F2=-F2a.T, with exact eq for the second one
F1p = 0.5*(F1 + F1a)

#the dual of the cplx sdp should be [A,-I/2 ; -I/2, C] = 2*(F1p + 1j*F2a)
#-> take the value ((F1 + 1j*F2a) + (F1a + 1j*F2a).H) is not so bad


#signal recov
D = pic.Problem()
y = D.add_variable('y',n)

D.add_constraint(pic.tools.diag(y) << M)
D.set_objective('max',1|y)

Pr = P.to_real()
