#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import numpy.linalg as linalg
import cvxopt.base
import cvxopt.solvers

def mat_cplx_to_real(cmat):
    rmat = np.bmat([[ cmat.real, -cmat.imag ], [ cmat.imag, cmat.real ]])
    return rmat

def mat_real_to_cplx(rmat):
    w = rmat.shape[0] // 2
    h = rmat.shape[1] // 2
    cmat = rmat[:w,:h] + 1j*rmat[w:,:h]
    # preserve the normalization
    cmat *= 2
    return cmat

def make_F_real(Fx_list, F0_list):
    '''
    Convert F0, Fx arrays to real if needed, by considering C as a vector space
    over R.  This is needed because cvxopt cannot handle complex inputs.
    '''

    F0_list_real = []
    Fx_list_real = []
    for (F0, Fx) in zip(F0_list, Fx_list):
        if F0.dtype.kind == 'c' or Fx.dtype.kind == 'c':
            F0_list_real.append(mat_cplx_to_real(F0))

            mr = np.zeros((Fx.shape[0]*2, Fx.shape[1]*2, Fx.shape[2]))
            for i in range(Fx.shape[2]):
                mr[:, :, i] = mat_cplx_to_real(Fx[:, :, i])
            Fx_list_real.append(mr)
        else:
            F0_list_real.append(F0)
            Fx_list_real.append(Fx)

    assert len(F0_list_real) == len(F0_list)
    assert len(Fx_list_real) == len(Fx_list)
    return (Fx_list_real, F0_list_real)

def call_sdp(c, Fx_list, F0_list):
    '''
    Solve the SDP which minimizes $c^T x$ under the constraint
    $\sum_i Fx_i x_i - F0 \ge 0$ for all (Fx, F0) in (Fx_list, F0_list).
    '''

    for (k, (F0, Fx)) in enumerate(zip(F0_list, Fx_list)):
        assert linalg.norm(F0 - F0.conj().T) < 1e-10
        for i in range(Fx.shape[2]):
            assert linalg.norm(Fx[:,:,i] - Fx[:,:,i].conj().T) < 1e-10

    (Fx_list, F0_list) = make_F_real(Fx_list, F0_list)
    Gs = [cvxopt.base.matrix(Fx.reshape(Fx.shape[0]**2, Fx.shape[2])) for Fx in Fx_list]
    hs = [cvxopt.base.matrix(F0) for F0 in F0_list]

    sol = cvxopt.solvers.sdp(cvxopt.base.matrix(c), Gs=Gs, hs=hs)

    return sol

def random_sdp_operator(d):
    P = np.random.standard_normal((d, d)) + 1j * np.random.standard_normal((d, d))
    P = np.dot(P.conj().T, P)
    assert linalg.eigvalsh(P)[0] >= 0
    return P

if __name__ == "__main__":
    # Application: compute the fidelity between two positive semidefinite operators using the
    # SDP formulation described in John Watrous' lecture notes:
    # https://cs.uwaterloo.ca/~watrous/CS766/LectureNotes/08.pdf

    cvxopt.solvers.options['abstol'] = float(1e-8)
    cvxopt.solvers.options['reltol'] = float(1e-8)
    # For some reason cvxopt seems to have rather high error regardless of how abstol and
    # reltol are set.  Even 1e-5 is exceeded sometimes!
    err_tol = 1e-5

    d = 3
    P = random_sdp_operator(d)
    Q = random_sdp_operator(d)

    # Create a real basis for d*d complex matrices
    clen = 2*d*d
    X_basis = np.zeros((d, d, clen), dtype=complex)
    for i in range(d):
        for j in range(d):
            X_basis[i, j, (i*d+j)*2 ] = 1
            X_basis[i, j, (i*d+j)*2+1] = 1j

    # We want max( 1/2 * Tr(X) + 1/2 * Tr(X^*) ), which is equal to
    # max( real(Tr(X)) ).
    cost = np.zeros(clen)
    for c in range(clen):
        X = X_basis[:,:,c]
        cost[c] = -np.trace(X).real

    # Subject to [[P, X], [X^*, Q]] \succeq 0, equivalently
    # [[0, -X], [-X^*, 0]] - [[P, 0], [0, Q]] \preceq 0.
    zero_dd = np.zeros((d, d))
    Fx = np.zeros((d*2, d*2, clen), dtype=complex)
    for c in range(clen):
        X = X_basis[:, :, c]
        Fx[:, :, c] = np.bmat([[ zero_dd, -X ], [ -X.conj().T, zero_dd ]])
    F0 = np.bmat([[ P, zero_dd ], [ zero_dd, Q ]])

    sol = call_sdp(cost, [Fx], [F0])
    assert sol['status'] == 'optimal'

    # The fidelity between P, Q
    primal_val = -sol['primal objective']
    print('F(P,Q) =', primal_val)

    xvec = np.array(sol['x']).flatten()
    X = np.dot(X_basis, xvec)
    # Check that [[P, X], [X^*, Q]] \succeq 0.
    G = np.bmat([[ P, X ], [ X.conj().T, Q ]])
    assert linalg.eigvalsh(G)[0] > -err_tol

    # Verify primal value
    err = primal_val - (0.5*np.trace(X) + 0.5*np.trace(X.conj().T))
    assert np.abs(err) < err_tol

    # Verify dual solution
    rho = mat_real_to_cplx(np.array(sol['zs'][0]))
    # Need to multiply by two to make my formulation match the one described by Watrous.
    rho *= 2.0
    assert linalg.eigvalsh(rho)[0] > -err_tol
    assert linalg.norm(rho[:d, d:] + np.eye(d)) < err_tol
    assert linalg.norm(rho[d:, :d] + np.eye(d)) < err_tol
    Y = rho[:d, :d]
    Z = rho[d:, d:]
    dual_val = 0.5*np.sum(Y.conj() * P) + 0.5*np.sum(Z.conj() * Q)
    assert np.abs(primal_val - dual_val) < err_tol
