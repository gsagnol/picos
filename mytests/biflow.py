import picos as pic
import networkx as nx
import cvxopt as cvx

"""
#directed
V = 5
N = 8
G = nx.random_graphs.gnm_random_graph(V,N,directed=True)
s = 0
t = sorted([(v,i) for (i,v) in nx.single_source_dijkstra_path_length(G,0).iteritems()])[-1][1]
"""

"""
#TOY
V=6
E = [(0,1),(1,2),(2,3),(0,4),(4,1),(4,3),(0,5),(5,2),(5,3)]
G = nx.DiGraph()
G.add_edges_from(E)
N = len(E)
s,t = (0,3)
"""

"""
#Directed acyclic
PP = []
l=0

V0=15
N0=30
while len(PP)<=2 or l<=2:
        G=nx.gnp_random_graph(V0,N0/float(V0*(V0-1)),directed=True)
        G = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
        E = G.edges()
        N = len(E)
        V = G.number_of_nodes()
        l,s,t = sorted([(v,i,j) for i,Di in nx.all_pairs_shortest_path_length(G).iteritems() for j,v in Di.iteritems()])[-1]
        PP = list(nx.all_simple_paths(G,s,t))
"""

#layers graph
layers = 4
#Nlayers = 3
max_conn_lays = 10
s = 0
E = [(0,1),(0,2),(0,3)]
last = 3
for i in range(layers):
        I = [last-2,last-1,last]
        J = [last+1,last+2,last+3]
        newE = []
        for k in range(max_conn_lays):
                e1 = I[int(cvx.uniform(1)[0]*3)]
                e2 = J[int(cvx.uniform(1)[0]*3)]
                if (e1,e2) not in newE:
                        newE.append((e1,e2))
        last+=3
        E.extend(newE)
t = last+1
E.extend([(J[0],t),(J[1],t),(J[2],t)])
G = nx.DiGraph()
G.add_edges_from(E)
V = G.number_of_nodes()
N = G.number_of_edges()




PP = list(nx.all_simple_paths(G,s,t))
E = G.edges()

"""
#flow formulation ? CHECK
import picos as pic
P = pic.Problem()
X = P.add_variable('X',(N,N),'symmetric')

diagdict = {(e0,e1):X[i,i] for (i,(e0,e1)) in enumerate(E)}
P.add_constraint(pic.flow_Constraint(G,diagdict,s,t,1,None,'G'))

rowdict = []
for i in range(N):
        rowdict.append({(e0,e1):X[i,j] for (j,(e0,e1)) in enumerate(E)})
        P.add_constraint(pic.flow_Constraint(G,rowdict[i],s,t,X[i,i],None,'G'))



P.set_objective('min', (L|X))
"""

MP = []
EP = []
for p in PP:
        edp = [(e0,e1) for (e0,e1) in zip(p[:-1],p[1:])]
        iedp = [E.index(e) for e in edp]
        EP.append(iedp)
        ep = cvx.spmatrix([1]*len(iedp),iedp,[0]*len(iedp),(N,1))
        MP.append(ep*ep.T)

xx = cvx.uniform(len(PP),1)
xx = xx/sum(xx)

X0 = sum([xx[i]*MP[i] for i in range(len(PP))])

A = nx.incidence_matrix(G,oriented=True)
A = cvx.matrix(A.todense())
ss,tt = G.nodes().index(s),G.nodes().index(t)
est = cvx.spmatrix([-1,1],[ss,tt],[0,0],(V,1))
diag = lambda M : cvx.matrix([M[i,i] for i in range(M.size[0])])

A*X0 - est*diag(X0).T
A*diag(X0) - est


dmax = 0
Lmax = 0
for iter in range(40):
        L = cvx.normal(N,N)
        L = L + L.T
        import picos as pic
        P = pic.Problem()
        X = P.add_variable('X',(N,N),'symmetric')
        P.add_constraint(A*X == est*pic.diag_vect(X).T)
        P.add_constraint(A*pic.diag_vect(X)==est)
        P.add_constraint(X<1)
        #P.add_constraint(X>>0)
        P.add_constraint(X>0)
        #P.set_objective('max', (L|X))
        P.set_objective('max', (L|X))
        #P.add_constraint(X[1,4]==0)
        #P.add_constraint(X[2,5]==0)
        #P.add_constraint(X[2,7]==0)
        P.solve()

        H = cvx.sparse([[MMP[:]] for MMP in MP])
        bX = X.value[:]
        Q = pic.Problem()
        lbd = Q.add_variable('lbd',H.size[1],lower=0)
        delta = Q.add_variable('delta',1)
        Q.add_constraint(pic.norm(H*lbd-bX,1)<delta)
        Q.minimize(delta)
        delta = delta.value[0]
        if delta>dmax:
                dmax = delta
                Lmax = L
        if delta > 1e-3:
                break


"""
#series of parallel arcs

count=0
NNN = 10
for iter in range(NNN):
        V=6
        s=0
        c=2
        t=V-1
        A = cvx.spmatrix([],[],[],(V,c*(V-1)))
        v = cvx.spmatrix([],[],[],(c*(V-1),1))
        for i in range(V-1):
                v[c*i] = 1.
                for j in range(c):
                        A[i,c*i+j]=-1
                        A[i+1,c*i+j]=1
        X0 = v*v.T
        #TODO HERE, why is X0 not feasible ? 

        est = cvx.spmatrix([-1,1],[s,t],[0,0],(V,1))
        N = c*(V-1)

        L = cvx.normal(N,N)
        L = L + L.T
        import picos as pic
        P = pic.Problem()
        X = P.add_variable('X',(N,N),'symmetric')
        P.add_constraint(A*X == est*pic.diag_vect(X).T)
        P.add_constraint(A*pic.diag_vect(X)==est)
        P.add_constraint(X<1)
        P.add_constraint(X>>0)
        P.add_constraint(X>0)
        #P.set_objective('max', (L|X))
        P.set_objective('max', (L|X))
        #P.add_constraint(X[1,4]==0)
        #P.add_constraint(X[2,5]==0)
        #P.add_constraint(X[2,7]==0)
        P.solve()
        if any([abs(x-0.5)<0.03 for x in X.value]):
                count+=1
                break


print count/float(NNN)

MP = []
for iedp in itertools.product([0,1],[2,3],[4,5],[6,7],[8,9]):
        ep = cvx.spmatrix([1]*len(iedp),iedp,[0]*len(iedp),(N,1))
        MP.append(ep*ep.T) 
        
H = cvx.sparse([[MMP[:]] for MMP in MP])
bX = X.value[:]
Q = pic.Problem()
lbd = Q.add_variable('lbd',H.size[1],lower=0)
delta = Q.add_variable('delta',1)
Q.add_constraint(pic.norm(H*lbd-bX,1)<delta)
Q.minimize(delta)
"""