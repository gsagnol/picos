import picos as pic
import networkx as nx

#number of nodes
N=20

#Generate a graph with LCF notation (you can change the values below to obtain another graph!)
#We use this deterministic generator in order to have a constant inuput for doctest.
G=nx.LCF_graph(N,[1,3,14],5)
G=nx.DiGraph(G) #edges are bidirected

#generate edge capacities
c={}
for i,e in enumerate(G.edges()):
        c[e]=((-2)**i)%17 #an arbitrary sequence of numbers
        
#---------------#
#  MAXCUT SDP   #
#---------------#
import cvxopt as cvx
import cvxopt.lapack
import numpy as np

#make G undirected
G=nx.Graph(G)

#allocate weights to the edges
for (i,j) in G.edges():
        G[i][j]['weight']=c[i,j]+c[j,i]


maxcut = pic.Problem()
X=maxcut.add_variable('X',(N,N),'symmetric')

#Laplacian of the graph  
L=pic.new_param('L',1/4.*nx.laplacian(G))

#ones on the diagonal
maxcut.add_constraint(pic.tools.diag_vect(X)==1)
#X positive semidefinite
maxcut.add_constraint(X>>0)

#objective
maxcut.set_objective('max',L|X)

print maxcut
maxcut.solve(verbose = 0)

#Cholesky factorization
V=X.value

cvxopt.lapack.potrf(V)
for i in range(N):
        for j in range(i+1,N):
                V[i,j]=0

#random projection algorithm
#Repeat 100 times or until we are within a factor .878 of the SDP optimal value
count=0
obj_sdp=maxcut.obj_value()
obj=0
while (count <100 or obj<.878*obj_sdp):
        r=cvx.normal(20,1)
        x=cvx.matrix(np.sign(V*r))
        o=(x.T*L*x).value[0]
        if o>obj:
                x_cut=x
                obj=o
        count+=1

S1=[n for n in range(N) if x[n]<0]
S2=[n for n in range(N) if x[n]>0]
        
print 'partition of the nodes:'
print 'S1: {0}'.format(S1)
print 'S2: {0}'.format(S2)

cut = [(i,j) for (i,j) in G.edges() if x[i]*x[j]<0]

#display the cut
import pylab

fig=pylab.figure(figsize=(11,8))

#a Layout for which the graph is planar (or use pos=nx.spring_layout(G) with another graph)
pos={
 0: (0.07, 0.7),
 1: (0.18, 0.78),
 2: (0.26, 0.45),
 3: (0.27, 0.66),
 4: (0.42, 0.79),
 5: (0.56, 0.95),
 6: (0.6,  0.8),
 7: (0.64, 0.65),
 8: (0.55, 0.37),
 9: (0.65, 0.3),
 10:(0.77, 0.46),
 11:(0.83, 0.66),
 12:(0.90, 0.41),
 13:(0.70, 0.1),
 14:(0.56, 0.16),
 15:(0.40, 0.17),
 16:(0.28, 0.05),
 17:(0.03, 0.38),
 18:(0.01, 0.66),
 19: (0, 0.95)}

node_colors=[('g' if n in S1 else 'b') for n in range(N)]
                
nx.draw_networkx(G,pos,
                edgelist=[e for e in G.edges() if e not in cut],
                node_color=node_colors)

nx.draw_networkx_edges(G,pos,
                edgelist=cut,
                edge_color='r')
                
#hide axis
fig.gca().axes.get_xaxis().set_ticks([])
fig.gca().axes.get_yaxis().set_ticks([])
                
pylab.show()