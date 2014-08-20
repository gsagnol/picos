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


#-------------#
#   min cut   #
#-------------#

mincut=pic.Problem()

#source and sink nodes
s=16
t=10

#convert the capacities as a picos expression
cc=pic.new_param('c',c)

#cut variable
d={}
for e in G.edges():
        d[e]=mincut.add_variable('d[{0}]'.format(e),1)
        
#potentials
p=mincut.add_variable('p',N)

#potential inequality
mincut.add_list_of_constraints(
        [d[i,j] > p[i]-p[j]
        for (i,j) in G.edges()],        #list of constraints
        ['i','j'],'edges')              #indices and set they belong to

#one-potential at source
mincut.add_constraint(p[s]==1)
#zero-potential at sink
mincut.add_constraint(p[t]==0)
#nonnegativity
mincut.add_constraint(p>0)
mincut.add_list_of_constraints(
        [d[e]>0 for e in G.edges()],    #list of constraints
        [('e',2)],                      #e is a double index (origin and desitnation of the edges)
        'edges'                         #set the index belongs to
        )

#objective
mincut.set_objective('min',
                     pic.sum([c[e]*d[e] for e in G.edges()],
                             [('e',2)],'edges')
                     )
        
#print mincut
mincut.solve(verbose=0)

cut=[e for e in G.edges() if d[e].value[0]==1]

#display the graph
import pylab
fig=pylab.figure(figsize=(11,8))


node_colors=['w']*N
node_colors[s]='g' #source is green
node_colors[t]='b' #sink is blue


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
 
 
#edges (not in the cut)
nx.draw_networkx(G,pos,
                edgelist=[e for e in G.edges() if e not in cut],
                node_color=node_colors)

#edges of the cut
nx.draw_networkx_edges(G,pos,
                edgelist=cut,
                edge_color='r')
                
#hide axis
fig.gca().axes.get_xaxis().set_ticks([])
fig.gca().axes.get_yaxis().set_ticks([])
                
pylab.show()