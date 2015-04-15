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
#  max flow   #
#-------------#

maxflow=pic.Problem()
#source and sink nodes
s=16
t=10
#convert the capacities as a picos expression
cc=pic.new_param('c',c)

#flow variable
f={}
for e in G.edges():
        f[e]=maxflow.add_variable('f[{0}]'.format(e),1)


#flow value
F=maxflow.add_variable('F',1)

#upper bound on the flows
maxflow.add_list_of_constraints(
        [f[e]<cc[e] for e in G.edges()], #list of constraints
        [('e',2)],                       #e is a double index (start and end node of the edges)
        'edges'                          #set the index belongs to
        )
        
#flow conservation
maxflow.add_list_of_constraints(
     [   pic.sum([f[p,i] for p in G.predecessors(i)],'p','pred(i)')
      == pic.sum([f[i,j] for j in G.successors(i)],'j','succ(i)')
      for i in G.nodes() if i not in (s,t)],
        'i','nodes-(s,t)')

#source flow at s
maxflow.add_constraint(
      pic.sum([f[p,s] for p in G.predecessors(s)],'p','pred(s)') + F
      == pic.sum([f[s,j] for j in G.successors(s)],'j','succ(s)')
      )

#sink flow at t
maxflow.add_constraint(
      pic.sum([f[p,t] for p in G.predecessors(t)],'p','pred(t)')
      == pic.sum([f[t,j] for j in G.successors(t)],'j','succ(t)') + F
      )

#nonnegativity of the flows
maxflow.add_list_of_constraints(
        [f[e]>0 for e in G.edges()],    #list of constraints
        [('e',2)],                      #e is a double index (origin and desitnation of the edges)
        'edges'                         #set the index belongs to
        )

#objective
maxflow.set_objective('max',F)
        
#print maxflow
maxflow.solve(verbose=0)


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


nx.draw_networkx(G,pos,
                edgelist=[e for e in G.edges() if f[e].value[0]>0],
                node_color=node_colors)

                
labels={e:'{0}/{1}'.format(f[e],c[e]) for e in G.edges() if f[e].value[0]>0}
#flow label
nx.draw_networkx_edge_labels(G, pos,
                        edge_labels=labels)

#hide axis
fig.gca().axes.get_xaxis().set_ticks([])
fig.gca().axes.get_yaxis().set_ticks([])

                
pylab.show()