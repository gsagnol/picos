import matplotlib.pyplot as plt
#display the graph
node_colors=['w']*N
node_colors[s]='g' #source is green
node_colors[t]='b' #sink is blue

pos=nx.spring_layout(G)
#edges
nx.draw_networkx(G,pos,
                edgelist=[e for e in G.edges() if f[e].value[0]>0],
                node_color=node_colors)


labels={e:'{0}/{1}'.format(f[e],c[e]) for e in G.edges() if f[e].value[0]>0}
#flow label
nx.draw_networkx_edge_labels(G, pos,
                        edge_labels=labels)
plt.show()