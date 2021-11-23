from networkx.readwrite.gexf import read_gexf
from nxviz import CircosPlot
import matplotlib.pyplot as plt

bipartite_G = read_gexf("bg.gexf")

# draw circle bipartite
c = CircosPlot(bipartite_G, node_color='bipartite', node_grouping="bipartite", node_order='neighbors_count')
# this is an old function in nxviz only implemented in earlier versions (e.g. 0.3.7 with python3.6)
# better build another env for this script
c.draw()
plt.savefig("bipartite.png", dpi=600, )
