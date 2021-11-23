from data import AtmoStructureGroup, AtmoStructure
import random
from networkx.readwrite.gexf import write_gexf
from utils import composition_elements

import networkx as nx

def get_chemsys_string(a: AtmoStructure):
    return "-".join(sorted(composition_elements(a.composition)))


def get_chemsys_fset(a: AtmoStructure):
    return frozenset(composition_elements(a.composition))

def group2bg(g:AtmoStructureGroup):
    bg = nx.Graph()
    for s in g:
        bg.add_node(s.amine, bipartite="amine")
        bg.add_node(get_chemsys_string(s), bipartite="system")
        bg.add_edge(s.amine, get_chemsys_string(s))
    for n, d in bg.nodes(data=True):
        bg.nodes[n]['neighbors_count'] = len(list(bg.neighbors(n)))
    return bg


if __name__ == '__main__':

    master_group = AtmoStructureGroup.from_csv()

    # how many unique nodes?
    unique_amines = set([s.amine for s in master_group])
    unique_chemsys_string = set([get_chemsys_string(s) for s in master_group])
    num_of_amine = len(unique_amines)
    num_of_chemsys_string = len(unique_chemsys_string)
    print("num of amines:", num_of_amine)
    print("num of chemsys:", num_of_chemsys_string)

    # write bipartite graph
    bipartite_G = group2bg(master_group)
    write_gexf(bipartite_G, "bipartite.gexf")

    # calculate probability
    random.seed(42)
    nodes_amine = [n for n, d in bipartite_G.nodes(data=True) if d['bipartite'] == "amine"]
    nodes_system = [n for n, d in bipartite_G.nodes(data=True) if d['bipartite'] == "system"]
    ntrials = int(1e6)
    # what is the probability of finding 2 edges for randomly selected 2 chemsys 1 amine
    nfound = 0
    for i in range(ntrials):
        amine1 = random.choice(nodes_amine)
        chemsys1, chemsys2 = random.sample(nodes_system, 2)
        if chemsys1 in bipartite_G[amine1] and chemsys2 in bipartite_G[amine1]:
            nfound += 1
    print("finding 2 edges for randomly selected 2 chemsys 1 amine", nfound/ntrials)

    # how about the inverse?
    nfound = 0
    for i in range(ntrials):
        amine1, amine2 = random.sample(nodes_amine, 2)
        chemsys1 = random.choice(nodes_system)
        if chemsys1 in bipartite_G[amine1] and chemsys1 in bipartite_G[amine2]:
            nfound += 1
    print("finding 2 edges for randomly selected 1 chemsys 2 amines", nfound/ntrials)

    # how about 1 amine and 1 chemsys
    nfound = 0
    for i in range(ntrials):
        chemsys1 = random.choice(nodes_system)
        amine1 = random.choice(nodes_amine)
        if chemsys1 in bipartite_G[amine1]:
            nfound += 1
    print("finding 1 edge for randomly selected 1 chemsys 1 amine", nfound/ntrials)

    # how about 2 amines sharing at least 1 chemsys
    nfound = 0
    for i in range(ntrials):
        amine1, amine2 = random.sample(nodes_amine, 2)
        if set(bipartite_G.neighbors(amine1)).intersection(set(bipartite_G.neighbors(amine2))):
            nfound += 1
    print("finding 2 amines sharing >= 1 chemsys", nfound / ntrials)

    """
    num of amines: 349
    num of chemsys: 685
    finding 2 edges for randomly selected 2 chemsys 1 amine 0.000982
    finding 2 edges for randomly selected 1 chemsys 2 amines 0.000596
    finding 1 edge for randomly selected 1 chemsys 1 amine 0.01026
    finding 2 amines sharing >= 1 chemsys 0.226437
    """