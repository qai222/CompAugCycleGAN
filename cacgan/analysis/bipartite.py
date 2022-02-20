import logging
import os
import random

import networkx as nx
from networkx.readwrite.gexf import write_gexf

from cacgan.data import AtmoStructureGroup, AtmoStructure
from cacgan.utils import composition_elements

this_dir = os.path.abspath(os.path.dirname(__file__))


def get_chemsys_string(a: AtmoStructure):
    return "-".join(sorted(composition_elements(a.composition)))


def get_chemsys_fset(a: AtmoStructure):
    return frozenset(composition_elements(a.composition))


def group2bg(g: AtmoStructureGroup):
    bg = nx.Graph()
    for s in g:
        bg.add_node(s.amine, bipartite="amine")
        bg.add_node(get_chemsys_string(s), bipartite="system")
        bg.add_edge(s.amine, get_chemsys_string(s))
    for n, d in bg.nodes(data=True):
        bg.nodes[n]['neighbors_count'] = len(list(bg.neighbors(n)))
    return bg


def get_amine_chemsys_bipartite_graph(master_group):
    unique_amines = set([s.amine for s in master_group])
    unique_chemsys_string = set([get_chemsys_string(s) for s in master_group])
    num_of_amine = len(unique_amines)
    num_of_chemsys_string = len(unique_chemsys_string)

    logging.warning("num of amines: {}".format(num_of_amine))
    logging.warning("num of chemical systems: {}".format(num_of_chemsys_string))

    # write bipartite graph
    logging.warning("writing bipartite graph in gexf format...")
    gexf_filename = "bipartite.gexf"
    bipartite_G = group2bg(master_group)
    write_gexf(bipartite_G, gexf_filename)
    logging.warning("bipartite graph saved at: {}/{}".format(os.getcwd(), gexf_filename))

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
    logging.warning("finding 2 edges for randomly selected 2 chemsys 1 amine: {}".format(nfound / ntrials))

    # how about the inverse?
    nfound = 0
    for i in range(ntrials):
        amine1, amine2 = random.sample(nodes_amine, 2)
        chemsys1 = random.choice(nodes_system)
        if chemsys1 in bipartite_G[amine1] and chemsys1 in bipartite_G[amine2]:
            nfound += 1
    logging.warning("finding 2 edges for randomly selected 1 chemsys 2 amines: {}".format(nfound / ntrials))

    # how about 1 amine and 1 chemsys
    nfound = 0
    for i in range(ntrials):
        chemsys1 = random.choice(nodes_system)
        amine1 = random.choice(nodes_amine)
        if chemsys1 in bipartite_G[amine1]:
            nfound += 1
    logging.warning("finding 1 edge for randomly selected 1 chemsys 1 amine: {}".format(nfound / ntrials))

    # how about 2 amines sharing at least 1 chemsys
    nfound = 0
    for i in range(ntrials):
        amine1, amine2 = random.sample(nodes_amine, 2)
        if set(bipartite_G.neighbors(amine1)).intersection(set(bipartite_G.neighbors(amine2))):
            nfound += 1
    logging.warning("finding 2 amines sharing >= 1 chemsys: {}".format(nfound / ntrials))
    # num of amines: 349
    # num of chemsys: 685
    # finding 2 edges for randomly selected 2 chemsys 1 amine 0.000982
    # finding 2 edges for randomly selected 1 chemsys 2 amines 0.000596
    # finding 1 edge for randomly selected 1 chemsys 1 amine 0.01026
    # finding 2 amines sharing >= 1 chemsys 0.226437


def load_amine_chemsys_bipartite() -> nx.Graph:
    from networkx.readwrite.gexf import read_gexf
    bg_defaul_path = os.path.join(this_dir, "bipartite.gexf")
    try:
        bg = read_gexf(bg_defaul_path)
        logging.warning("loaded gexf file at: {}".format(bg_defaul_path))
    except:
        logging.warning("gexf file not found at: {}".format(bg_defaul_path))
        logging.warning("regenerate gexf file...")
        bg = get_amine_chemsys_bipartite_graph(AtmoStructureGroup.from_csv())
    return bg


def draw_bipartite(bg: nx.Graph):
    from cacgan.analysis.nxviz_plot.plot import CircosPlot
    c = CircosPlot(bg, node_color='bipartite', node_grouping="bipartite", node_order='neighbors_count')
    c.draw()
    return c.ax
