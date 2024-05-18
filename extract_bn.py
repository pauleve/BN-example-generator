import itertools
import sys
import networkx as nx
from colomoto.minibn import BooleanNetwork
from mpbn.converters import *

gname = sys.argv[1]
fname = gname.replace(".dot", ".bnet")
assert fname != gname
igname = gname.replace(".dot", "_ig.dot")

g = nx.nx_pydot.read_dot(gname)
n = len(next(iter(g.nodes())))

if len(g.nodes()) != 2**n:
    for x in itertools.product("01", repeat=n):
        x = "".join(x)
        if x not in g:
            g.add_node(x)

names = [f"x{i+1}" for i in range(n)]
f = bn_of_asynchronous_transition_graph(g, names)
f.save(fname)
print()
print(f)
nx.nx_pydot.write_dot(f.influence_graph(), igname)

f = BooleanNetwork(f)

for mode in ["asynchronous", "general", "synchronous"]:
    print(f"Attractors {mode}")
    for a in nx.attracting_components(f.dynamics(mode)):
        print("\t-", a)
print()
