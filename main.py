
from functools import partial
import sys

#####
##
## code for generation
##
## go to end of file for specification
##
######

_executed = dict()
def once(func):
    name = func.__name__
    def wrapper(*args, **kwargs):
        key = (name, args, tuple(kwargs.items()))
        if key in _executed:
            return _executed[key]
        ret = func(*args, **kwargs)
        _executed[key] = ret
        return ret
    return wrapper


# size of target (0 * P_size)
P_size = int(sys.argv[1])
n = int(sys.argv[2])
names = [chr(ord('A')+i) for i in range(n)]

mode_Pn_fixed = False

def vec(v):
    return f"({','.join(v)})"
def repl(i, val, d=names):
    ret = d.copy()
    ret[i] = str(val)
    return vec(ret)

spec = ["(0;1)"] * n
print(f"node({vec(spec)}).")

spec = ["a"]*P_size + ["(a;0;1)"]*(n-P_size-1) + ["a"]
print(f"subspace({vec(spec)}).")

spec = ["_"]*n
for i in range(P_size):
    print(f"notP(ZG) :- terminal(ZG,{repl(i,1,spec)}).")
if mode_Pn_fixed:
    print(f"notP(ZG) :- terminal(ZG,{repl(i,n,spec)}).")
spec = spec[:-1]
for i in range(P_size):
    print(f"notP(ZG) :- terminal(ZG,{repl(i,1,spec)}).")

print("""
match(0,0).
match(1,1).
match(a,(0;1)).
""")

##
## generate transition graph
##
for i in range(n-1):
    print(f"{{ edge({vec(names)},{repl(i,'1-'+names[i])}) }} :- node({vec(names)}).")
x0 = repl(n-1,0)
x1 = repl(n-1,1)
print(f"1 {{ edge({x0},{x1}); edge({x1},{x0}) }} 1 :- node({repl(n-1,'_')}).")
print()

x = vec(names)
y = vec([f"{i}2" for i in names])

@once
def merge_general_asynchronous():
    print("% general asynchronous")
    cond = [f"edge({x},{repl(i,'1-'+a)}) : {a} != {a}2" for i,a in enumerate(names)]
    print(f"edge({x},{y}) :- node({x}), node({y}), {x} != {y} ;")
    print("\t\t" + " ;\n\t\t".join(cond) + ".")
    print()


rx = vec(names[:-1])

##
## compute subgraphs
##
mx = vec([f"M{i}" for i in names])
matches = ", ".join([f"match(M{i},{i})" for i in names])
print("% compute sub-graphs")
print(f"node({mx},{x}) :- subspace({mx}), node({x}), {matches}.")
print("edge(S,X,Y) :- edge(X,Y), node(S,X), node(S,Y), subspace(S).")
print()

##
## terminal nodes
##

print("""% compute terminal nodes
tc(G,X,Y) :- edge(G,X,Y).
tc(G,X,Y) :- tc(G,X,Z), edge(G,Z,Y).
nt(G,X) :- tc(G,X,Y), not tc(G,Y,X).
terminal(G,X) :- node(G,X), not nt(G,X).
""")

##
## reduction
##
rx = vec(names[:-1])
ry = vec([f"{i}2" for i in names[:-1]])
rx_r = vec(names[:-1]+["R"])
ry_r = vec([f"{i}2" for i in names[:-1]]+["R"])
xr = repl(n-1,'_')
print("% reduction")
print(f"node(r,{rx}) :- node({xr}).")
print(f"repr({rx},{names[-1]}) :- edge({xr},{x}).")
print(f"edge(r,{rx},{ry}) :- edge({rx_r},{ry_r}), repr({rx},R).")
print()

@once
def compute_reduction_strategies():
    print("% reduction strategies")
    spec = ["a"]*P_size + ["(a;0;1)"]*(n-P_size-1)
    print(f"subspace({vec(spec)}).")
    rmx = vec([f"M{i}" for i in names[:-1]])
    rmatches = ", ".join([f"match(M{i},{i})" for i in names[:-1]])
    print("% reduced sub-graphs")
    print(f"node({rmx},{rx}) :- subspace({rmx}), node(r,{rx}), {rmatches}.")
    print("edge(S,X,Y) :- edge(r,X,Y), node(S,X), node(S,Y), subspace(S).")
    print()

##
## Update modes
##

@once
def asynchronous_copy():
    print("% make asynchronous version of subgraphs")
    print("node((async,S),X) :-  node(S,X), subspace(S).")
    for i, a in enumerate(names):
        xa = repl(i, '1-'+a)
        print(f"edge((async,S),{x},{xa}) :- subspace(S), edge(S,{x},{xa}).")
    m = n-1
    for i, a in enumerate(names[:m]):
        rxa = repl(i, '1-'+a, names[:m])
        print(f"edge((async,S),{rx},{rxa}) :- subspace(S), edge(S,{rx},{rxa}).")

@once
def synchronous_dynamics():
    print("% make synchronous version of subgraphs")
    print("node((sync,S),X) :-  node(S,X), subspace(S).")
    for w in [names, names[:-1]]:
        c = []
        x = vec(w)
        y = vec([f"{a}2" for a in w])
        c += [f"{a}2={a}1:edge(S,{x},{repl(i,a+'1',w)})" for i, a in enumerate(w)]
        for v in [0,1]:
            c += [f"{a}2={v}:{a}={v},not edge(S,{x},{repl(i,1-v,w)})" for i, a in enumerate(w)]
        print(f"edge((sync,S),{x},{y}) :- {x} != {y}, subspace(S), node(S,{x}), node(S,{y});", "; ".join(c) + ".")
    print()

###
### Propagation
###
@once
def compute_propagations(scope, trivial_only=True, forall=False):
    assert scope in ["reduced", "initial"]
    in_reduced = scope == "reduced"
    print("%")
    print(f"% propagations in {scope}")
    print("%")
    k = n - P_size + (0 if in_reduced else 1)
    prefix = scope
    porder = f'porder_{prefix}'
    prop = f'prop_{prefix}'
    if forall:
        m = (n-1) if in_reduced else n
        s = vec([f"S{a}" for a in names[:m]])
        print(f"{porder}(S,(S,1)) :- subspace(S), S={s}.")
        for i in range(1,k):
            print(f"{porder}((S,{i}),(S,{i+1})) :- subspace(S), S={s}.")
        print(f"g_{prop}(S) :- subspace(S), S={s}.")
        print(f"g_{prop}(ZG) :- {porder}(_,ZG).")
    else:
        starting = 'r' if in_reduced else vec(['a']*n)
        porders = [f"{porder}({prefix}{i},{prefix}{i+1})." for i in range(1,k)]
        print(f"{porder}({starting},{prefix}1).\n{'\n'.join(porders)}")
        print(f"g_{prop}({starting}). g_{prop}(ZG) :- {porder}(_,ZG).")
    print()
    print("% propagations")
    m = n-1 if in_reduced else n
    a = ['_']*(m)
    r = names[:m]
    rx = vec(r)
    for i in range(m):
        print(f"{prop}(ZG,{i+1},T) :- edge(ZG,{repl(i,'_',r)},{repl(i,'T',r)}) : node(ZG,{repl(i,'1-T',r)}); "\
            f"node(ZG,{repl(i,'1-T',a)}); "\
            f"not edge(ZG,{repl(i,'T',a)},{repl(i,'1-T',a)}); "\
            f"T={0 if i < P_size else '(0;1)'}; g_{prop}(ZG).")
    print("% apply propagations")
    cond = [f"not {prop}(G1,{i+1},1-{r[i]})" for i in range(m)]
    print(f"node(ZG,{rx}) :- {porder}(G1,ZG), node(G1,{rx}), {', '.join(cond)}.")
    print("% corresponding subgraphs")
    print(f"edge(ZG,X,Y) :- node(ZG,X), node(ZG,Y), edge(G1,X,Y), {porder}(G1,ZG).")
    print()
    if not trivial_only and not forall:
        print("% find subspace to propagate")
        print(f"{{ {prop}({starting},I,(0;1)) }} 1 :- I = {P_size+1}..{m}.")
        print()
    return prop

def _dynamics_propagate(scope, trivial_only=False):
    prop = compute_propagations(scope, trivial_only=trivial_only)
    for i in range(P_size):
        print(f":- not {prop}(ZG,{i+1},0) : g_{prop}(ZG).")
    print()
initial_propagates = partial(_dynamics_propagate, "initial")
reduced_propagates = partial(_dynamics_propagate, "reduced")

def _dynamics_do_not_propagate(scope, trivial_only=False):
    prop = compute_propagations(scope, forall=not trivial_only)
    print(f"% no {'trivial ' if trivial_only else ''}propagation to P in {scope}")
    cond = [f"{prop}(ZG,{i+1},0)" for i in range(P_size)]
    cond += [f"g_{prop}(ZG)"]
    print(f":- {', '.join(cond)}.")
    print()
initial_does_not_propagate = partial(_dynamics_do_not_propagate, "initial")
reduced_does_not_propagate = partial(_dynamics_do_not_propagate, "reduced")


@once
def compute_influence_graph():
    for i in range(n):
        x0 = repl(i, 0)
        x1 = repl(i, 1)
        print(f"pos({i+1},{x0}) :- edge({x0},{x1}).")
        print(f"pos({i+1},{x1}) :- node({x1}), not edge({x1},{x0}).")
    for i in range(n):
        print(f"influence({i+1},J,2*{names[i]}-1) :- pos(J,{x}), not pos(J,{repl(i,'1-'+names[i])}).")

###############
######
####


###
#####
########  possible properties
#####
###

def enforce_MTS_preservation():
    compute_influence_graph()
    print(f"""
% Call I and J the set of regulators and targets of n respectively, and assume n ∈ / I .
% Suppose that I does not contain any regulator of variables in J . Then the minimal trap spaces of f
% are strictly preserved by the elimination of n.
:- influence(I,{n},_), influence({n},J,_), influence(I,J,_).
""")


def last_node_has_only_monotone_influences():
    compute_influence_graph()
    print(f":- influence({n},I,1), influence({n},I,-1).")

def last_node_is_mediator():
    compute_influence_graph()
    print(f":- not influence({n},_,_).")
    print(f":- not influence(_,{n},_).")
    print(f":- influence({n},I,S), influence({n},J,S2), (I,S) != (J,S2).")
    print(f":- influence(I,{n},S), influence(J,{n},S2), (I,S) != (J,S2).")

def last_node_has_disjoint_I_J():
    compute_influence_graph()
    print(f":- influence(I,{n},S), influence(J,{n},S2), I = J.")

def no_positive_cycles_of_length2():
    print("""
    :- influence(I,J,S), influence(J,I,S2), J != I, S*S2 = 1.
    """)

def no_synchronism_sensitivity():
    # WARNING: this is not tested at all!
    print("""
    e_path(I,J,S,1) :- influence(I,J,S), I < J.
    e_path(I,J,S1*S2,L+1) :- e_path(I,K,S1,L), influence(K,J,S2), K < J.
    e_cycle(I,S1*S2,L+1) :- e_path(I,J,S1,L), influence(J,I,S2).
    e_cycle(I,S,1) :- influence(I,I,S).

    % no positive feedback of even length
    :- e_cycle(I,1,L), L \\ 2 = 0.
    % no negative feedback of odd length
    :- e_cycle(I,-1,L), L \\ 2 = 1.
    """)

def initial_has_trivial_strategy(copy=None):
    spec = ["a"]*n
    graph = "S" if copy is None else f"({copy},S)"
    print(f"""
    % there is a trivial control strategy
    :- subspace(S), notP({graph}), S={vec(spec)}.
    """)

def initial_has_no_strategy(copy=None):
    spec = vec(names)
    graph = "S" if copy is None else f"({copy},S)"
    print(f"""
    :- subspace(S), not notP({graph}), S={spec}.
    """)

def reduction_has_no_strategy(copy=None):
    compute_reduction_strategies()
    spec = vec(names[:-1])
    graph = "S" if copy is None else f"({copy},S)"
    print(f"""
    :- subspace(S), not notP({graph}), S={spec}.
    """)

def reduction_has_strategy(copy=None):
    compute_reduction_strategies()
    spec = vec(names[:-1])
    graph = "S" if copy is None else f"({copy},S)"
    scope = "default" if copy is None else copy
    print(f"""
    strategy({scope},S) :- subspace(S), not notP({graph}), S={spec}.
    :- not strategy({scope},_).
    """)

def reduction_has_new_strategy(copy=None):
    compute_reduction_strategies()
    spec = vec(names[:-1])
    aspec = vec(names[:-1]+["a"])
    graph = "S" if copy is None else f"({copy},S)"
    if copy is not None:
        aspec = f"({copy},{aspec})"
    scope = "default" if copy is None else copy
    print(f"""
    strategy({scope},S) :- subspace(S), not notP({graph}), S={spec}.
    extra({scope},{spec}) :- strategy({scope},{spec}), notP({aspec}).
    :- not extra(_,_).
    """)

def minimize_influence_graph():
    compute_influence_graph()
    print("#minimize { 1,I,J,S: influence(I,J,S) }.")

"""
% there is a trivial control strategy
    strategy(S) :- subspace(S), not notP(S), S={x}.
    :- not strategy(_).

% strategy does not work in reduced
fail(S) :- strategy(S), notP({rx}), S={x}.
:- not fail(_).
"""

"""
% empty control works for initial network
:- subspace(ZG), notP(ZG), ZG={vec(['a']*n)}.
% empty control does not work for reduced
:- not notP({vec(['a']*(n-1))}).
"""

"""
not_trapspace(S) :- subspace(S), edge(X,Y), node(S, X), not edge(S,X,Y).
is_trapspace(S) :- not not_trapspace(S), subspace(S).
not_mintrap(S) :- is_trapspace(S), is_trapspace(S2), smaller(S2,S).
is_mintrap(S) :- is_trapsace(S), not not_mintrap(S).

"""

###
###
###


if __name__ == "__main__":
    ########################
    #### SPECIFICATION #####
    ########################

    #enforce_MTS_preservation()
    #last_node_is_mediator()
    #last_node_has_only_monotone_influences()
    #last_node_has_disjoint_I_J()
    #no_positive_cycles_of_length2() # encourage general = asynchronous

    #initial_does_not_propagate(); reduced_propagates()
    #enforce_MTS_preservation(); initial_does_not_propagate(); reduced_propagates()

    """
    enforce_MTS_preservation()
    last_node_is_mediator()
    merge_general_asynchronous()
    asynchronous_copy()
    synchronous_dynamics()
    for copy in [None, "async", "sync"]:
        initial_has_trivial_strategy(copy)
        reduction_has_no_strategy(copy)
    """

    #reduction_has_strategy()
    #reduction_has_new_strategy()

    #dynamics_do_not_propagate("initial")
    #dynamics_propagate("reduced")

    initial_has_trivial_strategy(); reduction_has_no_strategy()

    #asynchronous_copy()
    #initial_has_trivial_strategy("async")
    #reduction_has_no_strategy("async")
    #initial_has_no_strategy("async")
    #reduction_has_strategy("async")

    #minimize_influence_graph()

    #enforce_MTS_preservation(); last_node_is_mediator(); initial_has_no_strategy(); reduction_has_strategy(); merge_general_asynchronous()
    #print(":- 14 #count { 1,I,J,S: influence(I,J,S) }.")

# vi:tw=0:
