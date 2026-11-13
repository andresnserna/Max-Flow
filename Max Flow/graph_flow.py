# Dr Michaels
# graph_flow.py
# A weighted graph class built for MAXIMUM FLOW, using adjacency lists
# (a trimmed version of graph_sp.py: the shortest path code has been removed)
#
# In a flow network, the weight of each edge is its CAPACITY: the most flow that edge can carry.
# Capacities must not be negative, and should be whole numbers (see the NOTE on running times).
# Directed and undirected edges may be added. An undirected edge is two directed edges, one in
# each direction, each with the full capacity (a two way road).
# We will not allow multiple edges a -> b, and we ignore self loops. Edges a -> b AND b -> a
# may both exist, and max_flow handles that correctly.
#
# This file contains one method, max_flow, with three ways of choosing the next path.
# All three use the same idea (the Ford-Fulkerson method): while there is a path from the
# source to the sink with room left on every edge, push as much flow as fits along it.
#   "dfs"      Ford-Fulkerson with a depth first search for the path     O(E * f)   (f = the max flow)
#   "bfs"      Edmonds-Karp: a breadth first search, so the path with    O(V * E^2)
#              the fewest edges
#   "fattest"  the path that can carry the MOST flow (a Dijkstra-style   about O(E^2 log V log f)
#              search on bottleneck capacity)
# max_flow returns a FlowResult (see below). Set verbose=1 to print every path it uses,
# or verbose=2 to also print the residual graph before each search.
# For the shortest path algorithms, see graph_sp.py
#
# NOTE: "dfs" takes up to f augmentations when capacities are whole numbers, and f can be huge
# even when the graph is tiny, so its running time depends on the capacities and not just on
# V and E. "bfs" takes at most O(V * E) augmentations no matter what the capacities are.

import heapq
import random
import time
from collections import deque

INF = 2147483647   # "infinite" capacity; the same value as INF in GraphFlow.java, so both languages behave identically


class Graph:

    # Constructor for a capacity graph (directed and/or undirected edges)
    # It takes no input
    # We will store our node names in a dictionary as keys
    # The value at each key will be a dictionary
    # The dictionary will hold node names as keys, with capacities being the value at each key
    # NOTE: At most we can have n(n-1) directed edges in the graph, one per ordered pair (u, v)
    def __init__(self):
        self.nodes = {}
        self.num_nodes = 0
        self.num_edges = 0

    # Return the number of nodes in our graph
    def get_num_nodes(self):
        return self.num_nodes

    # Returns the number of edges in our graph
    def get_num_edges(self):
        return self.num_edges

    # Adds a node to our graph if it is not already in it
    def add_node(self, name):
        if not name in self.nodes:
            self.nodes[name] = {}
            self.num_nodes += 1

    # Checks to see if a given node is in our graph
    def has_node(self, name):
        if name in self.nodes:
            return True
        return False

    # Adds a directed edge node1 -> node2 with the given capacity
    # We will make sure that:
    # 1) Node1 != Node2
    # 2) That the edge node1 -> node2 does not already exist
    #    (node2 -> node1 may exist: the two edges are separate)
    def add_edge(self, node1, node2, capacity):
        if node1 == node2:
            return
        self.add_node(node1)
        self.add_node(node2)
        if node2 not in self.nodes[node1]:
            self.nodes[node1][node2] = capacity
            self.num_edges += 1

    # Adds an edge between node1 and node2 (stored as node1 -> node2 AND node2 -> node1)
    # Each of the two edges gets the full capacity. We will make sure that:
    # 1) Node1 != Node2
    # 2) That no edge already exists in either direction between the nodes
    # Each undirected edge counts as 2 edges in num_edges
    def add_undirected_edge(self, node1, node2, capacity):
        if node1 == node2:
            return
        self.add_node(node1)
        self.add_node(node2)
        if node2 not in self.nodes[node1] and node1 not in self.nodes[node2]:
            self.nodes[node1][node2] = capacity
            self.nodes[node2][node1] = capacity
            self.num_edges += 2

    # Returns every stored directed edge as a list of (u, v, capacity) tuples.
    # For a graph built with add_undirected_edge, each undirected edge appears TWICE this way
    def get_edges(self):
        result = []
        for node in self.nodes:
            for neighbor, capacity in self.nodes[node].items():
                result.append((node, neighbor, capacity))
        return result

    # ------------------------------------------------------------------
    # Maximum flow
    # ------------------------------------------------------------------
    # Finds the largest amount of flow that can be sent from source to sink without any edge
    # carrying more than its capacity, and without any other node keeping or creating flow.
    # Returns a FlowResult:
    #   value          the size of the maximum flow
    #   flow           flow[u][v] = how much flow the maximum flow sends along the edge u -> v
    #                  (there is an entry for every edge in the graph, 0 if it carries nothing)
    #   source_side    the set S of the minimum cut, as a sorted list: the nodes that can still be
    #                  reached from source once no more flow fits (the nodes NOT in S are the set T)
    #   cut_edges      the edges that go from S to T, as (u, v, capacity) tuples
    #   augmentations  how many paths the method used to get to the maximum flow
    # The value of the flow equals the total capacity of the cut_edges. A flow and a cut with the
    # same value prove each other: no flow can beat any cut, so this flow is the best possible.
    #
    # The idea: keep a RESIDUAL graph that says how much more can be pushed along each edge.
    #   residual[u][v] = (capacity of u -> v) - (flow on u -> v) + (flow on v -> u)
    # The first part is the room left on the edge. The second part is flow we are allowed to take
    # BACK: pushing flow from u to v along an edge v -> u cancels flow that v -> u was carrying.
    # An AUGMENTING PATH is a path from source to sink that has residual capacity above 0 on every
    # edge. Its BOTTLENECK is the smallest residual capacity along it. We push the bottleneck
    # amount along the path: every residual[u][v] on it goes DOWN by that much, and the opposite
    # residual[v][u] goes UP by that much (that increase is what lets us undo the push later).
    # When no augmenting path exists, the flow is maximum.
    # Why we can stop: the nodes still reachable from source form a set S. Every edge leaving S
    # is full and every edge entering S carries nothing (otherwise we could reach more nodes),
    # so the flow equals the capacity of the cut (S, T). That is the minimum cut.
    #
    # method is "bfs" (the default), "dfs", or "fattest". Each decides which augmenting path to
    # use when there are several. The final answer is the same for all three.
    # verbose=1 prints each augmenting path, its bottleneck, the flow so far, and the minimum cut.
    # verbose=2 also prints the residual graph before each search and the flow on each edge.
    # This graph is not changed: the residual graph is a copy.
    # Raises ValueError for a missing source or sink, a source equal to the sink, a negative
    # capacity, or a method that does not exist
    def max_flow(self, source, sink, method="bfs", verbose=0, counters=None):
        finders = {"dfs": find_path_dfs, "bfs": find_path_bfs, "fattest": find_path_fattest}
        if source not in self.nodes or sink not in self.nodes:
            raise ValueError("Illegal node entered")
        if source == sink:
            raise ValueError("The source and the sink must be different nodes")
        if method not in finders:
            raise ValueError("method must be 'dfs', 'bfs', or 'fattest'")
        for u, v, capacity in self.get_edges():
            if capacity < 0:
                raise ValueError("Capacities cannot be negative (edge %s -> %s)" % (u, v))
        find_path = finders[method]

        def say(message):
            if verbose >= 1:
                print(message)

        residual = self._residual_graph()
        value = 0
        augmentations = 0
        say("Maximum flow from %s to %s (method: %s)" % (source, sink, method))
        while True:
            if verbose >= 2:
                print_residual(residual)
            path = find_path(residual, source, sink, counters)
            if path is None:
                break
            bottleneck = min(residual[u][v] for u, v in zip(path, path[1:]))
            for u, v in zip(path, path[1:]):
                residual[u][v] -= bottleneck
                residual[v][u] += bottleneck
            value += bottleneck
            augmentations += 1
            if counters is not None:
                counters.augmenting_paths = getattr(counters, "augmenting_paths", 0) + 1
            # A step u -> v is "backward" if u -> v is not an edge of the graph itself:
            # it only exists in the residual graph, as a way to undo flow on v -> u
            backward = [(u, v) for u, v in zip(path, path[1:]) if v not in self.nodes[u]]
            note = ""
            if backward:
                note = "   backward: " + ", ".join("%s -> %s" % edge for edge in backward)
            say("Augmentation %d: %s   bottleneck %s   flow so far %s%s"
                % (augmentations, " -> ".join(str(node) for node in path), bottleneck, value, note))

        # No augmenting path is left. Whatever source can still reach is the s side of the min cut
        reachable = reachable_from(residual, source, counters)
        source_side = sorted_nodes(reachable)
        cut_edges = [(u, v, c) for u, v, c in self.get_edges() if u in reachable and v not in reachable]
        flow = {}
        for u in self.nodes:
            flow[u] = {}
            for v, capacity in self.nodes[u].items():
                # The flow on u -> v is the capacity minus what is left. If an edge v -> u also exists,
                # opposite flows cancel, so we report only the larger direction
                flow[u][v] = max(0, capacity - residual[u][v])
        say("No augmenting path is left. Nodes reachable from %s: %s" % (source, source_side))
        say("Minimum cut: S = %s, edges from S to T: %s, capacity %s"
            % (source_side, ", ".join("%s -> %s (%s)" % (u, v, c) for u, v, c in cut_edges) or "none",
               sum(c for u, v, c in cut_edges)))
        if verbose >= 2:
            print("Flow on each edge (flow / capacity):")
            for u, v, capacity in self.get_edges():
                print("  %s -> %s: %s / %s" % (u, v, flow[u][v], capacity))
        return FlowResult(value, flow, source_side, cut_edges, augmentations,
                          counters.relaxations if counters is not None else 0,
                          counters.pq_ops if counters is not None else 0,
                          counters.augmenting_paths if counters is not None else augmentations,
                          counters.residual_examinations if counters is not None else 0)

    # Returns a NEW dictionary of dictionaries with a residual capacity for every ordered pair
    # that is an edge in either direction. Before any flow, residual[u][v] is the capacity of
    # u -> v (0 if only v -> u is an edge). The order of the neighbors is the order the edges
    # were added, followed by any extra backward entries.
    def _residual_graph(self):
        residual = {node: {} for node in self.nodes}
        for u, v, capacity in self.get_edges():
            residual[u][v] = capacity
            residual[v].setdefault(u, 0)
        return residual

    def __str__(self):
        result = ""
        for node in self.nodes:
            result += "Node %s: " % node
            for edge in self.nodes[node]:
                result += "{Node %s, %s} " % (edge, self.nodes[node][edge])
            result += "\n"
        return result


# What max_flow gives back (see max_flow above for what each field means)
class FlowResult:

    def __init__(self, value, flow, source_side, cut_edges, augmentations,
                 relaxations=0, pq_ops=0, augmenting_paths=0, residual_examinations=0):
        self.value = value
        self.flow = flow
        self.source_side = source_side
        self.cut_edges = cut_edges
        self.augmentations = augmentations
        self.relaxations = relaxations
        self.pq_ops = pq_ops
        self.augmenting_paths = augmenting_paths
        self.residual_examinations = residual_examinations


# ----------------------------------------------------------------------
# Three ways to find an augmenting path in the residual graph.
# Each returns a list of nodes [source, ..., sink], or None if there is no path.
# They only use an edge u -> v if residual[u][v] is above 0.
# ----------------------------------------------------------------------

# Depth first search: follow the first edge with room, as deep as possible, and back up when stuck.
# This is the path search from the original Ford-Fulkerson method. The path it finds depends on
# the order the edges were added, and it can be long and twisty.
# It is written with a stack of iterators (instead of recursion) so it can handle long paths.
def find_path_dfs(residual, source, sink, counters=None):
    parent = {source: None}
    stack = [(source, iter(residual[source]))]
    while len(stack) > 0:
        node, neighbors = stack[-1]
        moved = False
        for neighbor in neighbors:
            if counters is not None:
                counters.residual_examinations = getattr(counters, "residual_examinations", 0) + 1
            if counters is not None:
                counters.relaxations += 1
            if neighbor not in parent and residual[node][neighbor] > 0:
                parent[neighbor] = node
                if neighbor == sink:
                    return build_path(parent, source, sink)
                stack.append((neighbor, iter(residual[neighbor])))
                moved = True
                break
        if not moved:
            stack.pop()
    return None


# Breadth first search: explore nodes in order of how many edges they are from the source.
# The first time we reach the sink, we have a path with the FEWEST edges. This is what makes it
# Edmonds-Karp, and it is what guarantees at most O(V * E) augmentations.
def find_path_bfs(residual, source, sink, counters=None):
    parent = {source: None}
    queue = deque([source])
    if counters is not None:
        counters.pq_ops += 1  # enqueue source
    while len(queue) > 0:
        node = queue.popleft()
        if counters is not None:
            counters.pq_ops += 1  # dequeue node
        for neighbor, room in residual[node].items():
            if counters is not None:
                counters.residual_examinations = getattr(counters, "residual_examinations", 0) + 1
            if counters is not None:
                counters.relaxations += 1
            if room > 0 and neighbor not in parent:
                parent[neighbor] = node
                if neighbor == sink:
                    return build_path(parent, source, sink)
                queue.append(neighbor)
                if counters is not None:
                    counters.pq_ops += 1  # enqueue discovered node
    return None


# Fattest path: the path whose bottleneck is as large as possible.
# This is Dijkstra's algorithm with a different meaning for "best": instead of the smallest
# total distance, we want the largest bottleneck so far, and the bottleneck of a route to a
# neighbor is min(bottleneck of the route to this node, room on the edge). The heap holds
# (negative bottleneck, tie counter, node) since heapq pops the smallest entry.
# Each augmentation recovers at least 1/E of the flow that is still left to push, so it needs
# about E * ln(f) augmentations in total.
def find_path_fattest(residual, source, sink, counters=None):
    best = {source: INF}
    parent = {source: None}
    heap = [(-INF, 0, source)]
    if counters is not None:
        counters.pq_ops += 1  # initial push
    counter = 1
    done = set()
    while len(heap) > 0:
        negative_width, _, node = heapq.heappop(heap)
        if counters is not None:
            counters.pq_ops += 1
        if node in done:
            continue
        done.add(node)
        if node == sink:
            return build_path(parent, source, sink)
        for neighbor, room in residual[node].items():
            if counters is not None:
                counters.residual_examinations = getattr(counters, "residual_examinations", 0) + 1
            if counters is not None:
                counters.relaxations += 1
            if room > 0 and neighbor not in done:
                width = min(-negative_width, room)
                if width > best.get(neighbor, 0):
                    best[neighbor] = width
                    parent[neighbor] = node
                    heapq.heappush(heap, (-width, counter, neighbor))
                    if counters is not None:
                        counters.pq_ops += 1
                    counter += 1
    return None


# The set of nodes that can be reached from source using edges with residual capacity above 0
def reachable_from(residual, source, counters=None):
    seen = {source}
    stack = [source]
    while len(stack) > 0:
        node = stack.pop()
        for neighbor, room in residual[node].items():
            if counters is not None:
                counters.residual_examinations = getattr(counters, "residual_examinations", 0) + 1
            if room > 0 and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


# Rebuilds a path from a parent dictionary (the same function that is in graph_sp.py)
# Starts at target and follows parents back until it reaches source
# Returns the path as a list [source, ..., target], or None if target cannot be reached from source
def build_path(parent, source, target):
    path = [target]
    node = target
    while node != source:
        node = parent[node]
        # None: no path. Longer than the graph: we are going around a cycle.
        if node is None or len(path) > len(parent):
            return None
        path.append(node)
    path.reverse()
    return path


# Sorts node names (they might be numbers or strings, a mix would not sort so we leave those alone)
def sorted_nodes(nodes):
    try:
        return sorted(nodes)
    except TypeError:
        return list(nodes)


# Prints the edges of the residual graph that still have room, one line per node:  node: neighbor(room) ...
def print_residual(residual):
    print("Residual graph (neighbor(room left)):")
    for node in residual:
        entries = ["%s(%s)" % (neighbor, room) for neighbor, room in residual[node].items() if room > 0]
        print("  %s: %s" % (node, " ".join(entries) if entries else "-"))


# Prints the flow on every edge of a graph after max_flow: u -> v: flow / capacity
def print_flow(graph, result):
    print("Maximum flow value: %s" % result.value)
    for u, v, capacity in graph.get_edges():
        print("  %s -> %s: %s / %s" % (u, v, result.flow[u][v], capacity))


# Makes a random graph with the given number of nodes (named 0 .. num_nodes - 1) and directed edges,
# with capacities between min_capacity and max_capacity (inclusive)
# We cannot add more than n(n-1) directed edges, so we stop at that
def make_graph(num_nodes, num_edges, min_capacity=1, max_capacity=10):
    graph = Graph()
    for i in range(num_nodes):
        graph.add_node(i)
    num_edges = min(num_edges, num_nodes * (num_nodes - 1))
    while graph.get_num_edges() < num_edges:
        graph.add_edge(random.randrange(num_nodes), random.randrange(num_nodes),
                       random.randint(min_capacity, max_capacity))
    return graph


# The example graph from class, with nodes s, a, b, c, d, t. The order the edges are added
# changes which path "dfs" and "bfs" find first, so there are two orders to try:
#   a_to_d_first=True:  a -> d is added before a -> c
#   a_to_d_first=False: a -> c is added before a -> d
# The maximum flow is 5 either way
def make_example_graph(a_to_d_first=True):
    graph = Graph()
    graph.add_edge("s", "a", 4)
    graph.add_edge("s", "b", 2)
    if a_to_d_first:
        graph.add_edge("a", "d", 4)
        graph.add_edge("a", "c", 2)
    else:
        graph.add_edge("a", "c", 2)
        graph.add_edge("a", "d", 4)
    graph.add_edge("a", "b", 1)
    graph.add_edge("b", "d", 2)
    graph.add_edge("c", "t", 3)
    graph.add_edge("d", "t", 3)
    return graph


def main():
    print("Welcome to our testing of maximum flow!")
    # print("This will randomly create a graph and add a number of specified edges to it")
    # num_nodes = int(input("Please enter the number of nodes you want in the directed graph: "))
    # num_edges = int(input("Please enter the number of edges you want in the directed graph: "))
    # test_graph = make_graph(num_nodes, num_edges)

    # The example from class: look for the path that uses a backward edge
    example = make_example_graph(a_to_d_first=True)
    print(example)
    print("Depth first search (Ford-Fulkerson). Watch for the backward edge:\n")
    example.max_flow("s", "t", method="dfs", verbose=1)

    # The same graph, with the edges typed in a different order, and a breadth first search
    print("\nBreadth first search (Edmonds-Karp), edges added in a different order:\n")
    make_example_graph(a_to_d_first=False).max_flow("s", "t", method="bfs", verbose=1)

    print("\nFattest path first:\n")
    example.max_flow("s", "t", method="fattest", verbose=1)

    # The numbers you can use for an experiment: how do the three methods compare on a bigger graph?
    # (random.seed makes the "random" graph the same every time we run this)
    random.seed(5)
    big = make_graph(2000, 12000, 1, 100)
    print("\nA random graph with 2000 nodes and 12000 edges, flow from node 0 to node 1999:")
    for method in ("dfs", "bfs", "fattest"):
        start = time.perf_counter()
        result = big.max_flow(0, 1999, method)
        print("  %-8s flow %d, %3d augmenting paths, %.2f seconds"
              % (method, result.value, result.augmentations, time.perf_counter() - start))

if __name__ == "__main__":
    main()
