import networkx as nx
import numpy as np
from ortools.linear_solver import pywraplp

def solve_tsp_branch_and_cut(graph):
    """Solves the TSP using branch and cut with subtour elimination constraints.

    Args:
        graph: A networkx graph representing the TSP instance.

    Returns:
        A list of nodes representing the optimal tour, or None if no solution is found.
    """

    num_nodes = len(graph.nodes)
    nodes = list(graph.nodes)

    solver = pywraplp.Solver.CreateSolver('SCIP')  # Use SCIP, which is good for MIPs

    # Decision variables: x[i, j] = 1 if edge (i, j) is in the tour, 0 otherwise
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = solver.IntVar(0, 1, f'x[{i},{j}]')

    # Objective function: minimize the total tour length
    objective = solver.Objective()
    for i in nodes:
        for j in nodes:
            if i != j:
                objective.SetCoefficient(x[i, j], graph[i][j]['weight'])
    objective.SetMinimization()

    # Degree constraints: each node must have exactly two incident edges
    for i in nodes:
        solver.Add(solver.Sum([x[i, j] for j in nodes if i != j]) == 1)
        solver.Add(solver.Sum([x[j, i] for j in nodes if i != j]) == 1)

    def get_solution_edges(solution):
        """Extracts the edges from a solver solution."""
        edges = []
        for i in nodes:
            for j in nodes:
                if i != j and solution.Value(x[i, j]) > 0.5:
                    edges.append((i, j))
        return edges

    def find_subtours(edges):
        """Finds subtours in a list of edges."""
        graph_sol = nx.DiGraph()
        graph_sol.add_edges_from(edges)
        components = list(nx.strongly_connected_components(graph_sol))
        if len(components) > 1:
            return components
        else:
            return None

    def add_subtour_constraints(solver, subtour):
        """Adds subtour elimination constraints to the solver."""
        solver.Add(solver.Sum([x[i, j] for i in subtour for j in subtour if i != j]) <= len(subtour) - 1)

    while True:
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            edges = get_solution_edges(solver)
            subtours = find_subtours(edges)

            if subtours is None:
                # Solution is a valid tour
                tour = []
                current_node = nodes[0]
                tour.append(current_node)
                remaining_edges = edges.copy()

                while len(tour) < num_nodes:
                    for u, v in remaining_edges:
                        if u == current_node:
                            tour.append(v)
                            current_node = v
                            remaining_edges.remove((u,v))
                            break
                return tour

            else:
                # Add subtour elimination constraints
                for subtour in subtours:
                    add_subtour_constraints(solver, subtour)
        else:
            return None  # No solution found

# Example usage:
graph = nx.complete_graph(5)  # Create a complete graph with 5 nodes
for u, v in graph.edges:
    graph[u][v]['weight'] = np.random.randint(1, 10)  # Assign random weights

tour = solve_tsp_branch_and_cut(graph)

if tour:
    print("Optimal tour:", tour)
    total_cost = 0;
    for i in range(len(tour)):
      total_cost += graph[tour[i]][tour[(i+1)%len(tour)]]['weight']
    print(f"Total Cost: {total_cost}")
else:
    print("No solution found.")