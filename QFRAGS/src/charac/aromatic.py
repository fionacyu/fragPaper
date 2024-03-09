from typing import List, Tuple, Set, Dict

import rings
import boxing
import mgraph

def small_aromatic_rings(graph: mgraph.mgraph, edge_cycle_idx_map: Dict[Tuple[int, int], Set[int]],
                         aromatic_cycles: Set[int], subgraph_idx: int) -> None:
    
    cycle_count: int = graph.m_subgraphs[subgraph_idx].m_ncycles
    aromatic_status: int 

    for icycle in range(0, cycle_count):
        aromatic_status = 0
        cycle_nodes: Set[int] = set()

        cycle_edges: List[int] = graph.m_subgraphs[subgraph_idx].m_cycles[icycle].m_edges
        cycle_edge_count: int = graph.m_subgraphs[subgraph_idx].m_cycles[icycle].m_nedges

        for iedge in range(0, cycle_edge_count):
            cycle_nodes.add(cycle_edges[2 * iedge])
            cycle_nodes.add(cycle_edges[2 * iedge + 1])

        pi_elec_sum: int = 0
        coordination_number_sum: int = 0

        for node in cycle_nodes:
            coordination_number: int = graph.m_coordination_number[node]

            if coordination_number < 4:
                coordination_number_sum += 1

            pi_elec_sum += graph.m_pi_electrons[node]

        if (coordination_number_sum == len(cycle_nodes)): # all atoms are conjugated
            if (pi_elec_sum % 4 == 2): # Huckel's rule
                # now check for planarity

                if len(cycle_nodes) == 3: # planar by definition 
                    aromatic_cycles.add(icycle)
                
                else:
                    # grab any three nodes, we pick the first 3
                    cycle_nodes_list: List[int] = list(cycle_nodes)
                    u: int = cycle_nodes_list[0]
                    v: int = cycle_nodes_list[1]
                    w: int = cycle_nodes_list[2]

                    u_coords: List[float] = [graph.m_coordinates[3 * u], graph.m_coordinates[3 * u + 1], graph.m_coordinates[3 * u + 2]]
                    v_coords: List[float] = [graph.m_coordinates[3 * v], graph.m_coordinates[3 * v + 1], graph.m_coordinates[3 * v + 2]]
                    w_coords: List[float] = [graph.m_coordinates[3 * w], graph.m_coordinates[3 * w + 1], graph.m_coordinates[3 * w + 2]]

                    # uv - vector from u to v, uw - vector from u to w
                    uv: List[float] = [v_coords[0] - u_coords[0], v_coords[1] - u_coords[1], v_coords[2] - u_coords[2]]
                    uw: List[float] = [w_coords[0] - u_coords[0], w_coords[1] - u_coords[1], w_coords[2] - u_coords[2]]

                    # now determine a normal vector by taking the cross product between the two vectors
                    norm_vec: List[float] = [0, 0, 0]
                    norm_vec[0] = uv[1] * uw[2] - uv[2] * uw[1]
                    norm_vec[1] = -(uv[0] * uw[2] - uv[2] * uw[0])
                    norm_vec[2] = uv[0] * uw[1] - uv[1] * uw[0]

                    dev_status: int = 0 # deviation
                    for node in cycle_nodes_list[3:]:
                        # need to take a dot product between norm_vec and node - u 
                        node_coords: List[float] = [graph.m_coordinates[3 * node], graph.m_coordinates[3 * node + 1], graph.m_coordinates[3 * node + 2]]

                        uc_coords: List[float] = [node_coords[0] - u_coords[0], node_coords[1] - u_coords[1], node_coords[2] - u_coords[2]]

                        dot_product: float = (uc_coords[0] * norm_vec[0]) + (uc_coords[1] * norm_vec[1]) + (uc_coords[2] * norm_vec[2])
                        if abs(dot_product) < 0.05:
                            dev_status += 1
                    
                    if dev_status == len(cycle_nodes) - 3:
                        # it's aromatic
                        aromatic_cycles.add(icycle)
                        aromatic_status = 1

        if aromatic_status == 1:
            for iedge in range(0, cycle_edge_count):
                graph.set_aromatic_at(cycle_edges[2 * iedge])
                graph.set_aromatic_at(cycle_edges[2 * iedge + 1])

                bond: Tuple[int, int] = (cycle_edges[2 * iedge], cycle_edges[2 * iedge + 1])
                if bond in edge_cycle_idx_map:
                    edge_cycle_idx_map[bond].add(icycle)
                else:
                    edge_cycle_idx_map[bond] = {icycle}
        

def larger_aromatic(graph: mgraph.mgraph, edge_cycle_idx_map: Dict[Tuple[int, int], Set[int]],
                    aromatic_cycles: Set[int], subgraph_idx: int) -> None:
    cycle_count: int = graph.m_subgraphs[subgraph_idx].m_ncycles
    cycle_neighbours: List[Set[int]] = [set() for _ in range(cycle_count)]

    for _, cycle_set in edge_cycle_idx_map.items():
        # at most, an edge can belong to two cycles/rings
        if len(cycle_set == 2):
            cycle_set_list: List[int] = list(cycle_set)
            cycle1: int = cycle_set_list[0]
            cycle2: int = cycle_set_list[1]

            cycle_neighbours[cycle1].add(cycle2)
            cycle_neighbours[cycle2].add(cycle1)

    aromatic_systems_: List[Set[int]] = [set() for _ in range(len(aromatic_cycles))]
    visited_cycles: Set[int] = set()
    aromatic_system_count: int = 0

    for icycle in range(0, cycle_count):
        if (icycle in aromatic_cycles) and (icycle not in visited_cycles):
            aromatic_systems_[aromatic_system_count] = cycle_neighbours[icycle]
            aromatic_systems_[aromatic_system_count].add(icycle)
            visited_cycles.add(icycle)
            
            for neighbour_cycles in cycle_neighbours[icycle]:
                visited_cycles.add(neighbour_cycles)
            aromatic_system_count += 1


