import os
import sys
from typing import List, Tuple, Set, Dict

import mgraph
import boxing

current_dir = os.path.dirname(os.path.realpath(__file__))
gtraverse_dir = os.path.join(current_dir, '..', 'gtraverse')
data_dir = os.path.join(current_dir, '..', 'data')
uff_dir = os.path.join(current_dir, '..', 'uff')

sys.path.append(gtraverse_dir)
sys.path.append(data_dir)
sys.path.append(current_dir)
sys.path.append(uff_dir)

import gtraverse
import atomic_data
import aromatic
import hyperconjugated
import uff


def get_bond_order(dist: float, atomic_no_a: int, atomic_no_b: int) -> float:
    tol: float = 0.05 # parameter

    hash_value: float
    if (atomic_no_a < atomic_no_b):
        hash_value = atomic_no_a + 53 * atomic_no_b
    else:
        hash_value = atomic_no_b + 53 * atomic_no_a
    
    pairBO = [[0.0, 0.0] for _ in range(4)] # pair bond order information
    match hash_value:
        case 54:
            # 1 1 
            pairBO[0][0] = 0.741
            pairBO[0][1] = 0.741
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0

        case 319:
            # 1 6
            pairBO[0][0] = 0.931
            pairBO[0][1] = 1.14
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0

        case 372:
            # 1 7
            pairBO[0][0] = 0.836
            pairBO[0][1] = 1.160
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 425:
            # 1 8
            pairBO[0][0] = 0.819
            pairBO[0][1] = 1.033
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 478:
            # 1 9
            pairBO[0][0] = 0.917
            pairBO[0][1] = 1.014
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 849:
            # 1 16
            pairBO[0][0] = 0.9
            pairBO[0][1] = 1.4
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 902:
            # 1 17
            pairBO[0][0] = 1.275
            pairBO[0][1] = 1.321
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 324:
            # 6 6 
            pairBO[0][0] = 1.370
            pairBO[0][1] = 1.682
            pairBO[1][0] = 1.370
            pairBO[1][1] = 1.432
            pairBO[2][0] = 1.243
            pairBO[2][1] = 1.382
            pairBO[3][0] = 1.187
            pairBO[3][1] = 1.268
        
        case 377:
            # 6 7
            pairBO[0][0] = 1.347
            pairBO[0][1] = 1.613
            pairBO[1][0] = 1.328
            pairBO[1][1] = 1.350
            pairBO[2][0] = 1.207
            pairBO[2][1] = 1.338
            pairBO[3][0] = 1.14
            pairBO[3][1] = 1.177
        
        case 430:
            # 6 8
            pairBO[0][0] = 1.273
            pairBO[0][1] = 1.448
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.135
            pairBO[2][1] = 1.272
            pairBO[3][0] = 1.115
            pairBO[3][1] = 1.145
        
        case 483:
            # 6 9
            pairBO[0][0] = 1.262
            pairBO[0][1] = 1.401
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0

        case 801:
            # 6 15
            pairBO[0][0] = 1.858
            pairBO[0][1] = 1.858
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.673
            pairBO[2][1] = 1.673
            pairBO[3][0] = 1.542
            pairBO[3][1] = 1.562
        
        case 854:
            # 6 16
            pairBO[0][0] = 1.714
            pairBO[0][1] = 1.920
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.553
            pairBO[2][1] = 1.647
            pairBO[3][0] = 1.478
            pairBO[3][1] = 1.535

        case 907:
            # 6 17
            pairBO[0][0] = 1.612
            pairBO[0][1] = 1.813
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 1861:
            # 6 35
            pairBO[0][0] = 1.789
            pairBO[0][1] = 1.950
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 2815:
            # 6 53
            pairBO[0][0] = 1.992
            pairBO[0][1] = 2.157
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 378:
            # 7 7 
            pairBO[0][0] = 1.181
            pairBO[0][1] = 1.864
            pairBO[1][0] = 1.332
            pairBO[1][1] = 1.332
            pairBO[2][0] = 1.139
            pairBO[2][1] = 1.252
            pairBO[3][0] = 1.098
            pairBO[3][1] = 1.133
        
        case 431:
            # 7 8 
            pairBO[0][0] = 1.184
            pairBO[0][1] = 1.507
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.066
            pairBO[2][1] = 1.258
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 484:
            # 7 9
            pairBO[0][0] = 1.317
            pairBO[0][1] = 1.512
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 855:
            # 7 16
            pairBO[0][0] = 1.440
            pairBO[0][1] = 1.719
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 1.448
            pairBO[3][1] = 1.448
            
        case 432:
            # 8 8 
            pairBO[0][0] = 1.116
            pairBO[0][1] = 1.516
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.2
            pairBO[2][1] = 1.208
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 485:
            # 8 9 
            pairBO[0][0] = 1.421
            pairBO[0][1] = 1.421
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 856:
            # 8 16
            pairBO[0][0] = 1.574
            pairBO[0][1] = 1.662
            pairBO[1][0] = 1.405
            pairBO[1][1] = 1.5
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0

        case 909:
            # 8 17
            pairBO[0][0] = 1.641
            pairBO[0][1] = 1.704
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.404
            pairBO[2][1] = 1.414
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 486:
            # 9 9 
            pairBO[0][0] = 1.322
            pairBO[0][1] = 1.412
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 864:
            # 16 16 
            pairBO[0][0] = 1.89
            pairBO[0][1] = 2.5
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 1.825
            pairBO[2][1] = 1.898
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0
        
        case 918:
            # 17 17
            pairBO[0][0] = 1.9879
            pairBO[0][1] = 1.9879
            pairBO[1][0] = 0.0
            pairBO[1][1] = 0.0
            pairBO[2][0] = 0.0
            pairBO[2][1] = 0.0
            pairBO[3][0] = 0.0
            pairBO[3][1] = 0.0

        case _:
            raise SystemExit(f"No bonding information between atomic numbers {atomic_no_a} and {atomic_no_b}. Please add them in.")

    bond_orders = [-1, -1, -1, -1]
    bond_orders[0] = dist <= pairBO[0][1] + tol
    bond_orders[1] = (dist >= (pairBO[1][0] - tol)) and (dist <= (pairBO[1][1] + tol))
    bond_orders[2] = (dist >= (pairBO[2][0] - tol)) and (dist <= (pairBO[2][1] + tol))
    bond_orders[3] = (dist >= (pairBO[3][0] - tol)) and (dist <= (pairBO[3][1] + tol))

    bo_array_sum: int = sum(bond_orders)

    if bo_array_sum == 0:
        return 0.0
    else:
        largest_index: int = -1

        for ibo in range(0, 4):
            if bond_orders[ibo]:
                largest_index = ibo
        
        match largest_index:
            case 0:
                return 1.0
            case 1:
                return 1.5
            case 2:
                return 2.0
            case 3:
                return 3.0
            case _:
                raise SystemExit(f"Error w bond determination, largest_index {largest_index}")


def set_edges(graph: mgraph.mgraph, box_array: boxing.mbox_array) -> None:

    # print(f"nx: {box_array.m_nx}, ny: {box_array.m_ny}, nz: {box_array.m_nz}")
    for ibox in range(0, box_array.m_nboxes):
        box_natoms: int = box_array.m_box_array[ibox].m_natoms
        box_nodes: List[int] = box_array.m_box_array[ibox].m_nodes
        neighbouring_boxes: List[int] = box_array.get_neighbours(ibox)

        for iatom in range(0, box_natoms):
            iatomic_no: int = graph.m_elements[box_nodes[iatom]]

            stop_at_idx: int = 0
            for katom in range(0, box_natoms):
                if (box_nodes[katom] < box_nodes[iatom]):
                    stop_at_idx = katom + 1
                if (box_nodes[katom] >= box_nodes[iatom]):
                    stop_at_idx = katom
                    break
            
            for katom in range(0, stop_at_idx):
                katomic_no: int = graph.m_elements[box_nodes[katom]]
                r: float = graph.distance(box_nodes[katom], box_nodes[iatom])
                bond_order: float = get_bond_order(r, iatomic_no, katomic_no)

                if bond_order > 0.0:
                    bond = tuple((box_nodes[katom], box_nodes[iatom]))
                    graph.m_bond_orders.append(bond_order)
                    graph.m_edges.append(bond)
                    graph.add_edge_adjacency(bond)

                    graph.m_degree[box_nodes[katom]] += 1
                    graph.m_degree[box_nodes[iatom]] += 1

                    graph.m_bonding_electrons[box_nodes[katom]] += 2 * bond_order
                    graph.m_bonding_electrons[box_nodes[iatom]] += 2 * bond_order

            for neigh_box in neighbouring_boxes:
                neigh_box_natoms: int = box_array.m_box_array[neigh_box].m_natoms
                neigh_box_nodes: List[int] = box_array.m_box_array[neigh_box].m_nodes
                stop_at_idx: int = 0
                for katom in range(0, neigh_box_natoms):
                    if (neigh_box_nodes[katom] < box_nodes[iatom]):
                        stop_at_idx = katom + 1
                    if (neigh_box_nodes[katom] >= box_nodes[iatom]):
                        stop_at_idx = katom
                        break
                
                for katom in range(0, stop_at_idx):
                    katomic_no: int = graph.m_elements[neigh_box_nodes[katom]]
                    r: float = graph.distance(box_nodes[iatom], neigh_box_nodes[katom])
                    bond_order: float = get_bond_order(r, iatomic_no, katomic_no)

                    if bond_order > 0.0:
                        bond = tuple((neigh_box_nodes[katom], box_nodes[iatom]))
                        graph.m_bond_orders.append(bond_order)
                        graph.m_edges.append(bond)
                        graph.add_edge_adjacency(bond)

                        graph.m_degree[neigh_box_nodes[katom]] += 1
                        graph.m_degree[box_nodes[iatom]] += 1

                        graph.m_bonding_electrons[neigh_box_nodes[katom]] += 2 * bond_order
                        graph.m_bonding_electrons[box_nodes[iatom]] += 2 * bond_order

    graph.set_edge_count()

def define_conjugate_regions(graph: mgraph.mgraph, boxarray: boxing.mbox_array) -> None:
    unsaturated_nodes: Set[int] = set()
    edges = graph.m_edges
    
    for edge in edges:
        node1: int = edge[0]
        node2: int = edge[1]

        coordination_number_1 = graph.m_coordination_number[node1]
        coordination_number_2 = graph.m_coordination_number[node2]

        if(coordination_number_1 > 1 and coordination_number_1 < 4 and coordination_number_2 > 1 and coordination_number_2 < 4):
            unsaturated_nodes.add(node1)
            unsaturated_nodes.add(node2)

    colors: List[int] = [0 for _ in range(graph.m_natoms)]

    conjsys_count: List[int] = [1]
    for unsat_node in unsaturated_nodes:
        
        if colors[unsat_node] == 0:
            gtraverse.bfs_conjugated(graph, unsat_node, conjsys_count, colors, boxarray)
        
        valence: int = atomic_data.get_valence(graph.m_elements[unsat_node])
        sigma_bonds_number: int = graph.m_degree[unsat_node]
        coordination_number: int = graph.m_coordination_number[unsat_node]
        charge: int = graph.m_charges[unsat_node]

        pi_electrons: int = valence + sigma_bonds_number - 2 * coordination_number - charge
        graph.m_pi_electrons[unsat_node] = pi_electrons


def define_cycles(graph: mgraph.mgraph, boxarray: boxing.mbox_array) -> None:
    subgraph_count: int = len(graph.m_subgraphs)

    for isubgraph in range(0, subgraph_count):
        offset: int = graph.m_subgraphs[isubgraph].m_offset
        natoms: int = graph.m_subgraphs[isubgraph].m_natoms

        ortho_edges: List[Tuple[int, int]] = []
        mst_edges: List[Tuple[int, int]] = [] # minimum spanning edges
        gtraverse.bfs_tree(graph, ortho_edges, mst_edges, offset, natoms)
        subgraph_mirror: mgraph.mgraph = mgraph.mgraph()
        subgraph_mirror.set_mirror(natoms * 2)

        nnodes: int = natoms
        for ortho_edge in ortho_edges:
            subgraph_mirror.add_edge_only_adjacency(ortho_edge[0] - offset + nnodes, ortho_edge[1] - offset)
            subgraph_mirror.add_edge_only_adjacency(ortho_edge[0] - offset, nnodes + ortho_edge[1] - offset)

        for mst_edge in mst_edges:
            subgraph_mirror.add_edge_only_adjacency(mst_edge[0] - offset, mst_edge[1] - offset)
            subgraph_mirror.add_edge_only_adjacency(mst_edge[0] - offset + nnodes, mst_edge[1] + nnodes - offset)
        
        shortest_paths: List[List[int]] = [[] for _ in range(0, len(ortho_edges))]
        valid_cycles: List[int] = []

        for iortho_edge, ortho_edge in enumerate(ortho_edges):
            mirror_node: int = ortho_edge[0] + nnodes
            shortest_paths[iortho_edge] = gtraverse.shortest_path_bfs(subgraph_mirror, ortho_edge[0], mirror_node, offset)
            if len(shortest_paths[iortho_edge]) <= 11:
                valid_cycles.append(iortho_edge)
        
        graph.m_subgraphs[isubgraph].setup_cycles(len(valid_cycles))
        for icycle, cycle_index in enumerate(valid_cycles):
            gtraverse.min_cycle(graph, shortest_paths[cycle_index], nnodes, icycle, boxarray, isubgraph)

def define_aromatic(graph: mgraph.mgraph) -> None:
    subgraph_count: int = len(graph.m_subgraphs)

    for isubgraph in range(0, subgraph_count):
        edge_cycle_idx_map: Dict[Tuple[int, int], Set[int]] = {}
        aromatic_cycles: Set[int] = set()
        aromatic.small_aromatic_rings(graph, edge_cycle_idx_map, aromatic_cycles, isubgraph)
        # aromatic.larger_aromatic(graph, edge_cycle_idx_map, aromatic_cycles, isubgraph)

def hv_edge(graph: mgraph.mgraph, node1: int, node2: int) -> int:
    coordination_number1: int = graph.m_coordination_number[node1] - 1
    coordination_number2: int = graph.m_coordination_number[node2] - 1

    valence1: int = atomic_data.get_valence(graph.m_elements[node1]) - 1
    valence2: int = atomic_data.get_valence(graph.m_elements[node2]) - 1

    r: float = graph.distance(node1, node2)
    bond_order: float = get_bond_order(r, graph.m_elements[node1], graph.m_elements[node2])
    bond_order_counter: int = atomic_data.bo_counter_map[bond_order]

    hash_value: int
    if valence1 < valence2:
        hash_value = valence1 + 7 * valence2 + 49 * bond_order_counter + 147 * coordination_number1 + 441 * coordination_number2
        # hash function = valence1 + m_v * valence2 + m_v^2 * bond_order_counter + m_v^2 * m_boc * coordination_number1 + m_v^2 * m_boc * m_hs * coordination_number2
        # m_v = 7, max of valence counter (8-1)
        # m_boc = 3, max of bond order counter
        # m_hs = 3, max of coordination numbercounter
    else:
        hash_value = valence2 + 7 * valence1 + 49 * bond_order_counter + 147 * coordination_number2 + 441 * coordination_number1
    return int(hash_value)

def hyperconjugated_donors_acceptors(graph: mgraph.mgraph, boxarray: boxing.mbox_array) -> None:
    box_array: List[boxing.mbox] = boxarray.get_boxes()

    # conjugated regions
    conj_system_count: int = len(graph.m_conjugated_systems)

    conj_edges_hash_value: Set[int] = set()
    hyperconj_classifications: List[int] = []
    visited_conj_carbon_nodes: List[int] = [0 for _ in range(graph.m_natoms)]
    conj_system_min_unsat_degree: List[int] = []
    conj_system_min_unsat_degree_no: List[int] = []

    for iconj_sys in range(0, conj_system_count):
        conj_nodes: List[int] = graph.m_conjugated_systems[iconj_sys].m_nodes
        edges_hash_values: List[int] = graph.m_conjugated_systems[iconj_sys].m_edges_hash_values
        conj_nodes_unsat_degree: List[int] = []

        min_unsaturated_degree: int = 4
        min_unsaturated_degree_no: int = 0
        for node in conj_nodes:
            visited_conj_carbon_nodes[node] = 1
            node_unsaturated_degree = graph.get_unsaturated_degree(node)
            conj_nodes_unsat_degree.append(node_unsaturated_degree)

            if (node_unsaturated_degree == min_unsaturated_degree):
                min_unsaturated_degree_no += 1
            elif (node_unsaturated_degree < min_unsaturated_degree):
                min_unsaturated_degree = node_unsaturated_degree
                min_unsaturated_degree_no = 1
        conj_system_min_unsat_degree.append(min_unsaturated_degree)
        conj_system_min_unsat_degree_no.append(min_unsaturated_degree_no)

        for hash_value in edges_hash_values:
            conj_edges_hash_value.add(hash_value)
        hyperconj_classifications.append(6)

    # unconjugated:
    # nodes:
    hyperconj_nodes: List[int] = []
    hyperconj_nodes_cases: List[int] = []
    for iatom in range(0, graph.m_natoms):
        if visited_conj_carbon_nodes[iatom] == 0:
            coordination_number: int = graph.m_coordination_number[iatom] - 1
            charge: int = graph.m_charges[iatom] + 1
            atomic_number: int = graph.m_elements[iatom] - 1
            hash_value: int = int(atomic_number) + 53 * int(charge) + 19663 * int(coordination_number)
            # hash function = a + m_a * c + m_a^2 * m_c * hs
            # m_a = 53, m_c = 7

            match hash_value:
                case 39331: 
                    # C-1(sp2)
                    hyperconj_classifications.append(4)
                    hyperconj_nodes.append(iatom)
                    hyperconj_nodes_cases.append(hash_value)
                case 39437:
                    # C+1(sp2)
                    hyperconj_classifications.append(5)
                    hyperconj_nodes.append(iatom)
                    hyperconj_nodes_cases.append(hash_value)
                case 59048 | 59049:
                    # N(sp3) and O(sp3)
                    hyperconj_classifications.append(4)
                    hyperconj_nodes.append(iatom)
                    hyperconj_nodes_cases.append(hash_value)
    
    # edges:
    edge_hash_idx_map: Dict[int, int] = {}
    hyperconj_iedges: List[int] = []
    hyperconj_edges_cases: List[int] = []
    for iedge, edge in enumerate(graph.m_edges):
        node1: int = edge[0]
        node2: int = edge[1]

        hash_value: int = hv_edge(graph, node1, node2)
        if hash_value == 1344 or hash_value == 1809:
            edge_hash_value: int = int(node1 + graph.m_natoms * node2)
            edge_hash_idx_map[edge_hash_value] = iedge
    
    for iedge, edge in enumerate(graph.m_edges):
        node1: int = edge[0]
        node2: int = edge[1]

        edge_hash_value: int = int(node1 + graph.m_natoms * node2)
        if edge_hash_value not in conj_edges_hash_value: # edge not part of conj system
            hash_value: int = hv_edge(graph, node1, node2)

            match hash_value:
                case 1298 | 759:
                    # C=C and C≡C bond
                    hyperconj_iedges.append(iedge)
                    hyperconj_edges_cases.append(hash_value)
                    hyperconj_classifications.append(6)

                case 1312:
                    # C=O bond
                    hyperconj_iedges.append(iedge)
                    hyperconj_edges_cases.append(hash_value)
                    hyperconj_classifications.append(5)
                
                case 1344 | 1809:
                    # C-H bond and C-X (X = F, Cl, Br, I)
                    carbon_node: int
                    valence1: int = atomic_data.get_valence(graph.m_elements[node1])
                    if valence1 == 4:
                        carbon_node = node1
                    else:
                        carbon_node = node2
                    
                    if visited_conj_carbon_nodes[carbon_node] == 0:
                        neighbours = graph.get_neighbours(carbon_node)
                        max_atomic_no: int = 0
                        min_atomic_no: int = 53
                        argmax_atomic_no: int = -1
                        argmin_atomic_no: int = -1
                        # print(f"neighbours: {neighbours}")
                        for neighbour in sorted(neighbours):
                            neighbour_atomic_no: int = graph.m_elements[neighbour]
                            if neighbour_atomic_no < min_atomic_no:
                                min_atomic_no = neighbour_atomic_no
                                argmin_atomic_no = neighbour
                            
                            if neighbour_atomic_no > max_atomic_no:
                                max_atomic_no = neighbour_atomic_no
                                argmax_atomic_no = neighbour

                        if max_atomic_no in {9, 17, 35, 53} and min_atomic_no > 1:
                            # halogen
                            # only one acceptor: C-X
                            hv: int = hv_edge(graph, carbon_node, argmax_atomic_no)
                            edge_hv: int

                            if carbon_node < argmax_atomic_no:
                                edge_hv = carbon_node + graph.m_natoms * argmax_atomic_no
                            else:
                                edge_hv = argmax_atomic_no + graph.m_natoms * carbon_node
                            
                            edge_idx: int = edge_hash_idx_map[edge_hv]
                            hyperconj_iedges.append(edge_idx)
                            hyperconj_classifications.append(1)
                            hyperconj_edges_cases.append(hv)

                            
                        elif max_atomic_no not in {9, 17, 35, 53} and min_atomic_no == 1:
                            # C-X absent, only C-H present (both acceptor and donor)
                            hv: int = hv_edge(graph, carbon_node, argmin_atomic_no)
                            edge_hv: int

                            if carbon_node < argmin_atomic_no:
                                edge_hv = carbon_node + graph.m_natoms * argmin_atomic_no
                            else:
                                edge_hv = argmin_atomic_no + graph.m_natoms * carbon_node

                            edge_idx: int = edge_hash_idx_map[edge_hv]
                            hyperconj_iedges.append(edge_idx)
                            hyperconj_classifications.append(2) # sigma donor and acceptor
                            hyperconj_edges_cases.append(hv)
                        
                        elif max_atomic_no in {9, 17, 35, 53} and min_atomic_no == 1:
                            # C-X present (acceptor), C-H present (donor)
                            hv: int = hv_edge(graph, carbon_node, argmax_atomic_no)
                            edge_hv: int
                            if carbon_node < argmax_atomic_no:
                                edge_hv = carbon_node + graph.m_natoms * argmax_atomic_no
                            else:
                                edge_hv = argmax_atomic_no + graph.m_natoms * carbon_node
                            edge_idx: int = edge_hash_idx_map[edge_hv]
                            hyperconj_iedges.append(edge_idx)
                            hyperconj_classifications.append(1) # sigma acceptor
                            hyperconj_edges_cases.append(hv)

                            hv: int = hv_edge(graph, carbon_node, argmax_atomic_no)
                            edge_hv: int
                            if carbon_node < argmin_atomic_no:
                                edge_hv = carbon_node + graph.m_natoms * argmin_atomic_no
                            else:
                                edge_hv = argmin_atomic_no + graph.m_natoms * carbon_node
                            edge_idx: int = edge_hash_idx_map[edge_hv]
                            hyperconj_iedges.append(edge_idx)
                            hyperconj_classifications.append(0) #sigma acceptor and donor
                            hyperconj_edges_cases.append(hv)
                        visited_conj_carbon_nodes[carbon_node] = 1

    # defining donors and acceptors
    da_total: int = len(hyperconj_classifications)
    graph.setup_hyper_donor_acceptors(da_total)

    # conjugated regions 
    
    for iconj_sys in range(0, conj_system_count):
        conj_system_natoms: int = graph.m_conjugated_systems[iconj_sys].m_natoms
        graph.m_da_array[iconj_sys].set_natoms(conj_system_natoms)
        graph.m_da_array[iconj_sys].set_classification(6)

        for iconj_node in range(0, conj_system_natoms):
            conj_node: int = graph.m_conjugated_systems[iconj_sys].m_nodes[iconj_node]
            graph.m_da_array[iconj_sys].m_nodes[iconj_node] = conj_node
            graph.m_da_array[iconj_sys].m_node_electrons[iconj_node] = graph.m_pi_electrons[conj_node]

            if graph.get_unsaturated_degree(conj_node) == conj_system_min_unsat_degree[iconj_sys]:
                graph.m_da_array[iconj_sys].m_terminal_nodes.append(conj_node)
            
            box_id: int = graph.get_boxid(conj_node)
            box_array[box_id].m_hyper_donor_acceptors.add(iconj_sys)
    
    hyperconj_idx: int = conj_system_count

    # nodes
    # print(f"hyperconj_nodes: {(hyperconj_nodes)}")
    for ihyperconj_node, hyperconj_node in enumerate(hyperconj_nodes):
        graph.m_da_array[hyperconj_idx].set_natoms(1)
        hash_value: int = hyperconj_nodes_cases[ihyperconj_node]

        match hash_value:
            case 39331:
                # C-1(sp2)
                graph.m_da_array[hyperconj_idx].m_nodes[0] = hyperconj_node
                graph.m_da_array[hyperconj_idx].m_node_electrons[0] = 2
                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(hyperconj_node)
                graph.m_da_array[hyperconj_idx].set_classification(hyperconj_classifications[hyperconj_idx]) # pi donor
                box_id: int = graph.get_boxid(hyperconj_node)
                box_array[box_id].m_hyper_donor_acceptors.add(hyperconj_idx)
            case 39437:
                # C+1 (sp2)
                graph.m_da_array[hyperconj_idx].m_nodes[0] = hyperconj_node
                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(hyperconj_node)
                graph.m_da_array[hyperconj_idx].set_classification(hyperconj_classifications[hyperconj_idx]) # pi acceptor
                box_id: int = graph.get_boxid(hyperconj_node)
                box_array[box_id].m_hyper_donor_acceptors.add(hyperconj_idx)
            case 59048 | 59049:
                # N(sp3) and O(sp3)
                graph.m_da_array[hyperconj_idx].m_nodes[0] = hyperconj_node
                graph.m_da_array[hyperconj_idx].m_node_electrons[0] = 2
                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(hyperconj_node)
                graph.m_da_array[hyperconj_idx].set_classification(hyperconj_classifications[hyperconj_idx]) # pi donor
                box_id: int = graph.get_boxid(hyperconj_node)
                box_array[box_id].m_hyper_donor_acceptors.add(hyperconj_idx)

        hyperconj_idx += 1

    # edges
    
    for ihyperconj_edge, hyperconj_edge in enumerate(hyperconj_iedges):
        graph.m_da_array[hyperconj_idx].set_natoms(2)
        bond = graph.m_edges[hyperconj_edge]
        hash_value: int = hyperconj_edges_cases[ihyperconj_edge]

        match hash_value:
            case 1298 | 759 | 1344 | 1809:
                # C=C bond, C≡C bond, C-H bond and C-X bond X = F, Cl, I, Br
                graph.m_da_array[hyperconj_idx].m_nodes[0] = bond[0]
                graph.m_da_array[hyperconj_idx].m_nodes[1] = bond[1]

                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(bond[0])
                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(bond[1])

                graph.m_da_array[hyperconj_idx].m_node_electrons[0] = 1
                graph.m_da_array[hyperconj_idx].m_node_electrons[1] = 1

                graph.m_da_array[hyperconj_idx].set_classification(hyperconj_classifications[hyperconj_idx])
                box_id1: int = graph.get_boxid(bond[0])
                box_id2: int = graph.get_boxid(bond[1])
                box_array[box_id1].m_hyper_donor_acceptors.add(hyperconj_idx)
                box_array[box_id2].m_hyper_donor_acceptors.add(hyperconj_idx)

            case 1312:
                # C=O bond
                graph.m_da_array[hyperconj_idx].m_nodes[0] = bond[0]
                graph.m_da_array[hyperconj_idx].m_nodes[1] = bond[1]

                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(bond[0])
                graph.m_da_array[hyperconj_idx].m_terminal_nodes.append(bond[1])

                graph.m_da_array[hyperconj_idx].set_classification(hyperconj_classifications[hyperconj_idx])
                box_id1: int = graph.get_boxid(bond[0])
                box_id2: int = graph.get_boxid(bond[1])
                box_array[box_id1].m_hyper_donor_acceptors.add(hyperconj_idx)
                box_array[box_id2].m_hyper_donor_acceptors.add(hyperconj_idx)

        hyperconj_idx += 1
    

def donor_acceptor_pair_shortest_path(donor_acceptor1: int, donor_acceptor2: int, 
                                      graph: mgraph.mgraph, shortest_path: List[int]) -> int:
    # returns shortest path
    hyper_classification1: int = graph.m_da_array[donor_acceptor1].m_classification
    hyper_classification2: int = graph.m_da_array[donor_acceptor2].m_classification
    daps: int = 0 # donor acceptor pi sigma
    # must have one donor and one acceptor
    # must have one of sigma nature and one of pi nature

    hash_value_classification: int
    if hyper_classification1 < hyper_classification2:
        hash_value_classification = hyper_classification1 + 6 * hyper_classification2
    else:
        hash_value_classification = hyper_classification2 + 6 * hyper_classification1

    match hash_value_classification:
        case 25 | 26 | 30 | 32 | 36 | 37:
            daps = 0
        case 38:
            # happens when we have d/a sigma and d/a pi
            # two possibilities: donor sigma - acceptor pi AND acceptor sigma - donor pi
            daps = 1
        case _:
            return 0

    # pick one set of terminal nodes to perform BFS on, do this based on distance criterion
    min_distance: float = 500
    node1: int
    node2: int
    for tnode1 in graph.m_da_array[donor_acceptor1].m_terminal_nodes:
        for tnode2 in graph.m_da_array[donor_acceptor2].m_terminal_nodes:
            r: float = graph.distance(tnode1, tnode2)
            if r < min_distance:
                min_distance = r
                node1 = tnode1
                node2 = tnode2
    
    gtraverse.shortest_path_hyperconjugated(node1, node2, graph, shortest_path)
    dist_edges: int = len(shortest_path) - 1

    if dist_edges <= 3 and dist_edges >= 1:
        return 1 + daps
    else:
        return 0

def determine_donor_acceptor(donor_acceptor1: int, donor_acceptor2: int, graph: mgraph.mgraph) -> Tuple[int, int]:
    donor: int
    acceptor: int
    
    classification1: int = graph.m_da_array[donor_acceptor1].m_classification
    classification2: int = graph.m_da_array[donor_acceptor2].m_classification

    donor_acceptor1_: int
    donor_acceptor2_: int

    hash_value: int

    if classification1 < classification2:
        hash_value = classification1 + 6 * classification2
        donor_acceptor1_ = donor_acceptor1
        donor_acceptor2_ = donor_acceptor2
    else:
        hash_value = classification2 + 6 * classification1
        donor_acceptor1_ = donor_acceptor2
        donor_acceptor2_ = donor_acceptor1

    match hash_value:
        case 25 | 26 | 37:
            donor = donor_acceptor2_
            acceptor = donor_acceptor1_
        case 30 | 32 | 36:
            donor = donor_acceptor1_
            acceptor = donor_acceptor2_

    return (donor, acceptor)


def pair_hyperconjugated_donors_acceptors(graph: mgraph.mgraph, boxarray: boxing.mbox_array) -> None:
    box_array: List[boxing.mbox] = boxarray.get_boxes()
    hsys_visited: Set[int] = set()

    for ibox in range(0, boxarray.m_nboxes):
        neighbour_boxes: List[int] = boxarray.get_neighbours(ibox)

        box_donor_acceptor_list: List[int] = sorted(list(box_array[ibox].m_hyper_donor_acceptors))

        for ida, donor_acceptor_i in enumerate(box_donor_acceptor_list):
            for jda in range(0, ida):
                # jda < ida
                # donor_acceptor_j < donor_acceptor_i
                donor_acceptor_j = box_donor_acceptor_list[jda]
                
                hash_value: int = int(donor_acceptor_j + graph.m_da_count * donor_acceptor_i)
                hv_status: int = 0

                if hash_value not in hsys_visited:
                    hv_status = 1
                    hsys_visited.add(hash_value)
                
                if hv_status == 1:
                    shortest_path: List[int] = []
                    status: int = donor_acceptor_pair_shortest_path(donor_acceptor_i, donor_acceptor_j, graph, shortest_path)

                    if status != 0:
                        hyperconj_system = hyperconjugated.hypersys()
                        hyperconj_system.m_separation = len(shortest_path) - 1

                        for node in shortest_path:
                            hyperconj_system.m_connection_path.append(node)
                        
                        donor: int
                        acceptor: int
                        hyperconj_sys_idx: int = len(graph.m_hyperconjugated_systems)

                        match status:
                            case 1:
                                donor, acceptor = determine_donor_acceptor(donor_acceptor_i, donor_acceptor_j, graph)
                                hyperconj_system.m_donor = donor
                                hyperconj_system.m_acceptor = acceptor
                                graph.m_hyperconjugated_systems.append(hyperconj_system)
                                box_array[ibox].m_hyper_systems.add(hyperconj_sys_idx)
                            
                            case 2:
                                hyperconj_system.m_donor = donor_acceptor_i
                                hyperconj_system.m_acceptor = donor_acceptor_j

                                hyperconj_system2 = hyperconjugated.hypersys(hyperconj_system)
                                hyperconj_system2.m_donor = donor_acceptor_j
                                hyperconj_system2.m_acceptor = donor_acceptor_i

                                graph.m_hyperconjugated_systems.append(hyperconj_system)
                                graph.m_hyperconjugated_systems.append(hyperconj_system2)

                                box_array[ibox].m_hyper_systems.add(hyperconj_sys_idx)
                                box_array[ibox].m_hyper_systems.add(hyperconj_sys_idx + 1)

            for neighbour_box in neighbour_boxes:
                box_donor_acceptor_list2: List[int] = sorted(list(box_array[neighbour_box].m_hyper_donor_acceptors))

                for kda, donor_acceptor_k in enumerate(box_donor_acceptor_list2):
                    if donor_acceptor_k < donor_acceptor_i:
                        hash_value: int = int(donor_acceptor_k + graph.m_da_count * donor_acceptor_i)
                        hv_status: int = 0

                        if hash_value not in hsys_visited:
                            hv_status = 1
                            hsys_visited.add(hash_value)
                        
                        if hv_status == 1:
                            shortest_path: List[int] = []
                            status: int = donor_acceptor_pair_shortest_path(donor_acceptor_i, donor_acceptor_k, graph, shortest_path)

                            if status != 0:
                                hyperconj_system = hyperconjugated.hypersys()
                                hyperconj_system.m_separation = len(shortest_path) - 1

                                for node in shortest_path:
                                    hyperconj_system.m_connection_path.append(node)

                                donor: int
                                acceptor: int
                                hyperconj_sys_idx: int = len(graph.m_hyperconjugated_systems)

                                match status:
                                    case 1:
                                        donor, acceptor = determine_donor_acceptor(donor_acceptor_i, donor_acceptor_k, graph)
                                        hyperconj_system.m_donor = donor
                                        hyperconj_system.m_acceptor = acceptor
                                        graph.m_hyperconjugated_systems.append(hyperconj_system)
                                        box_array[ibox].m_hyper_systems.add(hyperconj_sys_idx)
                                        box_array[neighbour_box].m_hyper_systems.add(hyperconj_sys_idx)
                                    case 2:
                                        hyperconj_system.m_donor = donor_acceptor_i
                                        hyperconj_system.m_acceptor = donor_acceptor_k

                                        hyperconj_system2 = hyperconjugated.hypersys(hyperconj_system)
                                        hyperconj_system2.m_donor = donor_acceptor_k
                                        hyperconj_system2.m_acceptor = donor_acceptor_i

                                        graph.m_hyperconjugated_systems.append(hyperconj_system)
                                        graph.m_hyperconjugated_systems.append(hyperconj_system2)

                                        box_array[ibox].m_hyper_systems.add(hyperconj_sys_idx)
                                        box_array[ibox].m_hyper_systems.add(hyperconj_sys_idx + 1)

                                        box_array[neighbour_box].m_hyper_systems.add(hyperconj_sys_idx)
                                        box_array[neighbour_box].m_hyper_systems.add(hyperconj_sys_idx + 1)

def distribute_hyper_conjsys_from_mgraph(graph: mgraph.mgraph) -> None:

    conj_system_count: int = len(graph.m_conjugated_systems)
    for iconj_sys in range(0, conj_system_count):
        conj_node: List[int] = graph.m_conjugated_systems[iconj_sys].m_nodes[0]

        subgraph_idx: int = graph.m_node_sg[conj_node] - 1
        graph.m_subgraphs[subgraph_idx].m_conjugated_systems.append(iconj_sys)
    
    hyperconj_system_count: int = len(graph.m_hyperconjugated_systems)
    for ihyperconj_sys in range(0, hyperconj_system_count):
        donor_idx: int = graph.m_hyperconjugated_systems[ihyperconj_sys].m_donor
        donor_node: int = graph.m_da_array[donor_idx].m_nodes[0]

        subgraph_idx: int = graph.m_node_sg[donor_node] - 1
        graph.m_subgraphs[subgraph_idx].m_hyperconjugated_systems.append(ihyperconj_sys)

def determine_feasible_edges(graph: mgraph.mgraph) -> None:
    subgraph_copy: mgraph.subgraph = mgraph.subgraph(graph.m_subgraphs[0])

    single_bonds: List[int] = graph.m_subgraphs[0].m_single_bonds
    for iedge in single_bonds:
        edge: Tuple[int, int] = graph.m_edges[iedge]

        node1: int = edge[0]
        node2: int = edge[1]

        node1_idx: int = graph.m_node_sg_nidx[node1]
        node2_idx: int = graph.m_node_sg_nidx[node2]

        node1_degree: int = graph.m_subgraphs[0].get_degree(node1_idx)
        node2_degree: int = graph.m_subgraphs[0].get_degree(node2_idx)

        if node1_degree == 1 or node2_degree == 1: 
            continue

        colors: List[int] = [0 for _ in range(graph.m_natoms)]
        subgraph_copy.delete_edge(node1_idx, node2_idx)

        size1: int = gtraverse.bfs_sg(subgraph_copy, node1_idx, colors, graph.m_target_frag_size)

        # if size1 != graph.m_subgraphs[0].m_natoms:
        size2: int = gtraverse.bfs_sg(subgraph_copy, node2_idx, colors, graph.m_target_frag_size)
        if (size1/graph.m_target_frag_size >= 0.6 and size2/graph.m_target_frag_size >= 0.6) and (graph.m_ring_status[node1] == 0 or graph.m_ring_status[node2] == 0):
            graph.m_subgraphs[0].m_feasible_edges.append(iedge)
        subgraph_copy.add_edge(node1_idx, node2_idx)


def characterise_graph(graph: mgraph.mgraph, boxarray: boxing.mbox_array) -> None:
    graph.check_atom_distance()
    set_edges(graph, boxarray)
    graph.determine_hybridisation()
    graph.check_hybridisation()
    graph.set_atom_types()
    gtraverse.identify_connected_components(graph)
    
    define_conjugate_regions(graph, boxarray)
    
    graph.check_bonds_adjacent_conj()
    graph.set_edges_sg()
    define_cycles(graph, boxarray)
    define_aromatic(graph)
    hyperconjugated_donors_acceptors(graph, boxarray)
    pair_hyperconjugated_donors_acceptors(graph, boxarray)
    distribute_hyper_conjsys_from_mgraph(graph)
    graph.determine_reference_volume()
    uff.calculate_evdw_sg(graph)
    determine_feasible_edges(graph)
    