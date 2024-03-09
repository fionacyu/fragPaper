import os
import sys
import queue
import math
from collections import Counter
from typing import List, Tuple, Set, Dict

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
gtraverse_dir = os.path.join(current_dir, '..', 'gtraverse')
data_dir = os.path.join(current_dir, '..', 'data')
uff_dir = os.path.join(current_dir, '..', 'uff')

sys.path.append(charac_dir)
sys.path.append(current_dir)
sys.path.append(gtraverse_dir)
sys.path.append(data_dir)

import mgraph
import gtraverse
import molecule
import atomic_data
import uff

def calculate_fragment_volume(self, subgraph_copy: mgraph.subgraph, seed_node: int, 
                              colors: List[int]) -> Tuple[float, int]:
    fragment_elements: List[int] = []

    elements: List[int] = self.m_graph.m_elements
    global_seed_node_idx: int = subgraph_copy.get_node(seed_node)
    seed_node_atomic_no: int = elements[global_seed_node_idx]

    fragment_elements.append(seed_node_atomic_no)
    fragment_size: int = 1

    bfs_queue: queue.Queue = queue.Queue()
    bfs_queue.put(seed_node)

    pair_volume: float = 0.0
    
    while not bfs_queue.empty():
        node_label: int = bfs_queue.get()
        neighbours: Set[int] = subgraph_copy.get_neighbours(node_label)
        global_node_idx: int = subgraph_copy.get_node(node_label)

        for neighbour in neighbours:
            global_neighbour_idx: int = subgraph_copy.get_node(neighbour)
            neighbour_atomic_no: int = elements[global_neighbour_idx]

            if colors[neighbour] == 0:
                colors[neighbour] = 1
                bfs_queue.put(neighbour)
                pair_volume += self.m_graph.pair_volume(global_node_idx, global_neighbour_idx)
                fragment_elements.append(neighbour_atomic_no)
                fragment_size += 1
            elif colors[neighbour] == 1: # account of edges in cycles/rings
                pair_volume += self.m_graph.pair_volume(global_node_idx, global_neighbour_idx)
        colors[node_label] = 2

    single_volume: float = 0.0
    element_counter: Dict[int, int] = Counter(fragment_elements)
    
    for element, count in element_counter.items():
        sigma: float = atomic_data.get_radii(element)
        sphere_volume: float = (4.0/3.0 * math.pi * sigma * sigma * sigma)
        single_volume += (sphere_volume * count)

    return (single_volume - pair_volume), fragment_size

def volume_sigmoid(volume_deviation: float) -> float:
    tol: float = 0.05
    x_ref: float = 0.5 # threshold is 50% error

    a: float = -1/(x_ref * x_ref) * math.log(-1 + 2/(2-tol)) # exponent
    return 2/(1 + math.exp(-a * (volume_deviation ) * (volume_deviation))) - 1

def volume_range_sigmoid(reference_volume: float, volume_range: float) -> float:
    tol: float = 0.05
    x_ref: float = -0.25 

    a: float = -4 * math.log(-1 + 1/(1-tol))
    x_value: float = volume_range/reference_volume - 1
    return 1/(1 + math.exp(-a * (x_value - x_ref)))

def vol_penalty(self, subgraph_copy: mgraph.subgraph) -> Tuple[float, float, float]:
    natoms: int = subgraph_copy.m_natoms
    reference_volume: float = self.m_graph.m_ref_vol_atom * self.m_target_size

    colors: List[int] = [0 for _ in range(natoms)]

    nfrags: int = 0
    fragment_volumes: List[float] = []
    fragment_sizes: List[int] = []
    for iatom in range(0, natoms):
        if colors[iatom] == 0:
            fragment_size: int
            fragment_volume: float
            fragment_volume, fragment_size = self.calculate_fragment_volume(subgraph_copy, iatom, colors)
            nfrags += 1
            fragment_volumes.append(fragment_volume)
            fragment_sizes.append(fragment_size)
    average_volume: float = sum(fragment_volumes)/nfrags
    volume_deviation: float = average_volume/reference_volume - 1
    volume_range: float = max(fragment_volumes) - min(fragment_volumes)

    # potential energy scaling factor
    min_fragment_volume: float = min(fragment_volumes)
    min_frag_size_idx: int = fragment_volumes.index(min_fragment_volume)
    min_frag_size: int = fragment_sizes[min_frag_size_idx]
    pe_sf: float = math.sqrt(nfrags) * min_frag_size / self.m_target_size
    return volume_sigmoid(volume_deviation), volume_range_sigmoid(reference_volume, volume_range), pe_sf

def hyper_conj_sigmoid(x: float, x_max: float):
    tol: float = 0.05
    a: float = -1/x_max * math.log(tol/(2-tol))
    numerator: float = (1 - math.exp(-1 * a * x))
    denominator: float = (1 + math.exp(-1 * a * x))

    return (numerator/denominator)

def conjugation_penalty(self, subgraph_copy: mgraph.subgraph, conjugated_systems: Set[int]) -> float:
    node_sg_nidx: List[int] = self.m_graph.m_node_sg_nidx
    natoms: int = self.m_subgraph.m_natoms
    no_cs_affected: int = 0 # no. of conj systems affected

    p_conj_total: float = 0.0

    for conj_system in conjugated_systems:
        pi_electrons: int = 0
        colors: List[int] = [0 for _ in range(natoms)]
        local_cs_score_sum: float = 0.0
        cs_natoms: int = self.m_graph.m_conjugated_systems[conj_system].m_natoms
        worst_cs: int = cs_natoms  - 1
        for node in self.m_graph.m_conjugated_systems[conj_system].m_nodes:
            local_node_idx: int = node_sg_nidx[node]
            pi_electrons += self.m_graph.m_pi_electrons[node]

            if colors[local_node_idx] == 0:
                pi_elec_subtotal: List[int] = [self.m_graph.m_pi_electrons[node]]
                connected_comp_size: List[int] = [1]

                gtraverse.bfs_conj_pi(self.m_graph, subgraph_copy, local_node_idx, pi_elec_subtotal, colors, connected_comp_size)
                local_cs_score: float = pi_elec_subtotal[0] / connected_comp_size[0]
                local_cs_score_sum += local_cs_score

        cs_score_after: float = local_cs_score_sum / cs_natoms
        cs_score: float = pi_electrons / (cs_natoms * cs_natoms)
        delta_conj: float = (cs_score_after - cs_score)/cs_score
        p_conj: float = hyper_conj_sigmoid(delta_conj, worst_cs)

        if (p_conj > 0.0):
            p_conj_total += p_conj
            no_cs_affected += 1
    
    if (no_cs_affected > 0):
        return p_conj_total/no_cs_affected
    else:
        return 0.0

def hyperconjugation_penalty(self, subgraph_copy: mgraph.subgraph, hyperconjugated_systems: Set[int]) -> float:
    p_hyper_total: float = 0.0
    no_hs_affected: int = 0

    for hyperconj_system in hyperconjugated_systems:
        unweighted_penalty: float
        worst_score: float
        unweighted_penalty, worst_score = gtraverse.delta_hyperconjugation(self.m_graph, subgraph_copy, hyperconj_system)
        p_hyper: float = (1.0 / self.m_graph.m_hyperconjugated_systems[hyperconj_system].m_separation) * hyper_conj_sigmoid(unweighted_penalty, worst_score)
        p_hyper_total += p_hyper
        if p_hyper > 0.0:
            no_hs_affected += 1
    return (p_hyper_total / no_hs_affected) if no_hs_affected > 0 else 0.0

def potential_energy_sigmoid(x: float, pe_sf: float) -> float:
    tol: float = 0.05
    xref: float = 10.0
    xmid: float = 25.0
    a: float = -1/pe_sf * math.log((1/tol) - 1) / (xref - xmid)

    try:
        pos = 1/(1 + math.exp(-a/pe_sf * (x - pe_sf * xmid)))
    except OverflowError:
        pos = 0.0
    
    try:
        neg = 1/(1 + math.exp(-a/pe_sf * (-1 * x - pe_sf * xmid)))
    except OverflowError:
        neg = 0.0
    
    return (pos + neg)

def calculate_score(self) -> None:
    # member function of inidividual
    natoms: int = self.m_subgraph.m_natoms
    global_natoms: int = self.m_graph.m_natoms
    subgraph_copy: mgraph.subgraph = mgraph.subgraph(self.m_subgraph)

    edges: List[Tuple[int, int]] = self.m_graph.m_edges
    feasible_edges: List[int] = self.m_subgraph.m_feasible_edges
    node_sg_nidx: List[int] = self.m_graph.m_node_sg_nidx
    broken_bonds: List[int] = []

    conjugated_systems: Set[int] = set()
    hyperconjugated_systems: Set[int] = set()

    for isol, sol in enumerate(self.m_solution):
        if sol == 1:
            # edge has been cut
            edge: Tuple[int, int] = edges[feasible_edges[isol]]
            node1: int = edge[0]
            node2: int = edge[1]
            node1_idx: int = node_sg_nidx[node1]
            node2_idx: int = node_sg_nidx[node2]

            subgraph_copy.delete_edge(node1_idx, node2_idx)
            broken_bonds.append(node1_idx)
            broken_bonds.append(node2_idx)

            box_id1: int = self.m_graph.get_boxid(node1)
            box_id2: int = self.m_graph.get_boxid(node2)

            conjugated_systems1: Set[int] = self.m_boxarray.get_boxes()[box_id1].m_conj_systems
            conjugated_systems2: Set[int] = self.m_boxarray.get_boxes()[box_id2].m_conj_systems

            hyperconjugated_systems1: Set[int] = self.m_boxarray.get_boxes()[box_id1].m_hyper_systems
            hyperconjugated_systems2: Set[int] = self.m_boxarray.get_boxes()[box_id2].m_hyper_systems

            for conj_system in self.m_subgraph.m_conjugated_systems:
                if conj_system in conjugated_systems1 or conj_system in conjugated_systems2:
                    conjugated_systems.add(conj_system)
            
            for hyperconj_system in self.m_subgraph.m_hyperconjugated_systems:
                if hyperconj_system in hyperconjugated_systems1 or hyperconj_system in hyperconjugated_systems2:
                    hyperconjugated_systems.add(hyperconj_system)


    fragid: List[int] = [0 for _ in range(natoms)]
    nfrags: int = gtraverse.determine_fragid(fragid, subgraph_copy)

    # define monomers and dimers
    monomer_list: List[molecule.Molecule] = []
    dimer_list: List[molecule.Molecule] = []

    for ifrag in range(0, nfrags):
        monomer: molecule.Molecule = molecule.Molecule([ifrag])
        monomer_list.append(monomer)
    
    for iatom in range(0, natoms):
        fid: int = fragid[iatom]
        atom_idx: int = self.m_subgraph.m_nodes[iatom]
        monomer_list[fid - 1].m_atom_ids.append(atom_idx)
    
    for ifrag in range(0, nfrags): 
        monomer_list[ifrag].set_natoms_nohcap()
    
    # adding hydrogen caps to monomers
    for isol, sol in enumerate(self.m_solution):
        if sol == 1:
            edge: Tuple[int, int] = edges[feasible_edges[isol]]
            node1: int = edge[0]
            node2: int = edge[1]

            node1_idx: int = node_sg_nidx[node1]
            node2_idx: int = node_sg_nidx[node2]

            fid1: int = fragid[node1_idx]
            fid2: int = fragid[node2_idx]

            if (fid1 != fid2):
                monomer_list[fid1 - 1].m_atom_ids.append(node2 + global_natoms)
                monomer_list[fid2 - 1].m_atom_ids.append(node1 + global_natoms)
            else:
                self.m_solution[isol] = 0
    
    for ifrag in range(0, nfrags): 
        monomer_list[ifrag].set_natoms()

    # now create dimers
    ndimers: int = int((nfrags * (nfrags-1))/2)
    mon_dimer_map: List[int] = [-1 for _ in range(ndimers)]
    dimer_count: int = 0
    for imon in range(0, nfrags):
        for jmon in range(0, imon):
            dimer: molecule.Molecule = molecule.Molecule([jmon, imon], monomer_list[jmon], monomer_list[imon],
                                     fragid, self.m_graph)
            dimer_list.append(dimer)
            
            y: int = imon
            x: int = jmon
            idx: int = int(y*(y-1)/2 + x)
            mon_dimer_map[idx] = dimer_count
            dimer_count += 1

    dimer_energy: List[float] = [0.0 for _ in range(len(dimer_list))]

    pe_sf: float
    self.p_comp = 1 / nfrags
    self.p_vol, self.p_vrange, pe_sf = self.vol_penalty(subgraph_copy)
    self.p_conj = self.conjugation_penalty(subgraph_copy, conjugated_systems)
    self.p_hyper = self.hyperconjugation_penalty(subgraph_copy, hyperconjugated_systems)
    self.pe_diff = uff.mbe1_diff(self.m_graph, monomer_list, dimer_list, dimer_energy)
    self.p_pe = potential_energy_sigmoid(self.pe_diff, pe_sf)

    final_penalty = self.m_graph.m_beta_vrange * self.p_vrange
    final_penalty += self.m_graph.m_beta_comp * self.p_comp
    final_penalty += self.m_graph.m_beta_pe * self.p_pe
    final_penalty += self.m_graph.m_beta_conj * self.p_conj
    final_penalty += self.m_graph.m_beta_hyper * self.p_hyper
    final_penalty += self.m_graph.m_beta_vol * self.p_vol
    self.m_score = final_penalty
    # update off limit edges
    for isol, sol in enumerate(self.m_solution):
        if (sol == 1):
            edge: Tuple[int, int] = edges[feasible_edges[isol]]
            node1: int = edge[0]
            node2: int = edge[1]
            node1_idx: int = node_sg_nidx[node1]
            node2_idx: int = node_sg_nidx[node2]

            fid1: int = fragid[node1_idx] - 1
            fid2: int = fragid[node2_idx] - 1

            jmon: int
            imon: int

            if (fid1 < fid2):
                jmon = fid1
                imon = fid2
            else:
                imon = fid1
                jmon = fid2
            
            y: int = imon
            x: int = jmon
            idx: int = int(y*(y-1)/2 + x)
            dimer_idx: int = mon_dimer_map[idx]

            if abs(dimer_energy[dimer_idx]) > 10.0:
                self.m_off_lim_edges[isol] = 1