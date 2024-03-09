import os
import sys
import math
import json
import shutil
from typing import List, Tuple, Set, Dict, Any

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
gtraverse_dir = os.path.join(current_dir, '..', 'gtraverse')
uff_dir = os.path.join(current_dir, '..', 'uff')
data_dir = os.path.join(current_dir, '..', 'data')

sys.path.append(charac_dir)
sys.path.append(current_dir)
sys.path.append(gtraverse_dir)
sys.path.append(uff_dir)
sys.path.append(data_dir)

import mgraph
import boxing
import hyperconjugated
import ga
import gtraverse
import uff
import atomic_data

class Optimiser_Frag:
    def __init__(self, graph: mgraph.mgraph, boxarray: boxing.mbox_array) -> None:
        self.m_target_size: int
        self.m_edge_solution: List[int] = [0 for _ in range(graph.m_nedges)]

        # the following are references to pre-existing objects
        self.m_graph: mgraph.mgraph = graph
        self.m_boxarray: boxing.mbox_array = boxarray
        self.m_starting_subgraph: mgraph.subgraph = graph.m_subgraphs[0]
        self.m_target_size: int = graph.m_target_frag_size
        self.m_subgraphs: List[mgraph.subgraph] = []
    
    def create_new_sg(self, parent_subgraph: mgraph.subgraph, best_sol: List[int], child_subgraph: List[mgraph.subgraph]) -> None:
        ngenes: int = len(parent_subgraph.m_feasible_edges)
        subgraph_copy: mgraph.subgraph = mgraph.subgraph(parent_subgraph)
        node_sg_nidx: List[int] = self.m_graph.m_node_sg_nidx
        edges: List[Tuple[int, int]] = self.m_graph.m_edges

        for iedge in range(0, ngenes):
            if best_sol[iedge] == 1:
                cut_edge: Tuple[int, int] = edges[parent_subgraph.m_feasible_edges[iedge]]
                node1: int = cut_edge[0]
                node2: int = cut_edge[1]

                local_node1_idx: int = node_sg_nidx[node1]
                local_node2_idx: int = node_sg_nidx[node2]
        
                subgraph_copy.delete_edge(local_node1_idx, local_node2_idx)
        
        natoms: int = parent_subgraph.m_natoms
        fragid: List[int] = [0 for _ in range(natoms)]
        nfrags: int = gtraverse.determine_fragid(fragid, subgraph_copy)
        child_subgraph.extend([mgraph.subgraph() for _ in range(nfrags)])

        frag_natoms: List[int] = [0 for _ in range(nfrags)]

        for iatom in range(0, natoms):
            fid: int = fragid[iatom] - 1
            frag_natoms[fid] += 1
        
        # distribute feasible edges to child subgraphs
        parent_subgraph_copy: mgraph.subgraph = mgraph.subgraph(parent_subgraph)
        node_no_broken_bonds: List[int] = self.m_graph.m_node_no_broken_bonds

        for iedge in range(0, ngenes):
            edge: Tuple[int, int] = edges[parent_subgraph.m_feasible_edges[iedge]]

            node1: int = edge[0]
            node2: int = edge[1]

            local_node1_idx: int = node_sg_nidx[node1]
            local_node2_idx: int = node_sg_nidx[node2]

            if parent_subgraph.get_degree(local_node1_idx) == 1 or parent_subgraph.get_degree(local_node2_idx) == 1:
                continue

            fid1: int = fragid[local_node1_idx] 
            fid2: int = fragid[local_node2_idx] 

            if fid1 == fid2:
                min_frag_size: int = int(0.3 * self.m_target_size)
                if (node_no_broken_bonds[node1] > 0 or node_no_broken_bonds[node2] > 0):
                    continue
                colors: List[int] = [0 for _ in range(natoms)]
                parent_subgraph_copy.delete_edge(local_node1_idx, local_node2_idx)
                size1: int = gtraverse.bfs_sg(parent_subgraph_copy, local_node1_idx, colors, min_frag_size)
                size2: int = gtraverse.bfs_sg(parent_subgraph_copy, local_node2_idx, colors, min_frag_size)

                if (size1/min_frag_size >= 0.6 and size2/min_frag_size >= 0.6):
                    child_subgraph[fid1 - 1].add_feasible_edge(parent_subgraph.m_feasible_edges[iedge])

                parent_subgraph_copy.add_edge(local_node1_idx, local_node2_idx)
        
        # distribute conjugated and hyperconjugated systems
        box_array: List[boxing.mbox] = self.m_boxarray.get_boxes()
        da_array: List[hyperconjugated.hyper_da] = self.m_graph.m_da_array
        conjugated_indices: List[int] = parent_subgraph.m_conjugated_systems
        hyperconjugated_indices: List[int] = parent_subgraph.m_hyperconjugated_systems

        for iconj in conjugated_indices:
            conjugated_nodes: List[int] = self.m_graph.m_conjugated_systems[iconj].m_nodes

            conj_fids: Set[int] = set()
            for node in conjugated_nodes:
                local_node_idx: int = node_sg_nidx[node]
                fid: int = fragid[local_node_idx]
                conj_fids.add(fid)
            
            if len(conj_fids) == 1:
                conj_fid: int = list(conj_fids)[0]
                child_subgraph[conj_fid - 1].m_conjugated_systems.append(iconj)

        for ihsys in hyperconjugated_indices:
            hyperconjugated_system: hyperconjugated.hypersys = self.m_graph.m_hyperconjugated_systems[ihsys]
            donor_idx: int = hyperconjugated_system.m_donor
            acceptor_idx: int = hyperconjugated_system.m_acceptor

            donor_nodes: List[int] = da_array[donor_idx].m_nodes
            acceptor_nodes: List[int] = da_array[acceptor_idx].m_nodes
            connection_path: List[int] = hyperconjugated_system.m_connection_path

            all_nodes_list: List[int] = donor_nodes.copy() + acceptor_nodes.copy() + connection_path.copy()
            all_nodes_set: Set[int] = set(all_nodes_list)

            hyperconj_fids: Set[int] = set()
            for node in all_nodes_set:
                local_node_idx: int = node_sg_nidx[node]
                fid: int = fragid[local_node_idx]
                hyperconj_fids.add(fid)
            
            if len(hyperconj_fids) == 1:
                hyperconj_fid: int = list(hyperconj_fids)[0]
                child_subgraph[hyperconj_fid - 1].m_hyperconjugated_systems.append(ihsys)

        # add nodes to child subgraph and update node_sg_idx
        for iatom in range(0, natoms):
            fid: int = fragid[iatom] - 1
            global_atom_idx: int = parent_subgraph.get_node(iatom)
            child_subgraph[fid].m_nodes.append(global_atom_idx)
            node_sg_nidx[global_atom_idx] = child_subgraph[fid].get_natoms() - 1
        
        # create adjacency for each subgraph + energy
        for ifrag in range(0, nfrags):
            child_subgraph[ifrag].set_size()
            child_subgraph[ifrag].setup_adjacency()
            uff.calculate_evdw_sg(self.m_graph, child_subgraph[ifrag])

        for iatom in range(0, natoms):
            fid1: int = fragid[iatom] - 1
            neighbours: Set[int] = parent_subgraph.get_neighbours(iatom)
            global_atom_idx: int = parent_subgraph.get_node(iatom)
            new_local_atom_idx: int = node_sg_nidx[global_atom_idx]

            for neighbour in neighbours:
                fid2: int = fragid[neighbour] - 1
                global_neighbour_idx: int = parent_subgraph.get_node(neighbour)
                new_local_neighbour_idx: int = node_sg_nidx[global_neighbour_idx]

                if (fid1 == fid2):
                    child_subgraph[fid1].add_edge(new_local_atom_idx, new_local_neighbour_idx)
        
    def frag(self, subgraph: mgraph.subgraph, child_subgraphs: List[mgraph.subgraph], first: bool = False) -> None:
        natoms: int = subgraph.m_natoms
        size_ratio: float = natoms / self.m_target_size

        target_size: int # we fragment recursively

        if (size_ratio <= 1.0):
            raise SystemExit("Your fragment size is greater than system size.")
        elif (size_ratio > 1.0 and size_ratio <= 1.5):
            target_size = int(natoms / 2.0)
        elif (size_ratio > 1.5 and size_ratio <= 6.0):
            target_size = self.m_target_size
        elif (size_ratio >= 6.0):
            target_size = math.floor(natoms / 5.0)
        
        converged: bool = False
        pop: ga.population = ga.population(self.m_graph, subgraph, target_size, self.m_boxarray)

        while not converged:
            if pop.m_num_offspring == 0:
                break
            pop.create_next_generation()
            if pop.m_min_score_iters > 50 or pop.m_generation >= 100:
                converged = True


        node_no_broken_bonds: List[int] = self.m_graph.m_node_no_broken_bonds
        ngenes: int = len(subgraph.m_feasible_edges)
        best_sol: List[int] = pop.m_best_sol
        edges: List[Tuple[int, int]] = self.m_graph.m_edges

        for iedge in range(0, ngenes):
            if best_sol[iedge] == 1:
                edge_idx: int = subgraph.m_feasible_edges[iedge]
                cut_edge: Tuple[int, int] = edges[edge_idx]
                node1: int = cut_edge[0]
                node2: int = cut_edge[1]
                node_no_broken_bonds[node1] += 1
                node_no_broken_bonds[node2] += 1
                self.m_edge_solution[edge_idx] = 1
        self.create_new_sg(subgraph, best_sol, child_subgraphs)

        print(f"Parent subgraph: ", end="")
        for child_sg in child_subgraphs:
            print(f"{child_sg.m_natoms}, ", end="")
        print(f"\n{'-' * shutil.get_terminal_size()[0]}")

        if len(child_subgraphs) == 1:
            child_subgraphs.clear()
        
        if first:
            temp_subgraphs: List[mgraph.subgraph] = []
            for child_sg in child_subgraphs:
                if (child_sg.m_natoms >= 1.2 * self.m_target_size and len(child_sg.m_feasible_edges) > 0):
                    temp_subgraphs.append(mgraph.subgraph(child_sg))
        
            child_subgraphs.clear()
            child_subgraphs.extend(temp_subgraphs)

    def fragmentation_round(self):
        child_subgraphs_list: List[List[mgraph.subgraph]] = [[] for _ in range(len(self.m_subgraphs))]
        for isubgraph, subgraph in enumerate(self.m_subgraphs):
            self.frag(subgraph, child_subgraphs_list[isubgraph])
        
        child_subgraph_count: int = 0
        for isubgraph in range(len(self.m_subgraphs)):
            child_subgraph_count += len(child_subgraphs_list[isubgraph])
        
        parent_subgraph_count = len(self.m_subgraphs)
        self.m_subgraphs.clear()

        for iparent_subgraph in range(0, parent_subgraph_count):
            for child_subgraph in child_subgraphs_list[iparent_subgraph]:
                if (child_subgraph.m_natoms >= 1.2 * self.m_target_size and len(child_subgraph.m_feasible_edges) > 0):
                    self.m_subgraphs.append(mgraph.subgraph(child_subgraph))
    
    def final_solution(self):
        ngenes: int = len(self.m_starting_subgraph.m_feasible_edges)
        final_solution: List[int] = [0 for _ in range(ngenes)]
        feasible_edges: List[int] = self.m_starting_subgraph.m_feasible_edges

        node_sg_nidx: List[int] = self.m_graph.m_node_sg_nidx
        natoms: int = self.m_graph.m_natoms

        for iatom in range(0, natoms):
            global_atom_idx: int = self.m_starting_subgraph.get_node(iatom)
            node_sg_nidx[global_atom_idx] = iatom
        
        print(f"Broken bonds: ", end="")
        for fiedge in range(0, ngenes):
            iedge: int = feasible_edges[fiedge]
            if self.m_edge_solution[iedge] == 1:
                edge: Tuple[int, int] = self.m_graph.m_edges[iedge]
                node1: int = edge[0]
                node2: int = edge[1]

                print(f"({node1}, {node2}), ", end="")
                final_solution[fiedge] = 1
        print("")
        indv: ga.individual = ga.individual(final_solution, ngenes, self.m_graph, self.m_starting_subgraph, 
                                            self.m_target_size, self.m_boxarray)
        indv.calculate_score()
        final_score: float = indv.m_score
        print(f"Final score {final_score}, p_comp: {indv.p_comp}, p_vol: {indv.p_vol}, p_vrange: {indv.p_vrange}, p_pe: {indv.p_pe}, p_conj: {indv.p_conj}, p_hyper: {indv.p_hyper}, pe_diff: {indv.pe_diff}")
        
    def post_processing(self):
        self.final_solution()
        self.print_output_files()

    def print_output_files(self):
        # frag xyz and topo file
        connectivity: List[Tuple[int, int, int]] = []

        nedges: int = self.m_graph.m_nedges
        edges: List[Tuple[int, int]] = self.m_graph.m_edges
        subgraph_copy: mgraph.subgraph = mgraph.subgraph(self.m_starting_subgraph)
        coordinates: List[float] = self.m_graph.m_coordinates
        elements: List[int] = self.m_graph.m_elements
        charges: List[int] = self.m_graph.m_charges

        natoms: int = self.m_graph.m_natoms
        graph_node_sg: List[int] = [0 for _ in range(natoms)]

        symbol_list: List[str] = []

        for iatom in range(0, natoms):
            global_node_idx: int = self.m_starting_subgraph.get_node(iatom)
            graph_node_sg[global_node_idx] = iatom
            atomic_no: int = elements[iatom]
            symbol: str = atomic_data.atom_symbols[atomic_no - 1]
            symbol_list.append(symbol)
        
        for iedge in range(0, nedges):
            if self.m_edge_solution[iedge] == 1:
                edge: Tuple[int, int] = edges[iedge]

                node1: int = edge[0]
                node2: int = edge[1]
            
                local_node1_idx: int = graph_node_sg[node1]
                local_node2_idx: int = graph_node_sg[node2]

                subgraph_copy.delete_edge(local_node1_idx, local_node2_idx)
        
        fragid: List[int] = [0 for _ in range(natoms)]
        nfrags: int = gtraverse.determine_fragid(fragid, subgraph_copy)
        print(f"{nfrags} fragments generated.")

        frag_atoms: List[List[int]] = [[] for _ in range(nfrags)]
        frag_atoms_nohcap: List[List[int]] = [[] for _ in range(nfrags)]
        frag_charges: List[int] = [0 for _ in range(nfrags)]

        for iatom in range(0, natoms):
            local_atom_idx: int = graph_node_sg[iatom]
            fid: int = fragid[local_atom_idx]

            frag_atoms[fid - 1].append(iatom)
            frag_atoms_nohcap[fid - 1].append(iatom)
            frag_charges[fid -1] += charges[iatom]
        
        hcap_neighbour: List[int] = [-1 for _ in range(natoms)]
        for iedge in range(0, nedges):
            if self.m_edge_solution[iedge] == 1:
                edge: Tuple[int, int] = edges[iedge]

                node1: int = edge[0]
                node2: int = edge[1]
            
                local_node1_idx: int = graph_node_sg[node1]
                local_node2_idx: int = graph_node_sg[node2]

                fid1: int = fragid[local_node1_idx]
                fid2: int = fragid[local_node2_idx]

                if fid1 != fid2:
                    frag_atoms[fid1 - 1].append(node2 + natoms)
                    frag_atoms[fid2 - 1].append(node1 + natoms)

                    hcap_neighbour[node1] = node2
                    hcap_neighbour[node2] = node1

                    connectivity.append((node1, node2, 1))
        
        frag_xyz_string = ""
        for ifrag in range(0, nfrags):
            frag_xyz_string += f"{len(frag_atoms[ifrag])}\n"
            frag_xyz_string += f"{ifrag}.xyz\n"

            for atom in frag_atoms[ifrag]:

                if atom >= natoms: # hydrogen cap
                    atom_idx: int = atom - natoms

                    x_coord: float = coordinates[3 * atom_idx]
                    y_coord: float = coordinates[3 * atom_idx + 1]
                    z_coord: float = coordinates[3 * atom_idx + 2]

                    frag_xyz_string += "{}   {}   {}   {}\n".format("H", x_coord, y_coord, z_coord) 
                else:
                    atomic_no: int = elements[atom] 
                    symbol: str = atomic_data.atom_symbols[atomic_no - 1]

                    x_coord: float = coordinates[3 * atom]
                    y_coord: float = coordinates[3 * atom + 1]
                    z_coord: float = coordinates[3 * atom + 2]
                    frag_xyz_string += "{}   {}   {}   {}\n".format(symbol, x_coord, y_coord, z_coord) 
        with open(f"{self.m_graph.m_name}_all_fragments.xyz", "w") as f:
            f.write(frag_xyz_string)
        f.close()


        output_json: Dict[str, Any] = {}
        output_json["symbols"] = symbol_list
        output_json["coordinates"] = coordinates.copy()
        output_json["connectivity"] = connectivity.copy()
        output_json["fragment_charges"] = frag_charges.copy()
        output_json["fragments"] = frag_atoms_nohcap.copy()

        with open(f"{self.m_graph.m_name}_topo.qdxf", "w") as f:
            json.dump(output_json, f, indent=4)
        f.close()
                
    def run(self) -> None:
        fragmentation_round: int = 0
        print(f"Fragmentation round: {fragmentation_round}")
        print(f"{'-' * shutil.get_terminal_size()[0]}")
        self.frag(self.m_starting_subgraph, self.m_subgraphs, True)

        while True:
            fragmentation_round += 1
            print(f"Fragmentation round: {fragmentation_round}")
            print(f"{'-' * shutil.get_terminal_size()[0]}")
            self.fragmentation_round()
            if (len(self.m_subgraphs) == 0 or fragmentation_round >= 50):
                break
        
        self.post_processing()