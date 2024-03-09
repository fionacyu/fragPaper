import os
import sys
import math
import random
from typing import List, Tuple, Set, Dict

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
penalty_dir = os.path.join(current_dir, '..', 'penalties')

sys.path.append(current_dir)
sys.path.append(penalty_dir)
sys.path.append(charac_dir)

import mgraph
import boxing
import penalty
import initpop

class individual_sorter:
    def __init__(self, idx: int, score: float) -> None:
        self.score: float = score
        self.idx: int = idx

class unphysical_child:
    def __init__(self, idx: int, natoms_bad: int) -> None:
        self.child_idx: int = idx
        self.natoms_bad: int = natoms_bad

class individual:
    def __init__(self, initial_solution: List[int], no_genes: int, graph: mgraph.mgraph,
                 subgraph: mgraph.subgraph, target_fsize: int, boxarray: boxing.mbox_array) -> None:
        
        self.m_target_size: int = target_fsize
        self.m_ngenes: int = no_genes
        self.m_off_lim_edges: List[int] = [0 for _ in range(self.m_ngenes)]

        # the following are references to pre-existing objects, these are not modified
        self.m_graph: mgraph.mgraph = graph
        self.m_subgraph: mgraph.subgraph = subgraph
        self.m_boxarray: boxing.mbox_array = boxarray

        # copy solution
        self.m_solution: List[int] = initial_solution.copy()

        self.p_comp: float
        self.p_vol: float
        self.p_vrange: float
        self.p_pe: float
        self.p_conj: float
        self.p_hyper: float
        self.pe_diff: float

        self.m_score: float
    
    def calculate_score(self) -> float:
        return penalty.calculate_score(self)
    
    def vol_penalty(self, subgraph_copy: mgraph.subgraph) -> float:
        return penalty.vol_penalty(self, subgraph_copy)
    
    def calculate_fragment_volume(self, subgraph_copy: mgraph.subgraph, seed_node: int, colors: List[int]) -> Tuple[float, int]:
        return penalty.calculate_fragment_volume(self, subgraph_copy, seed_node, colors)

    def conjugation_penalty(self, subgraph_copy: mgraph.subgraph, conjugated_systems: Set[int]) -> float:
        return penalty.conjugation_penalty(self, subgraph_copy, conjugated_systems)
    
    def hyperconjugation_penalty(self, subgraph_copy: mgraph.subgraph, hyperconjugated_systems: Set[int]) -> float:
        return penalty.hyperconjugation_penalty(self, subgraph_copy, hyperconjugated_systems)

    def off_lim_edge_status(self, igene: int) -> int:
        return self.m_off_lim_edges[igene]

class population:
    def __init__(self, graph: mgraph.mgraph, subgraph: mgraph.subgraph, target_fsize: int,
                 boxarray: boxing.mbox_array) -> None:
        
        self.m_target_size: int = target_fsize
        self.m_generation: int = 0
        self.m_min_score_iters: int = 0
        self.m_min_score: float = 1.0
        self.m_ngenes: int = len(subgraph.m_feasible_edges)
        self.m_best_sol: List[int] = [0 for _ in range(self.m_ngenes)]

        self.p_comp: float
        self.p_vol: float
        self.p_vrange: float
        self.p_pe: float
        self.p_conj: float
        self.p_hyper: float
        self.pe_diff: float

        # references to pre-existing objects
        self.m_graph: mgraph.mgraph = graph
        self.m_subgraph: mgraph.subgraph = subgraph
        self.m_boxarray: boxing.mbox_array = boxarray

        self.m_off_lim_edges: List[int] = [0 for _ in range(self.m_ngenes)]

        initial_population: List[List[int]] = initpop.get_initial_population(graph, subgraph, target_fsize)
        print(f"initial_population size: {len(initial_population)}")
        
        init_population_size: int = len(initial_population)
        self.num_parents_mating: int

        if (init_population_size > 1 and init_population_size <= 8):
            self.m_num_parents_mating = 2
        elif (init_population_size == 1):
            self.m_num_parents_mating = 1
        elif (init_population_size > 8):
            self.m_num_parents_mating = math.floor(0.25 * init_population_size)
        
        self.m_num_offspring: int = init_population_size - self.m_num_parents_mating
        self.m_mutation_num_genes: int = math.floor(0.1 * self.m_ngenes)
        if (self.m_mutation_num_genes < 1):
            self.m_mutation_num_genes = 0
        
        self.m_solutions: List[individual] = []
        individual_list: List[individual_sorter] = []
        
        for isol, solution in enumerate(initial_population):
            indv: individual = individual(solution, self.m_ngenes, self.m_graph, self.m_subgraph, self.m_target_size, self.m_boxarray)
            indv.calculate_score()
            self.m_solutions.append(indv)
            individual_list.append(individual_sorter(isol, indv.m_score))
        
        individual_list.sort(key=lambda x: x.score)
        local_min_score: float = individual_list[0].score
        local_argmin: int = individual_list[0].idx

        self.m_min_score_iters += 1
        if (local_min_score < self.m_min_score and self.m_solutions[local_argmin].p_comp < 1.0):
            self.m_min_score = local_min_score
            self.m_best_sol = self.m_solutions[local_argmin].m_solution.copy()
            self.m_min_score_iters = 1

            self.p_comp = self.m_solutions[local_argmin].p_comp
            self.p_vol = self.m_solutions[local_argmin].p_vol
            self.p_vrange = self.m_solutions[local_argmin].p_vrange
            self.p_pe = self.m_solutions[local_argmin].p_pe
            self.p_conj = self.m_solutions[local_argmin].p_conj
            self.p_hyper = self.m_solutions[local_argmin].p_hyper

            print(f"score: {self.m_min_score}, p_comp: {self.p_comp}, p_vol: {self.p_vol}, p_vrange: {self.p_vrange}, p_pe: {self.p_pe}, p_conj: {self.p_conj}, p_hyper: {self.p_hyper}, pe_diff: {self.m_solutions[local_argmin].pe_diff}")
        self.update_off_lim_edges()
            
    def update_off_lim_edges(self) -> None:
        for igene in range(0, self.m_ngenes):
            if self.m_off_lim_edges[igene] == 1:
                continue
            for solution in self.m_solutions:
                edge_status: int = solution.off_lim_edge_status(igene)
                if edge_status == 1:
                    self.m_off_lim_edges[igene] = 1
                    break


    def select_parents(self) -> List[int]:
        parent_indices: List[int] = []
        for _ in range(0, self.m_num_parents_mating):
            parent_list: List[individual_sorter] = []

            rdm_idx1: int = random.randint(0, len(self.m_solutions) - 1)
            rdm_idx2: int = random.randint(0, len(self.m_solutions) - 1)
            rdm_idx3: int = random.randint(0, len(self.m_solutions) - 1)

            parent_list.append(individual_sorter(rdm_idx1, self.m_solutions[rdm_idx1].m_score))
            parent_list.append(individual_sorter(rdm_idx2, self.m_solutions[rdm_idx2].m_score))
            parent_list.append(individual_sorter(rdm_idx3, self.m_solutions[rdm_idx3].m_score))

            parent_list.sort(key=lambda x: x.score)
            parent_idx: int = parent_list[0].idx
            parent_indices.append(parent_idx)
        return parent_indices
    
    def check_solution(self, solution: List[int]) -> int:
        natoms: int = self.m_subgraph.m_natoms
        edges: List[Tuple[int, int]] = self.m_graph.m_edges
        node_sg_nidx: List[int] = self.m_graph.m_node_sg_nidx
        feasible_edges: List[int] = self.m_subgraph.m_feasible_edges

        atom_no_broken_bonds: List[int] = [0 for _ in range(natoms)]
        for iedge in range(0, self.m_ngenes):
            if solution[iedge] == 1:
                cut_edge: Tuple[int, int] = edges[feasible_edges[iedge]]

                node1: int = cut_edge[0]
                node2: int = cut_edge[1]

                local_node1_idx: int = node_sg_nidx[node1]
                local_node2_idx: int = node_sg_nidx[node2]

                atom_no_broken_bonds[local_node1_idx] += 1
                atom_no_broken_bonds[local_node2_idx] += 2
        
        natom_bad: int = 0
        for iatom in range(0, natoms):
            if atom_no_broken_bonds[iatom] > 1:
                natom_bad += 1
        
        return natom_bad

    def remove_off_lim_edges(self, solution: List[int]) -> None:
        for igene in range(0, self.m_ngenes):
            if self.m_off_lim_edges[igene] == 1:
                solution[igene] = 0

    def cleanup_solution(self, solution: List[int]) -> None:
        natoms: int = self.m_subgraph.m_natoms
        edges: List[Tuple[int, int]] = self.m_graph.m_edges
        node_sg_nidx: List[int] = self.m_graph.m_node_sg_nidx
        feasible_edges: List[int] = self.m_subgraph.m_feasible_edges

        atom_broken_bonds: List[List[int]] = [[] for _ in range(natoms)]
        for iedge in range(0, self.m_ngenes):
            if solution[iedge] == 1:
                cut_edge: Tuple[int, int] = edges[feasible_edges[iedge]]

                node1: int = cut_edge[0]
                node2: int = cut_edge[1]

                local_node1_idx: int = node_sg_nidx[node1]
                local_node2_idx: int = node_sg_nidx[node2]

                atom_broken_bonds[local_node1_idx].append(iedge)
                atom_broken_bonds[local_node2_idx].append(iedge)
        
        for iatom in range(0, natoms):
            no_broken_bonds: int = len(atom_broken_bonds[iatom])
            if no_broken_bonds > 1:
                keep_idx: int = random.randint(0, no_broken_bonds - 1)
                # keep this bond broken
                for ibond in range(no_broken_bonds):
                    if ibond != keep_idx:
                        fedge_idx: int = atom_broken_bonds[iatom][ibond]
                        solution[fedge_idx] = 0

    def offspring(self, next_generation: List[individual], parent_indices: List[int], 
                  individual_list: List[individual_sorter]) -> None:

        for ioffspring in range(0, self.m_num_offspring):
            success_status: bool = False
            niters: int = 0

            child_solutions: List[List[int]] = []
            unphysical_child_list: List[unphysical_child] = []

            while not success_status and niters < 5:
                parent1_idx: int = ioffspring % self.m_num_parents_mating
                parent2_idx: int = (ioffspring + 1) % self.m_num_parents_mating

                parent1: int = parent_indices[parent1_idx]
                parent2: int = parent_indices[parent2_idx]

                crossover_point: int = random.randint(0, self.m_ngenes - 1)

                child_solution: List[int] = [0 for _ in range(self.m_ngenes)]
                for igene in range(0, crossover_point):
                    child_solution[igene] = self.m_solutions[parent1].m_solution[igene]
                for igene in range(crossover_point, self.m_ngenes):
                    child_solution[igene] = self.m_solutions[parent2].m_solution[igene]
                
                gen: random.Random = random.Random()
                gen.seed()
                
                for _ in range(self.m_mutation_num_genes):
                    gene_to_mutate: int = gen.randint(0, self.m_ngenes - 1)
                    mutated_value: int = random.randint(0, 1)
                    child_solution[gene_to_mutate] = mutated_value

                natoms_bad: int = self.check_solution(child_solution)
                if (natoms_bad == 0):
                    success_status = True
                else:
                    unphysical_child_list.append(unphysical_child(niters, natoms_bad))
                child_solutions.append(child_solution)
                niters += 1
            
            if success_status:
                if self.m_generation < 10:
                    self.remove_off_lim_edges(child_solutions[-1])

                indv: individual = individual(child_solutions[-1], self.m_ngenes, self.m_graph, self.m_subgraph, self.m_target_size, self.m_boxarray)
                indv.calculate_score()
                next_generation.append(indv)
                individual_list.append(individual_sorter(ioffspring + self.m_num_parents_mating, indv.m_score))
            else:
                unphysical_child_list.sort(key=lambda x: x.natoms_bad)
                child_sol_idx: int = unphysical_child_list[0].child_idx
                self.cleanup_solution(child_solutions[child_sol_idx])
                if self.m_generation < 10:
                    self.remove_off_lim_edges(child_solutions[child_sol_idx])
                
                indv: individual = individual(child_solutions[child_sol_idx], self.m_ngenes, self.m_graph, self.m_subgraph, self.m_target_size, self.m_boxarray)
                indv.calculate_score()
                next_generation.append(indv)
                individual_list.append(individual_sorter(ioffspring + self.m_num_parents_mating, indv.m_score))
        self.update_off_lim_edges()

    def create_next_generation(self) -> None:
        self.m_generation += 1
        parent_indices: List[int] = self.select_parents()
        next_generation: List[individual] = []
        individual_list: List[individual_sorter] = []

        for iparent in range(0, self.m_num_parents_mating):
            parent_idx: int = parent_indices[iparent]
            next_generation.append(self.m_solutions[parent_idx])
            individual_list.append(individual_sorter(iparent, self.m_solutions[parent_idx].m_score))
        
        self.offspring(next_generation, parent_indices, individual_list)
        individual_list.sort(key=lambda x: x.score)
        local_min_score: float = individual_list[0].score
        local_argmin: int = individual_list[0].idx

        self.m_min_score_iters += 1
        if (local_min_score < self.m_min_score and next_generation[local_argmin].p_comp < 1.0):
            self.m_min_score = local_min_score
            self.m_best_sol = next_generation[local_argmin].m_solution.copy()
            self.m_min_score_iters = 1

            self.p_comp = next_generation[local_argmin].p_comp
            self.p_vol = next_generation[local_argmin].p_vol
            self.p_vrange = next_generation[local_argmin].p_vrange
            self.p_pe = next_generation[local_argmin].p_pe
            self.p_conj = next_generation[local_argmin].p_conj
            self.p_hyper = next_generation[local_argmin].p_hyper

            print(f"score: {self.m_min_score}, p_comp: {self.p_comp}, p_vol: {self.p_vol}, p_vrange: {self.p_vrange}, p_pe: {self.p_pe}, p_conj: {self.p_conj}, p_hyper: {self.p_hyper}, pe_diff: {next_generation[local_argmin].pe_diff}")
        
        self.m_solutions.clear()
        self.m_solutions = next_generation
