import math
from collections import Counter
from typing import List, Tuple, Set, Dict

import atomic_data
import conjugated
import rings
import hyperconjugated

class subgraph:
    def __init__(self, value = None) -> None: # also acts as copy constructor
        self.m_ncycles: int
        self.m_cycles: List[rings.cycle] = []
        self.m_naromatic_systems: int
        self.m_aromatic_cycles: List[Set[int]] = []
        
        self.m_single_bonds: List[int] = []
        
        if not value:
            self.m_offset: int # points to the atom w the smallest global index
            self.m_nodes: List[int] = []
            self.m_natoms: int
            self.m_energy: float = 0.0
            self.m_adjacency: List[Set[int]] = []
            self.m_feasible_edges: List[int] = [] # contains edge indices localed in global mgraph
            self.m_conjugated_systems: List[int] = []
            self.m_hyperconjugated_systems: List[int] = []
        else:
            self.m_natoms = value.m_natoms
            self.m_offset = min(value.m_nodes)
            self.m_nodes = value.m_nodes.copy()
            self.m_energy = value.m_energy
            self.m_adjacency = [x.copy() for x in value.m_adjacency]
            self.m_feasible_edges = value.m_feasible_edges.copy()
            self.m_conjugated_systems = value.m_conjugated_systems.copy()
            self.m_hyperconjugated_systems = value.m_hyperconjugated_systems.copy()

    
    def get_node(self, node: int) -> int:
        return self.m_nodes[node]

    def get_natoms(self) -> int:
        return len(self.m_nodes)

    def set_size(self) -> None:
        self.m_natoms = len(self.m_nodes)

    def setup_adjacency(self) -> None:
        self.m_adjacency = [set() for _ in range(self.m_natoms)]
    
    def setup_cycles(self, cycle_count: int) -> None:
        self.m_ncycles = cycle_count
        self.m_cycles = [rings.cycle() for _ in range(cycle_count)]

    def setup_cycle(self, cycle_idx: int, cycle_edge_count: int) -> None:
        self.m_cycles[cycle_idx].set_edge_count(cycle_edge_count)
    
    def setup_cycle_boxes(self, cycle_idx: int, box_no: int) -> None:
        self.m_cycles[cycle_idx].set_box_no(box_no)

    def get_degree(self, node: int) -> int:
        return len(self.m_adjacency[node])
    
    def get_neighbours(self, node: int) -> Set[int]:
        return self.m_adjacency[node]

    def delete_edge(self, node1: int, node2: int) -> None:
        self.m_adjacency[node1].remove(node2)
        self.m_adjacency[node2].remove(node1)
         
    def add_edge(self, node1: int, node2: int) -> None:
        self.m_adjacency[node1].add(node2)
        self.m_adjacency[node2].add(node1)
    
    def add_feasible_edge(self, iedge: int) -> None:
        self.m_feasible_edges.append(iedge)

class mgraph:
    def __init__(self) -> None:
        self.m_name: str
        self.m_natoms: int
        self.m_coordinates: List[float] = []
        self.m_elements: List[int] = []
        self.m_coordination_number: List[int] = [] # the hybridisation, sp = 2, sp2 = 3, sp3 = 4
        self.m_charges: List[int] = []
        self.m_lone_electrons: List[int] = []
        self.m_degree: List[int] = []
        self.m_bonding_electrons: List[float] = []
        self.m_pi_electrons: List[int] = [] 
        self.m_ring_status: List[int] = []
        self.m_atom_types: List[int] = []
        self.m_conjugation_status: List[int] = []
        self.m_node_no_broken_bonds: List[int] = []

        self.m_atom_pair_connections: List[int] = []
        self.m_range: List[float] = []

        self.m_adjacency: List[List[Set[int]]] = [] # includes adjancy info between atoms: atoms 1, 2, and 3 bonds apart
        self.m_nedges: int # eq. to number of edges

        self.m_nx: int
        self.m_ny: int
        self.m_nz: int

        self.m_ref_vol_atom: float = 0.0 # reference volume of an atom in system
        self.m_target_frag_size: float

        self.m_edges: List[Tuple[int, int]]  = []
        self.m_bond_orders: List[float] = []
        self.m_edge_change: List[int] = []

        self.m_subgraphs: List[subgraph] = []
        self.m_conjugated_systems: List[conjugated.conjsys] = []
        self.m_da_array: List[hyperconjugated.hyper_da] = []
        self.m_hyperconjugated_systems: List[hyperconjugated.hypersys] = []

        self.m_sg_count: int
        self.m_da_count: int

        self.m_node_sg: List[int] = []
        self.m_node_sg_nidx: List[int] = []

        # weights
        self.m_beta_pe = 0.13581564349379915
        self.m_beta_conj = 0.14594295898207302
        self.m_beta_hyper = 0.3133251394715002
        self.m_beta_vol = 0.10941644829196222
        self.m_beta_comp = 0.0014263024478876928
        self.m_beta_vrange = 0.29407350731277765

    def set_range(self, x_min: float, x_max: float, y_min:float, 
        y_max: float, z_min: float, z_max: float) -> None:
        self.m_range = [x_min, x_max, y_min, y_max, z_min, z_max]
    
    def set_natoms(self, natoms: int) -> None:
        self.m_natoms = natoms
        self.m_lone_electrons = [0 for _ in range(natoms)]
        self.m_conjugation_status = [0 for _ in range(natoms)]
        self.m_degree = [0 for _ in range(natoms)]
        self.m_pi_electrons = [0 for _ in range(natoms)]
        self.m_ring_status = [0 for _ in range(natoms)]
        self.m_atom_types = [0 for _ in range(natoms)]
        self.m_node_no_broken_bonds = [0 for _ in range(natoms)]
        self.m_bonding_electrons = [0 for _ in range(natoms)]
        self.m_coordination_number = [0 for _ in range(natoms)]
        self.m_node_sg = [0 for _ in range(natoms)]
        self.m_node_sg_nidx = [0 for _ in range(natoms)]
        self.m_atom_pair_connections = [0 for _ in range(int(natoms * (natoms-1)/2))] 

        self.m_adjacency = [[set() for _ in range(3)] for _ in range(natoms)]
    
    def distance(self, iatom: int, jatom: int) -> float:
        x1: float = self.m_coordinates[3 * iatom]
        y1: float = self.m_coordinates[3 * iatom + 1]
        z1: float = self.m_coordinates[3 * iatom + 2]

        x2: float = self.m_coordinates[3 * jatom]
        y2: float = self.m_coordinates[3 * jatom + 1]
        z2: float = self.m_coordinates[3 * jatom + 2]

        return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2))
    
    def distance2(self, iatom: int, jatom: int) -> float:
        # distance squared
        x1: float = self.m_coordinates[3 * iatom]
        y1: float = self.m_coordinates[3 * iatom + 1]
        z1: float = self.m_coordinates[3 * iatom + 2]

        x2: float = self.m_coordinates[3 * jatom]
        y2: float = self.m_coordinates[3 * jatom + 1]
        z2: float = self.m_coordinates[3 * jatom + 2]

        return (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2)

    def check_atom_distance(self) -> None:
        for iatom in range(0, self.m_natoms):
            for jatom in range(0, iatom):
                r: float = self.distance(iatom, jatom)
                if (r < 0.2):
                    raise SystemExit(f"Atoms {iatom} and {jatom} are less than 0.2 Å apart")

    def add_edge_adjacency(self, bond: tuple) -> None:
        node1: int = bond[0]
        node2: int = bond[1]
        
        atom_pair_idx: int
        if (node1 < node2):
            atom_pair_idx = int(node2 * (node2 - 1) / 2 + node1)
        else:
            atom_pair_idx = int(node1 * (node1 - 1) / 2 + node2)
        
        self.m_atom_pair_connections[atom_pair_idx] = 1

        node1_neighbours: Set[int] = self.m_adjacency[node1][0]
        node2_neighbours: Set[int] = self.m_adjacency[node2][0]

        for neighbour1 in node1_neighbours:
            for neighbour2 in self.m_adjacency[neighbour1][0]:
                if neighbour2 != node1:
                    self.m_adjacency[neighbour2][2].add(node2)
                    self.m_adjacency[node2][2].add(neighbour2)

            self.m_adjacency[neighbour1][1].add(node2)
            self.m_adjacency[node2][1].add(neighbour1)

            atom_pair_idx: int
            if (neighbour1 < node2):
                atom_pair_idx = int(node2 * (node2 - 1) / 2 + neighbour1)
            else:
                atom_pair_idx = int(neighbour1 * (neighbour1 - 1) / 2 + node2)
            # print("atom_pair_idx: ", atom_pair_idx)
            self.m_atom_pair_connections[atom_pair_idx] = 1

        for neighbour1 in node2_neighbours:
            for neighbour2 in self.m_adjacency[neighbour1][0]:
                if neighbour2 != node2:
                    self.m_adjacency[neighbour2][2].add(node1)
                    self.m_adjacency[node1][2].add(neighbour2)
            
            self.m_adjacency[neighbour1][1].add(node1)
            self.m_adjacency[node1][1].add(neighbour1)

            atom_pair_idx: int
            if (neighbour1 < node1):
                atom_pair_idx = int(node1 * (node1 - 1) / 2 + neighbour1) 
            else:
                atom_pair_idx = int(neighbour1 * (neighbour1 - 1) / 2 + node1)
            self.m_atom_pair_connections[atom_pair_idx] = 1

        self.m_adjacency[node1][0].add(node2)
        self.m_adjacency[node2][0].add(node1)
        # print(f"self.m_adjacency: {self.m_adjacency}")

    def get_neighbours(self, node: int) -> Set[int]:
        return self.m_adjacency[node][0]

    def get_neighbours2(self, node: int) -> Set[int]:
        return self.m_adjacency[node][1]

    def get_neighbours3(self, node: int) -> Set[int]:
        return self.m_adjacency[node][2]
        
    def set_edge_count(self) -> None:
        self.m_nedges = len(self.m_edges)
        self.m_edge_change = [-1 for _ in range(self.m_nedges)]


    def determine_hybridisation(self) -> None:
        for iatom in range(0, self.m_natoms):
            valence_elec: int = atomic_data.get_valence(self.m_elements[iatom])
            degree: int = self.m_degree[iatom]
            bonding_elec: int = self.m_bonding_electrons[iatom]
            charge: int = self.m_charges[iatom]

            cbe: float = valence_elec - 0.5 * bonding_elec
            electron_domain: int = math.ceil(degree + 0.5 * (cbe - charge))

            self.m_coordination_number[iatom] = electron_domain
            self.m_lone_electrons[iatom] = math.ceil(0.5 * (cbe - charge))
            if (self.m_lone_electrons[iatom] < 0):
                self.m_lone_electrons[iatom] = 0

    def check_hybridisation(self) -> None:
        # cases of N, O, S in aromatic 5 membered rings (furan, pyrrole, thiophene)
        # have been given a coordination number of 4 (sp3 hybridized), need to change to 3(sp2 hybridized)

        for iatom in range(0, self.m_natoms):
            atomic_number: int = self.m_elements[iatom]
            coordination_number: int = self.m_coordination_number[iatom]

            neighbours: Set[int] = self.get_neighbours(iatom)
            sp2_neighbour_no: int = 0
            match atomic_number:
                case 7:
                    if coordination_number < 4:
                        continue
                    for neighbour in neighbours:
                        neighbour_coordination_number: int = self.m_coordination_number[neighbour]
                        neighbour_atomic_number: int = self.m_elements[neighbour]

                        if (neighbour_coordination_number < 4 and neighbour_coordination_number > 1 and neighbour_atomic_number == 6):
                            sp2_neighbour_no += 1
                    if sp2_neighbour_no >= 1:
                        self.m_coordination_number[iatom] = 3
                        sp2_neighbour_no = 0
                        continue
                case 8 | 16:
                    if coordination_number < 4:
                        continue
                    for neighbour in neighbours:
                        neighbour_coordination_number: int = self.m_coordination_number[neighbour]
                        if neighbour_coordination_number < 4 and neighbour_coordination_number > 1:
                            sp2_neighbour_no += 1
                    if sp2_neighbour_no >= 2:
                        self.m_coordination_number[iatom] = 3
                        continue
                
    def set_atom_types(self) -> None: # for UFF
        for iatom in range(0, self.m_natoms):
            atomic_number: int = self.m_elements[iatom]
            coordination_number: int = self.m_coordination_number[iatom]

            hash_value: int = atomic_number + 53 * coordination_number
            atom_type_idx: int
            oxygen_neighbours: int = 0

            neighbours: List[int] = self.m_adjacency[iatom][0]
            match hash_value:
                case 54: # H, normal
                    atom_type_idx = 0
                case 112: #C_1
                    atom_type_idx = 10
                case 165: #C_2 
                    atom_type_idx = 9
                case 218: #C_3
                    atom_type_idx = 7
                case 113: #N_1
                    atom_type_idx = 14
                case 166: #N_2
                    atom_type_idx = 13
                case 219: #N_3
                    atom_type_idx = 11
                case 114: # O_1
                    atom_type_idx = 19
                case 167: # O_2
                    atom_type_idx = 18
                case 220: # O_3
                    atom_type_idx = 15
                case 221: # F
                    atom_type_idx = 20
                case 229: #Cl
                    atom_type_idx = 34
                case 247: # Br
                    atom_type_idx = 54
                case 265: # I
                    atom_type_idx = 73
                case 228: # S_3
                    atom_type_idx = 29 # generic case: S_3+2
                    # need to account for charges/oxidation state

                    for neighbour in neighbours:
                        neighbour_atomic_number:int = self.m_elements[neighbour]
                        if neighbour_atomic_number == 8:
                            oxygen_neighbours += 1

                        if (oxygen_neighbours == 2 and len(neighbours) == 2):
                            atom_type_idx = 30
                        elif (oxygen_neighbours == 3 and len(neighbours) == 3):
                            atom_type_idx = 31
                        elif (oxygen_neighbours == 4 and len(neighbours) == 4):
                            atom_type_idx = 31
                case 175: # S_2
                    atom_type_idx = 33 # S_2
                case 227: # P_3
                    atom_type_idx = 26 # generic P: P_3+3
                    # account for charges/oxidation state
                case _:
                    raise SystemExit(f"No atom type for atom {iatom} with atomic no. {atomic_number} and {coordination_number} electron domains")
            self.m_atom_types[iatom] = atom_type_idx
    
    def get_boxid(self, node: int) -> int:
        box_len: float = 2.157

        xmin: float = self.m_range[0]
        ymin: float = self.m_range[2]
        zmin: float = self.m_range[4]

        x: float = self.m_coordinates[3 * node]
        y: float = self.m_coordinates[3 * node + 1]
        z: float = self.m_coordinates[3 * node + 2]

        bx: float = math.floor( (x - xmin)/box_len)
        by: float = math.floor( (y - ymin)/box_len)
        bz: float = math.floor( (z - zmin)/box_len)

        box_id: int = int((bz * self.m_nx * self.m_ny) + (by * self.m_nx) + bx)

        return box_id
    
    def check_bonds_adjacent_conj(self) -> None:
        non_conj_atoms_adjacent: List[List[Tuple[int, int]]] = [[] for _ in range(self.m_natoms)]

        for iedge in range(0, self.m_nedges):
            edge: Tuple[int] = self.m_edges[iedge]

            node1: int = edge[0]
            node2: int = edge[1]

            if (self.m_conjugation_status[node1] == 1 and self.m_conjugation_status[node2] == 0):
                non_conj_atoms_adjacent[node2].append((node1, iedge))
            elif (self.m_conjugation_status[node1] == 0 and self.m_conjugation_status[node2] == 1):
                non_conj_atoms_adjacent[node1].append((node2, iedge))
        
        for iatom in range(0, self.m_natoms):
            if (len(non_conj_atoms_adjacent[iatom]) > 1):
                min_priority: float = 100.0

                conj_atom_priority: List[Tuple[float, int]] = []

                for neighbour_info in non_conj_atoms_adjacent[iatom]:
                    conj_neighbour: int = neighbour_info[0]
                    edge_idx: int = neighbour_info[1]
                    conj_atom_degree: int = self.m_degree[conj_neighbour]
                    
                    priority: float = (self.m_lone_electrons[conj_neighbour] / conj_atom_degree)
                    conj_atom_priority.append((priority, edge_idx))

                    if (priority < min_priority):
                        min_priority = priority

                conj_atom_priority.sort()
                iedge_min_priority: List[int] =[]

                if (len(conj_atom_priority) > 1):
                    for neighbour_info in conj_atom_priority:
                        if (neighbour_info[0] == min_priority):
                            iedge_min_priority.append(neighbour_info[1])
                    
                    low_priority_edge: int = iedge_min_priority[0]

                    if (len(iedge_min_priority) == len(conj_atom_priority)):
                        for ineigh in range(1, len(conj_atom_priority)):
                            self.m_edge_change[iedge_min_priority[ineigh]] = low_priority_edge
                    else:
                        for ineigh in range(len(iedge_min_priority), len(conj_atom_priority)):
                            self.m_edge_change[conj_atom_priority[ineigh][1]] = low_priority_edge
            elif (len(non_conj_atoms_adjacent[iatom]) == 1):
                edge_adj_conj: int = non_conj_atoms_adjacent[iatom][0][1]
                self.m_edge_change[edge_adj_conj] = -2
    
    def setup_sg_adjacency(self) -> None:
        for sg in self.m_subgraphs:
            sg.setup_adjacency()

    def sg_add_edge(self, subgraph_idx: int, node1_idx: int, node2_idx: int) -> None:
        self.m_subgraphs[subgraph_idx].m_adjacency[node1_idx].add(node2_idx)
        self.m_subgraphs[subgraph_idx].m_adjacency[node2_idx].add(node1_idx)


    def set_edges_sg(self) -> None:
        self.setup_sg_adjacency()

        single_bond_priority: List[tuple[float, int]] = []

        for iedge, edge in enumerate(self.m_edges):
            node1: int = edge[0]
            node2: int = edge[1]

            subgraph1: int = self.m_node_sg[node1] - 1

            node1_idx: int = self.m_node_sg_nidx[node1]
            node2_idx: int = self.m_node_sg_nidx[node2]

            self.sg_add_edge(subgraph1, node1_idx, node2_idx)

            if self.m_bond_orders[iedge] == 1.0 and self.m_edge_change[iedge] == -1:
                node1_lone_elec: int = self.m_lone_electrons[node1]
                node2_lone_elec: int = self.m_lone_electrons[node2]

                node1_degree: int = self.m_degree[node1]
                node2_degree: int = self.m_degree[node2]

                priority: float = 0.5 * (node1_lone_elec/node1_degree + node2_lone_elec/node2_degree)
                single_bond_priority.append((priority, iedge))
        

        single_bond_priority.sort()

        single_bond_no: int = len(single_bond_priority)
        min_priority: float = single_bond_priority[0][0]

        sorted_priorities: List[float] = [min_priority]
        current_priority: float = min_priority

        for single_bond_info in single_bond_priority:
            priority: float = single_bond_info[0]
            if current_priority != priority:
                sorted_priorities.append(priority)
                current_priority = priority

        edge_index: int = math.ceil(0.8 * single_bond_no)
        priority_cutoff_incl: float = single_bond_priority[edge_index][0]

        no_high_priority_bonds: bool = False

        if (priority_cutoff_incl == min_priority):
            if (len(sorted_priorities) == 1):
                no_high_priority_bonds = True
            else:
                priority_cutoff_incl = sorted_priorities[1]
            

        if not no_high_priority_bonds:
            for iedge in range(0, single_bond_no):
                if (single_bond_priority[iedge][0] >= priority_cutoff_incl):
                    self.m_edge_change[single_bond_priority[iedge][1]] = -2
        
        for single_bond_info in single_bond_priority:
            edge_idx: int = single_bond_info[1]
            node1: int = self.m_edges[edge_idx][0]
            node2: int = self.m_edges[edge_idx][1]

            subgraph1: int = self.m_node_sg[node1] - 1
            if (self.m_edge_change[edge_idx] == -1):
                self.m_subgraphs[subgraph1].m_single_bonds.append(edge_idx)

        
    def set_mirror(self, natoms: int) -> None:
        self.m_natoms = natoms
        self.m_adjacency = [[set()] for _ in range(natoms)]

    def add_edge_only_adjacency(self, node1: int, node2: int) -> None:
        self.m_adjacency[node1][0].add(node2)
        self.m_adjacency[node2][0].add(node1)
    
    def set_aromatic_at(self, node: int) -> None:
        atom_type: int = self.m_atom_types[node]

        if atom_type in {8, 12, 17, 32}:
            # already labelled as aromatic
            return
        atomic_number: int = self.m_elements[node]
        atom_type_index: int

        match atomic_number:
            case 6:
                atom_type_index = 8
            case 7:
                atom_type_index = 12
            case 8:
                atom_type_index = 17
            case 16:
                atom_type_index = 32
            case _:
                raise SystemExit(f"No atom type for aromatic atom with atomic number {atomic_number}")
        
        self.m_atom_types[node] = atom_type_index

    def get_unsaturated_degree(self, node: int) -> int:
        neighbours = self.m_adjacency[node][0]

        unsaturated_degree: int = 0
        for neighbour in neighbours:
            coordination_number: int = self.m_coordination_number[neighbour]
            if (coordination_number < 4) and (coordination_number > 1):
                unsaturated_degree += 1
        return unsaturated_degree

    def setup_hyper_donor_acceptors(self, da_count: int) -> None:
        self.m_da_count = da_count
        self.m_da_array = [hyperconjugated.hyper_da() for _ in range(da_count)]

    def determine_reference_volume(self) -> None:
        element_counter: Dict[int, int] = Counter(self.m_elements)
        element_volume: Dict[int, float] = {element: 0.0 for element in element_counter}
        element_edge_counter: Dict[int, int] = {element: 0 for element in element_counter}

        for edge in self.m_edges:
            node1: int = edge[0]
            node2: int = edge[1]

            atomic_no1: int = self.m_elements[node1]
            atomic_no2: int = self.m_elements[node2]

            r2: float = self.distance2(node1, node2)
            radius1: float = atomic_data.get_radii(atomic_no1)
            radius2: float = atomic_data.get_radii(atomic_no2)
            alpha1: float = math.pi * ((3 * 2 * math.sqrt(2))/(4 * math.pi * radius1*radius1*radius1)) ** (2.0/3.0)
            alpha2: float = math.pi * ((3 * 2 * math.sqrt(2))/(4 * math.pi * radius2*radius2*radius2)) ** (2.0/3.0)

            # volume overlap
            vij: float = 8 * math.exp(-1 * (alpha1 * alpha2 * r2)/(alpha1 + alpha2)) * ((math.pi/(alpha1 + alpha2)) ** (3.0/2.0))

            element_volume[atomic_no1] += vij
            element_volume[atomic_no2] += vij

            element_edge_counter[atomic_no1] += 1
            element_edge_counter[atomic_no2] += 1
        
        for element in element_volume:
            sigma: float = atomic_data.get_radii(element)
            include_vol: float = element_counter[element]/self.m_natoms * (4.0/3.0 * math.pi * sigma * sigma * sigma - element_volume[element]/element_edge_counter[element])
            self.m_ref_vol_atom += include_vol

    def pair_volume(self, node1: int, node2: int) -> float:
        atomic_no1: int = self.m_elements[node1]
        atomic_no2: int = self.m_elements[node2]

        dist2: float = self.distance2(node1, node2)
        r1: float = atomic_data.get_radii(atomic_no1)
        r2: float = atomic_data.get_radii(atomic_no2)

        alpha1: float = math.pi * ((3 * 2 * math.sqrt(2))/(4 * math.pi * r1*r1*r1))**(2.0/3.0)
        alpha2: float = math.pi * ((3 * 2 * math.sqrt(2))/(4 * math.pi * r2*r2*r2))**(2.0/3.0)
        
        vij: float = 8 * math.exp(-1 * (alpha1 * alpha2 * dist2)/(alpha1 + alpha2)) * (math.pi/(alpha1 + alpha2))**(3.0/2.0)

        return vij

    def is_one_two_bonds_apart(self, node1: int, node2: int) -> bool:
        atom_pair_idx: int
        if node1 < node2:
            atom_pair_idx = int(node2 * (node2 - 1) / 2 + node1)
        else:
            atom_pair_idx = int(node1 * (node1 -1) / 2 + node2)

        return self.m_atom_pair_connections[atom_pair_idx] == 1