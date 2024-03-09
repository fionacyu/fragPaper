import math
from typing import List, Set

import mgraph

class mbox:
    def __init__(self, natoms):
        self.m_nodes: List[int] = []
        self.m_natoms: int = natoms
        self.m_cycles: Set[int] = set()
        self.m_conj_systems: Set[int] = set()
        self.m_hyper_donor_acceptors: Set[int] = set()
        self.m_hyper_systems: Set[int] = set()
    
    def add_conj_sys(self, conj_sys_idx: int):
        self.m_conj_systems.add(conj_sys_idx)

class mbox_array:
    def __init__(self):
        self.m_box_array: List[mbox] = []
        self.m_nboxes: int
        self.m_nx: int
        self.m_ny: int
        self.m_nz: int


    def define_boxes(self, graph: mgraph) -> None:
        box_len: float = 2.157 # angstroms, this is a parameter

        ranges: List[float] = graph.m_range
        x_range: float = ranges[1] - ranges[0]
        y_range: float = ranges[3] - ranges[2]
        z_range: float = ranges[5] - ranges[4]

        self.m_nx = math.ceil(x_range/box_len)
        self.m_ny = math.ceil(y_range/box_len)
        self.m_nz = math.ceil(z_range/box_len)

        self.m_nboxes = self.m_nx * self.m_ny * self.m_nz

        graph.m_nx = self.m_nx
        graph.m_ny = self.m_ny
        graph.m_nz = self.m_nz

        natoms: int = graph.m_natoms
        coordinates: List[float] = graph.m_coordinates

        xmin: float = ranges[0]
        ymin: float = ranges[2]
        zmin: float = ranges[4]

        node_box: List[int] = [0 for _ in range(natoms)] # stores the box id each node belongs to
        box_node_count: List[int] = [0 for _ in range(self.m_nboxes)] # stores no. of nodes in a box

        for iatom in range(0, natoms):
            x: float = coordinates[3 * iatom]
            y: float = coordinates[3 * iatom + 1]
            z: float = coordinates[3 * iatom + 2]

            bx: int = math.floor((x - xmin) / box_len)
            by: int = math.floor((y - ymin) / box_len)
            bz: int = math.floor((z - zmin) / box_len)

            box_idx: int = (bz * self.m_nx * self.m_ny) + (by * self.m_nx) + bx
            node_box[iatom] = box_idx
            box_node_count[box_idx] += 1

        for ibox in range(0, self.m_nboxes):
            self.m_box_array.append(mbox(box_node_count[ibox]))
        
        for iatom in range(0, natoms):
            box_id: int = node_box[iatom]
            self.m_box_array[box_id].m_nodes.append(iatom)

    def get_neighbours(self, box_id: int) -> List[int]:
        neighbours: List[int] = [] # max of 26 neighbours
        neighbours.append(box_id - self.m_nx *  self.m_ny -  self.m_nx - 1)
        neighbours.append(box_id - self.m_nx *  self.m_ny -  self.m_nx)
        neighbours.append(box_id - self.m_nx *  self.m_ny -  self.m_nx + 1)
        neighbours.append(box_id - self.m_nx *  self.m_ny - 1)
        neighbours.append(box_id - self.m_nx *  self.m_ny)
        neighbours.append(box_id - self.m_nx *  self.m_ny + 1)
        neighbours.append(box_id - self.m_nx *  self.m_ny +  self.m_nx - 1)
        neighbours.append(box_id - self.m_nx *  self.m_ny +  self.m_nx)
        neighbours.append(box_id - self.m_nx *  self.m_ny +  self.m_nx + 1)
        neighbours.append(box_id - self.m_nx - 1)
        neighbours.append(box_id - self.m_nx)
        neighbours.append(box_id - self.m_nx + 1)
        neighbours.append(box_id - 1)
        neighbours.append(box_id + 1)
        neighbours.append(box_id +  self.m_nx - 1)
        neighbours.append(box_id +  self.m_nx)
        neighbours.append(box_id +  self.m_nx + 1)
        neighbours.append(box_id +  self.m_nx *  self.m_ny -  self.m_nx - 1)
        neighbours.append(box_id +  self.m_nx *  self.m_ny -  self.m_nx)
        neighbours.append(box_id +  self.m_nx *  self.m_ny -  self.m_nx + 1)
        neighbours.append(box_id +  self.m_nx *  self.m_ny - 1)
        neighbours.append(box_id +  self.m_nx *  self.m_ny)
        neighbours.append(box_id +  self.m_nx *  self.m_ny + 1)
        neighbours.append(box_id +  self.m_nx *  self.m_ny +  self.m_nx - 1)
        neighbours.append(box_id +  self.m_nx *  self.m_ny +  self.m_nx)
        neighbours.append(box_id +  self.m_nx *  self.m_ny +  self.m_nx + 1)

        final_neighbours: List[int] = []
        for ineigh in range(0, 26):
            neighbour: int = neighbours[ineigh]
            if (neighbour >= 0 and neighbour < self.m_nboxes and neighbour != box_id):
                final_neighbours.append(neighbour)
        return final_neighbours

    def get_boxes(self) -> List[mbox]:
        return self.m_box_array