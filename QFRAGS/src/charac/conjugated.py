from typing import List, Tuple, Set

# conjugated system
class conjsys: 
    def __init__(self) -> None:
        self.m_natoms: int
        self.m_nedges: int
        self.m_nodes: List[int] = []
        self.m_edges_hash_values: List[int] = []

    def set_natoms(self, natoms: int) -> None:
        self.m_natoms = natoms
    
    def set_nedges(self, nedges: int) -> None:
        self.m_edges = nedges