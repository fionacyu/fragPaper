from typing import List

class hyper_da: 
    # donor and acceptor
    def __init__(self) -> None:
        self.m_natoms: int 
        self.m_nodes: List[int] = []
        self.m_node_electrons: List[int] = []
        self.m_terminal_nodes: List[int] = []
        self.m_nterminal_nodes: int
        self.m_classification: int
    
    def set_natoms(self, natoms: int) -> None:
        self.m_natoms = natoms
        self.m_nodes = [0 for _ in range(natoms)]
        self.m_node_electrons = [0 for _ in range(natoms)]
     
    def set_classification(self, classification: int) -> None:
        self.m_classification = classification

class hypersys:
    def __init__(self, value = None) -> None:

        if not value:
            self.m_donor: int
            self.m_acceptor: int
            self.m_connection_path: List[int] = []
            self.m_separation: int
        else:
            self.m_donor: int = value.m_donor
            self.m_acceptor: int = value.m_acceptor
            self.m_connection_path: List[int] = value.m_connection_path.copy()
            self.m_separation: int = value.m_separation
