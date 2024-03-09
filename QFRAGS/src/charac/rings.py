from typing import List, Tuple, Set

class cycle:
    def __init__(self) -> None:
        self.m_nedges: int
        self.m_edges: List[int] = [] # length is number of edges * 2 bc each edge contains 2 atoms
        self.m_boxes: List[int] = [] # contains box id the cycle is part of
        self.m_nboxes: int

    def set_edge_count(self, edge_count: int) -> None:
        self.m_nedges = edge_count
        self.m_edges = [0 for _ in range(2 * edge_count)]

    def set_box_no(self, box_no: int) -> None:
        self.m_nboxes = box_no