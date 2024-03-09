import os
import sys
from typing import List, Set

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
sys.path.append(charac_dir)
sys.path.append(current_dir)

import mgraph

class Molecule:
    def __init__(self, monid: List[int] = None, monomer1 = None, monomer2 = None, 
                 fragid: List[int] = None, graph: mgraph.mgraph = None) -> None: 
        # acts as a constructor for two cases: monomer and dimer
        # monomer1 and monomer2 are Molecule objects

        self.m_natoms: int
        self.m_natoms_nohcap: int
        self.m_energy: float = 0.0
        self.m_joined_dimer: bool = False
        self.m_atom_ids: List[int] = []
        self.m_mon1_hcaps: Set[int] = set()
        self.m_mon2_hcaps: Set[int] = set()

        if len(monid) == 1:
            # monomer case:
            self.m_monomers = 1
            self.m_id: List[int] = [monid[0]]
            
        elif len(monid) == 2:
            # dimer case
            self.m_monomers = 2
            self.m_id: List[int] = [monid[0], monid[1]]

            global_natoms: int = graph.m_natoms
            node_sg_nidx: List[int] = graph.m_node_sg_nidx
            self.m_natoms = monomer1.m_natoms + monomer2.m_natoms

            # adding atoms that are not hydrogen caps
            for iatom in range(0, monomer1.m_natoms_nohcap):
                atom = monomer1.m_atom_ids[iatom]
                self.m_atom_ids.append(atom)
            
            for iatom in range(0, monomer2.m_natoms_nohcap):
                atom = monomer2.m_atom_ids[iatom]
                self.m_atom_ids.append(atom)

            for ihcap in range(monomer1.m_natoms_nohcap, monomer1.m_natoms):
                hcap: int = monomer1.m_atom_ids[ihcap]
                hcap_idx_sg: int = node_sg_nidx[hcap - global_natoms]

                if fragid[hcap_idx_sg] - 1 != monid[1]:
                    self.m_atom_ids.append(hcap)
                else:
                    self.m_mon1_hcaps.add(hcap)
            
            for ihcap in range(monomer2.m_natoms_nohcap, monomer2.m_natoms):
                hcap: int = monomer2.m_atom_ids[ihcap]
                hcap_idx_sg: int = node_sg_nidx[hcap - global_natoms]

                if fragid[hcap_idx_sg] - 1 != monid[0]:
                    self.m_atom_ids.append(hcap)
                else:
                    self.m_mon2_hcaps.add(hcap)
            

            if len(self.m_mon1_hcaps) > 0 or len(self.m_mon2_hcaps) > 0:
                self.m_joined_dimer = True
            
            self.m_natoms = len(self.m_atom_ids)
    
    def set_natoms_nohcap(self) -> None:
        self.m_natoms_nohcap = len(self.m_atom_ids)
    
    def set_natoms(self) -> None:
        self.m_natoms = len(self.m_atom_ids)
    
    def atom_ids(self) -> List[int]:
        return self.m_atom_ids

    def atom_ids(self, iatom: int) -> int:
        return self.m_atom_ids[iatom]

    def is_mon_hcap(self, atom_idx: int, mon_id: int) -> bool:
        match mon_id:
            case 0:
                return atom_idx in self.m_mon1_hcaps
            case 1:
                return atom_idx in self.m_mon2_hcaps
    
    def get_mon_hcaps(self, mon_id: int) -> Set[int]:
        match mon_id:
            case 0:
                return self.m_mon1_hcaps
            case 1:
                return self.m_mon2_hcaps
            