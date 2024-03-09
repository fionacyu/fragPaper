import os
import sys
import math
from typing import List, Tuple, Set, Dict

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
penalty_dir = os.path.join(current_dir, '..', 'penalties')

sys.path.append(current_dir)
sys.path.append(charac_dir)
sys.path.append(penalty_dir)

import mgraph
import param
import molecule

def eg_vdw(iatom: int, jatom: int, graph: mgraph.mgraph, istatus: bool = False, jstatus: bool = False) -> float:
    atom_types: List[int] = graph.m_atom_types
    
    iatom_type: int = 0 if istatus else atom_types[iatom]
    jatom_type: int = 0 if jstatus else atom_types[jatom]
    Ri: float = param.x1_array[iatom_type]
    Rj: float = param.x1_array[jatom_type]

    ki: float = param.D1sqrt_array[iatom_type]
    kj: float = param.D1sqrt_array[jatom_type]

    kij: float = param.KCAL_TO_KJ * ki * kj
    kaSquared: float = Ri * Rj
    rijSquared: float = graph.distance2(iatom, jatom)

    term6: float = kaSquared / rijSquared
    term6squared: float = term6 * term6 
    term6_final: float = term6squared * term6

    term12: float = term6_final * term6_final
    energy = kij * ((term12) - (2.0 * term6_final))
    return energy


def calculate_evdw_sg(graph: mgraph.mgraph, subgraph: mgraph.subgraph = None) -> None:
    if not subgraph:
        subgraph: mgraph.subgraph = graph.m_subgraphs[0]
    natoms: int = subgraph.m_natoms
    natom_pairs: int = int((natoms * (natoms - 1))/ 2)

    evdw_total: float = 0.0
    for k in range(0, natom_pairs):
        jatom: int = int(natoms - 2 - math.floor(math.sqrt(-8*k + 4*natoms*(natoms-1)-7)/2.0 - 0.5))
        iatom: int = int(k + jatom + 1 - natoms*(natoms-1)/2 + (natoms-jatom)*((natoms-jatom)-1)/2)

        global_jatom: int = subgraph.get_node(jatom)
        global_iatom: int = subgraph.get_node(iatom)

        if graph.distance2(global_iatom, global_jatom) > 144.0:
            continue
        if not graph.is_one_two_bonds_apart(global_iatom, global_jatom):
            

            evdw_total += eg_vdw(global_iatom, global_jatom, graph)

    subgraph.m_energy = evdw_total

def mbe1_diff(graph: mgraph.mgraph, monomer_list: List[molecule.Molecule], 
              dimer_list: List[molecule.Molecule], dimer_energies: List[float]) -> float:
    global_natoms: int = graph.m_natoms

    delta_eij: float = 0.0
    # interaction between atoms lying on separate monomers
    for idimer, dimer in enumerate(dimer_list):
        mon1: int = dimer.m_id[0]
        mon2: int = dimer.m_id[1]

        mon1_natoms: int = monomer_list[mon1].m_natoms
        mon2_natoms: int = monomer_list[mon2].m_natoms

        disjoint_dimer: bool = not dimer.m_joined_dimer

        local_evdw: float = 0.0
        for iatom in range(0, mon1_natoms):
            global_atom_idx: int = monomer_list[mon1].atom_ids(iatom)
            istatus: bool = False # whether it's a hcap or not
            skip_iatom: bool = False

            if (global_atom_idx >= global_natoms):
                istatus = True
                if dimer.is_mon_hcap(global_atom_idx, 0):
                    skip_iatom = True
                global_atom_idx -= global_natoms
            
            if skip_iatom:
                continue

            for jatom in range(0, mon2_natoms):
                global_atom_jdx: int = monomer_list[mon2].atom_ids(jatom)
                jstatus: bool = False
                skip_jatom: bool = False

                if (global_atom_jdx >= global_natoms):
                    jstatus = True
                    if dimer.is_mon_hcap(global_atom_jdx, 1):
                        skip_jatom = True
                    global_atom_jdx -= global_natoms

                if skip_jatom:
                    continue

                if (global_atom_idx == global_atom_jdx):
                    continue

                r2: float = graph.distance2(global_atom_idx, global_atom_jdx)
                if (r2 > 144.0):
                    continue
                if (disjoint_dimer):
                    local_evdw += eg_vdw(global_atom_idx, global_atom_jdx, graph, istatus, jstatus)
                else:
                    if not graph.is_one_two_bonds_apart(global_atom_idx, global_atom_jdx):
                        local_evdw += eg_vdw(global_atom_idx, global_atom_jdx, graph, istatus, jstatus)

        # need to subtract off the interaction of hydrogen caps (joining mon1 and mon2) to atoms in their respect monomer
        mon1_hcaps: Set[int] = dimer.get_mon_hcaps(0)
        mon2_hcaps: Set[int] = dimer.get_mon_hcaps(1)

        hcap_atom_evdw: float = 0.0
        for hcap in mon1_hcaps:
            for iatom in range(0, mon1_natoms):
                global_atom_idx: int = monomer_list[mon1].atom_ids(iatom)

                if (hcap == global_atom_idx):
                    continue
                
                istatus: bool = False
                if (global_atom_idx >= global_natoms):
                    istatus = True
                    global_atom_idx -= global_natoms
                
                r2: float = graph.distance2(global_atom_idx, hcap - global_natoms)
                if (r2 > 144.0):
                    continue
                if not graph.is_one_two_bonds_apart(global_atom_idx, hcap - global_natoms):
                    hcap_atom_evdw += eg_vdw(global_atom_idx, hcap - global_natoms, graph, istatus, True)
        
        for hcap in mon2_hcaps:
            for iatom in range(0, mon2_natoms):
                global_atom_idx: int = monomer_list[mon2].atom_ids(iatom)

                if (hcap == global_atom_idx):
                    continue
                
                istatus: bool = False
                if (global_atom_idx >= global_natoms):
                    istatus = True
                    global_atom_idx -= global_natoms

                r2: float = graph.distance2(global_atom_idx, hcap - global_natoms)
                if (r2 > 144.0):
                    continue
                if not graph.is_one_two_bonds_apart(global_atom_idx, hcap - global_natoms):
                    hcap_atom_evdw += eg_vdw(global_atom_idx, hcap - global_natoms, graph, istatus, True)
        
        if not disjoint_dimer:
            dimer_energies[idimer] = local_evdw - hcap_atom_evdw
        
        delta_eij += (local_evdw - hcap_atom_evdw)
    return delta_eij


