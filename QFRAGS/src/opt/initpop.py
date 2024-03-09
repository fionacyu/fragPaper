import os
import sys
import math
import queue
import random
from typing import List, Tuple, Set, Dict

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
data_dir = os.path.join(current_dir, '..', 'data')
gtraverse_dir = os.path.join(current_dir, '..', 'gtraverse')
monsorter_dir = os.path.join(current_dir, '..', 'mon_sorter')

sys.path.append(current_dir)
sys.path.append(charac_dir)
sys.path.append(data_dir)
sys.path.append(gtraverse_dir)
sys.path.append(monsorter_dir)

import mgraph
import atomic_data
import gtraverse
import mon_sorter

def get_centre_of_mass(graph: mgraph.mgraph, subgraph: mgraph.subgraph) -> Tuple[int, int, int]:
    natoms: int = subgraph.m_natoms
    total_mass: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    mz: float = 0.0

    elements: List[int] = graph.m_elements
    coordinates: List[float] = graph.m_coordinates

    for iatom in range(0, natoms):
        atom_idx: int = subgraph.get_node(iatom)
        atomic_no: int = elements[atom_idx]

        mass = atomic_data.atomic_masses[atomic_no - 1]
        x: float = coordinates[3 * atom_idx]
        y: float = coordinates[3 * atom_idx + 1]
        z: float = coordinates[3 * atom_idx + 2]

        mx += (mass * x)
        my += (mass * y)
        mz += (mass * z)
        total_mass += mass
    
    return mx / total_mass, my / total_mass, mz / total_mass


def get_inertia_tensor(graph: mgraph.mgraph, subgraph: mgraph.subgraph) -> List[float]:
    tensor: List[float] = [0.0 for _ in range(9)]

    elements: List[int] = graph.m_elements
    coordinates: List[float] = graph.m_coordinates
    natoms: int = subgraph.m_natoms

    # centre of mass
    com_x: float
    com_y: float
    com_z: float
    com_x, com_y, com_z = get_centre_of_mass(graph, subgraph)

    for iatom in range(0, natoms):
        atom_idx: int = subgraph.get_node(iatom)
        atomic_no: int = elements[atom_idx]
        mass = atomic_data.atomic_masses[atomic_no - 1]
        x: float = coordinates[3 * atom_idx] - com_x
        y: float = coordinates[3 * atom_idx + 1] - com_y
        z: float = coordinates[3 * atom_idx + 2] - com_z

        tensor[0] += mass * (y*y + z*z)
        tensor[1] -= mass * x * y
        tensor[2] -= mass * x * z
        tensor[4] += mass * (x*x + z*z)
        tensor[5] -= mass * y * z
        tensor[8] += mass * (x*x + y*y)
    
    tensor[3] = tensor[1]
    tensor[6] = tensor[2]
    tensor[7] = tensor[5]
    
    return tensor

def get_eigenvectors(tensor: List[float]) -> List[float]:
    l: List[float] = [0.0 for _ in range(3)]
    m: List[float] = [0.0 for _ in range(3)]
    v1: List[float] = [0.0 for _ in range(3)]
    v2: List[float] = [0.0 for _ in range(3)]
    v3: List[float] = [0.0 for _ in range(3)]

    eigenvectors: List[float] = [0.0 for _ in range(9)]

    x1: float = tensor[0]*tensor[0] + tensor[4]*tensor[4] + tensor[8]*tensor[8] - tensor[0]*tensor[4] - tensor[0]*tensor[8] - tensor[4]*tensor[8] + 3*(tensor[1]*tensor[1] + tensor[2]*tensor[2] + tensor[5]*tensor[5])
    x2: float = -1*(2.0 * tensor[0] - tensor[4] - tensor[8]) * (2.0 * tensor[4] - tensor[0] - tensor[8]) * (2.0 * tensor[8] - tensor[0] - tensor[4]) + 9.0*(abs(tensor[1])*abs(tensor[1])*(2.0*tensor[8] - tensor[0] - tensor[4]) + abs(tensor[2])*abs(tensor[2])*(2.0*tensor[4] - tensor[0] - tensor[8]) + abs(tensor[5])*abs(tensor[5])*(2.0*tensor[0] - tensor[4] - tensor[8])) - 54.0*(tensor[1]*tensor[2]*tensor[5])

    phi: float
    if (x2 > 0):
        phi = math.atan(math.sqrt(4*x1*x1*x1 - x2*x2)/x2)
    elif (x2 == 0):
        phi = math.pi/2.0
    else:
        phi = math.atan(math.sqrt(4*x1*x1*x1 - x2*x2)/x2) + math.pi 

    l[0] = (tensor[0] + tensor[4] + tensor[8] - 2.0*math.sqrt(x1)*math.cos(phi/3))/3.0
    l[1] = (tensor[0] + tensor[4] + tensor[8] + 2.0*math.sqrt(x1)*math.cos((phi - math.pi)/3))/3.0
    l[2] = (tensor[0] + tensor[4] + tensor[8] + 2.0*math.sqrt(x1)*math.cos((phi + math.pi)/3))/3.0

    m[0] = (tensor[1]*(tensor[8] - l[0]) - tensor[5]*tensor[2])/(tensor[2]*(tensor[4] - l[0]) - tensor[1]*tensor[5])
    m[1] = (tensor[1]*(tensor[8] - l[1]) - tensor[5]*tensor[2])/(tensor[2]*(tensor[4] - l[1]) - tensor[1]*tensor[5])
    m[2] = (tensor[1]*(tensor[8] - l[2]) - tensor[5]*tensor[2])/(tensor[2]*(tensor[4] - l[2]) - tensor[1]*tensor[5])

    v1[0] = (l[0] - tensor[8] - tensor[5]*m[0])/tensor[2]
    v1[1] = m[0]
    v1[2] = 1.0

    v2[0] = (l[1] - tensor[8] - tensor[5]*m[1])/tensor[2]
    v2[1] = m[1]
    v2[2] = 1.0

    v3[0] = (l[2] - tensor[8] - tensor[5]*m[2])/tensor[2]
    v3[1] = m[2]
    v3[2] = 1.0

    v1_r: float = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2])
    v2_r: float = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])
    v3_r: float = math.sqrt(v3[0]*v3[0] + v3[1]*v3[1] + v3[2]*v3[2])

    eigenvectors[0] = v1[0]/v1_r
    eigenvectors[1] = v2[0]/v2_r
    eigenvectors[2] = v3[0]/v3_r

    eigenvectors[3] = v1[1]/v1_r
    eigenvectors[4] = v2[1]/v2_r
    eigenvectors[5] = v3[1]/v3_r

    eigenvectors[6] = v1[2]/v1_r
    eigenvectors[7] = v2[2]/v2_r
    eigenvectors[8] = v3[2]/v3_r

    return eigenvectors


def get_ref_points(graph: mgraph.mgraph, subgraph: mgraph.subgraph, target_size: int) -> Tuple[List[float], int]:
    inertia_tensor = get_inertia_tensor(graph, subgraph)
    vr = get_eigenvectors(inertia_tensor)

    coordinates: List[float] = graph.m_coordinates
    tcoordinates: List[float] = [0.0 for _ in range(len(coordinates))]
    natoms: int = subgraph.m_natoms

    tcoord_x: List[float] = []
    tcoord_y: List[float] = []
    tcoord_z: List[float] = []

    for iatom in range(0, natoms):
        atom_idx: int = subgraph.get_node(iatom)
        tcoordinates[3 * iatom] = vr[0] * coordinates[3 * atom_idx] + vr[1] * coordinates[3 * atom_idx + 1] + vr[2] * coordinates[3 * atom_idx +2]
        tcoordinates[3 * iatom + 1] = vr[3] * coordinates[3 * atom_idx] + vr[4] * coordinates[3 * atom_idx + 1] + vr[5] * coordinates[3 * atom_idx +2]
        tcoordinates[3 * iatom + 2] = vr[6] * coordinates[3 * atom_idx] + vr[7] * coordinates[3 * atom_idx + 1] + vr[8] * coordinates[3 * atom_idx +2]

        tcoord_x.append(tcoordinates[3 * iatom])
        tcoord_y.append(tcoordinates[3 * iatom + 1])
        tcoord_z.append(tcoordinates[3 * iatom + 2])

    x_min: float = min(tcoord_x)
    x_max: float = max(tcoord_x)
    y_min: float = min(tcoord_y)
    y_max: float = max(tcoord_y)
    z_min: float = min(tcoord_z)
    z_max: float = max(tcoord_z)

    xrange: float = x_max - x_min
    yrange: float = y_max - y_min
    zrange: float = z_max - z_min

    r_min: float = min([xrange, yrange, zrange])
    box_length: float = math.cbrt(graph.m_ref_vol_atom * target_size)
    divisor: float
    if (r_min >= box_length):
        divisor = box_length
    else:
        divisor = r_min
    
    n_min: int = math.ceil(r_min/divisor)

    n_i: int = n_min * math.ceil(xrange/r_min)
    n_j: int = n_min * math.ceil(yrange/r_min)
    n_k: int = n_min * math.ceil(zrange/r_min)

    if n_i <= 0:
        n_i = 1
    if n_j <= 0:
        n_j = 1
    if n_k <= 0:
        n_k = 1

    l_i: float = xrange/n_i
    l_j: float = yrange/n_j
    l_k: float = zrange/n_k


    reference_points = [0.0 for _ in range(n_i * n_j * n_k * 3)]
    for i in range(0, n_i):
        for j in range(0, n_j):
            for k in range(0, n_k):
                ref_tx: float = x_min + i * l_i  + 0.5*l_i
                ref_ty: float = y_min + j * l_j  + 0.5*l_j
                ref_tz: float = z_min + k * l_k  + 0.5*l_k

                ref_x: float = vr[0] * ref_tx + vr[3] * ref_ty + vr[6] * ref_tz
                ref_y: float = vr[1] * ref_tx + vr[4] * ref_ty + vr[7] * ref_tz
                ref_z: float = vr[2] * ref_tx + vr[5] * ref_ty + vr[8] * ref_tz

                idx: int = int(n_k * (n_j * i + j) + k)
                reference_points[3*idx] = ref_x
                reference_points[3*idx + 1] = ref_y
                reference_points[3*idx + 2] = ref_z

    return reference_points, n_i * n_j * n_k

def calc_mon_centroid(graph: mgraph.mgraph, subgraph: mgraph.subgraph, mon_centroids: List[float],
                      mon_atoms: List[List[int]], nfrags: int) -> None:
    coordinates: List[float] = graph.m_coordinates
    natoms: int = subgraph.m_natoms

    for ifrag in range(0, nfrags):
        x_tot: float = 0.0
        y_tot: float = 0.0
        z_tot: float = 0.0
        mon_natoms: int = len(mon_atoms[ifrag])
        for atom in mon_atoms[ifrag]:
            node: int = atom
            if node >= natoms:
                # hcap
                node -= natoms
            node_idx: int = subgraph.get_node(node)
            x_tot += coordinates[3 * node_idx]
            y_tot += coordinates[3 * node_idx + 1]
            z_tot += coordinates[3 * node_idx + 2]
        mon_centroids[3 * ifrag] = x_tot/mon_natoms
        mon_centroids[3 * ifrag + 1] = y_tot/mon_natoms
        mon_centroids[3 * ifrag + 2] = z_tot/mon_natoms

def mon_dist_ref_point(mon_centroids: List[float], nfrags: int, reference_point: List[float],
                       monomer_list: List[mon_sorter.monomer]) -> None:
    for ifrag in range(0, nfrags):
        delta_x: float = reference_point[0] - mon_centroids[3*ifrag]
        delta_y: float = reference_point[1] - mon_centroids[3*ifrag + 1]
        delta_z: float = reference_point[2] - mon_centroids[3*ifrag + 2]

        r2: float = delta_x*delta_x + delta_y*delta_y + delta_z*delta_z
        r: float = math.sqrt(r2)

        mon: mon_sorter.monomer = mon_sorter.monomer(r, ifrag)
        monomer_list.append(mon)

def get_dimer_broken_bonds(mon_atoms: List[List[int]], mon_hcaps: List[int],
                           fragid: List[int], mon1: int, mon2: int, natoms: int) -> List[int]:
    broken_bonds: List[int] = []

    mon1_hcaps: int = mon_hcaps[mon1]
    mon2_hcaps: int = mon_hcaps[mon2]

    for ihcap in range(len(mon_atoms[mon1]) - 1, len(mon_atoms[mon1]) - 1 - mon1_hcaps, -1):
        for jhcap in range(len(mon_atoms[mon2]) - 1, len(mon_atoms[mon2]) - 1 - mon2_hcaps, -1):
            hcap1: int = mon_atoms[mon1][ihcap]
            hcap2: int = mon_atoms[mon2][jhcap]

            fid1: int = fragid[hcap1 - natoms]
            fid2: int = fragid[hcap2 - natoms]

            if (fid1-1 == mon2 and fid2-1 == mon1):
                broken_bonds.append(hcap1)
                broken_bonds.append(hcap2)

    return broken_bonds

def remove_overlap_hcaps(broken_bonds: List[int], frag_atoms: Set[int]) -> None:
    for node in broken_bonds:
        frag_atoms.remove(node)

def build_fragment(seed_mon: int, mon_atoms: List[List[int]], mon_hcaps: List[int], mon_hcap_fids: List[List[int]],
                   mon_adjacency: List[Set[int]], colors: List[int], target_size: int, mon_fragid: List[int],
                   frag_idx: int, fragid: List[int], natoms: int) -> None:
    mon_fragid[seed_mon] = frag_idx
    if (len(mon_atoms[seed_mon])/target_size >= 0.9):
        colors[seed_mon] = 2
        return

    frag_atoms: Set[int] = set()
    for atom in mon_atoms[seed_mon]:
        frag_atoms.add(atom)
    
    bfs_queue: queue.Queue = queue.Queue()
    bfs_queue.put(seed_mon)
    while not bfs_queue.empty():
        
        mon_label: int = bfs_queue.get()
        neighbours: Set[int] = mon_adjacency[mon_label]
        for neigh_mon in neighbours:
            if colors[neigh_mon] == 0:
                dimer_broken_bonds: List[int] = get_dimer_broken_bonds(mon_atoms, mon_hcaps,
                                                                       fragid, mon_label, neigh_mon, natoms)
                colors[neigh_mon] = 1
                mon_fragid[neigh_mon] = frag_idx
                for atom in mon_atoms[neigh_mon]:
                    frag_atoms.add(atom)
                remove_overlap_hcaps(dimer_broken_bonds, frag_atoms)
                bfs_queue.put(neigh_mon)

                if len(frag_atoms)/target_size >= 0.9:
                    return
        colors[mon_label] = 2
    
def patch_fragments(fragid: List[int], mon_fragid: List[int], old_mon_atoms: List[List[int]], old_mon_hcaps: List[int],
                    old_mon_adjacency: List[Set[int]], new_nfrags: int, old_nfrags: int, target_fsize: int) -> None:
    mon_adjacency: List[Set[int]] = [set() for _ in range(new_nfrags)]
    new_mon_atoms: List[List[int]] = [[] for _ in range(new_nfrags)]
    new_mon_size: List[int] = [0 for _ in range(new_nfrags)]

    for imon in range(0, old_nfrags):
        mon1: int = mon_fragid[imon] - 1
        mon_neighbours: Set[int] = old_mon_adjacency[imon]

        for mon_neighbour in mon_neighbours:
            mon2: int = mon_fragid[mon_neighbour] - 1

            if mon1 != mon2:
                mon_adjacency[mon1].add(mon2)
                mon_adjacency[mon2].add(mon1)
        new_mon_size[mon1] += (len(old_mon_atoms[imon]) - old_mon_hcaps[imon])

        for iatom in range(0, len(old_mon_atoms[imon]) - old_mon_hcaps[imon]):
            new_mon_atoms[mon1].append(old_mon_atoms[imon][iatom])

    new_mon_fragids: List[int] = [0 for _ in range(new_nfrags)]
    colors: List[int] = [0 for _ in range(new_nfrags)]

    frag_idx: int = 0
    for ifrag in range(0, new_nfrags):
        if colors[ifrag] == 0:
            frag_idx += 1
            gtraverse.patch_initial_pop_bfs(mon_adjacency, ifrag, colors, frag_idx, new_mon_fragids,
                                            new_mon_size, target_fsize)
    
    for ifrag in range(0, new_nfrags):
        fid: int = new_mon_fragids[ifrag]
        for atom in new_mon_atoms[ifrag]:
            fragid[atom] = fid


def construct_fragments(sorted_mon_list: List[mon_sorter.monomer], mon_atoms: List[List[int]],
                        mon_hcaps: List[int], mon_hcap_fids: List[List[int]], mon_adjacency: List[Set[int]],
                        nfrags: int, fragid: List[int], natoms: int, target_size: int) -> None:
    
    colors: List = [0 for _ in range(nfrags)]
    mon_fragid: List[int] = [0 for _ in range(nfrags)]

    frag_idx: int = 0
    for ifrag in range(0, nfrags):
        mon_idx: int = sorted_mon_list[ifrag].idx
        if colors[mon_idx] == 0:
            frag_idx += 1
            build_fragment(mon_idx, mon_atoms, mon_hcaps, mon_hcap_fids, mon_adjacency, colors,
                           target_size, mon_fragid, frag_idx, fragid, natoms)
    
    # print(f"frag_idx {frag_idx}")
    # now need to patch fragments
    patch_fragments(fragid, mon_fragid, mon_atoms, mon_hcaps, mon_adjacency, frag_idx, nfrags, target_size)

def init_pop_solution(graph: mgraph.mgraph, feasible_edges: List[int], frag_id: List[int], natoms: int) -> List[int]:
    edges: List[Tuple[int, int]] = graph.m_edges
    solution: List[int] = [0 for _ in range(len(feasible_edges))]

    node_sg_nidx: List[int] = graph.m_node_sg_nidx
    atom_broken_bonds: List[List[int]] = [[] for _ in range(natoms)]

    for iedge in range(0, len(feasible_edges)):
        edge: Tuple[int, int] = edges[feasible_edges[iedge]]
        node1: int = edge[0]
        node2: int = edge[1]

        node1_idx: int = node_sg_nidx[node1]
        node2_idx: int = node_sg_nidx[node2]

        fid1: int = frag_id[node1_idx]
        fid2: int = frag_id[node2_idx]

        if fid1 != fid2:
            solution[iedge] = 1
            atom_broken_bonds[node1_idx].append(iedge)
            atom_broken_bonds[node2_idx].append(iedge)
    
    for iatom in range(0, natoms):
        no_broken_bonds: int = len(atom_broken_bonds[iatom])
        if no_broken_bonds > 1:
            keep_idx: int = random.randint(0, no_broken_bonds - 1)

            for ibond in range(0, no_broken_bonds):
                if ibond != keep_idx:
                    edge_idx: int = atom_broken_bonds[iatom][ibond]
                    solution[edge_idx] = 1
    return solution

def degenerate_solutions(ngenes: int, solution1: List[int], solution2: List[int]) -> bool:

    difference_solution: List[int] = [0 for _ in range(ngenes)]
    for igene in range(0, ngenes):
        difference_solution[igene] = abs(solution1[igene] - solution2[igene])
    
    sum_difference: int = sum(difference_solution)
    return sum_difference == 0


def get_initial_population(graph: mgraph.mgraph, subgraph: mgraph.subgraph, target_size: int) -> List[List[int]]:
    feasible_edges: List[int] = subgraph.m_feasible_edges
    edges: List[Tuple[int, int]] = graph.m_edges
    node_sg_nidx: List[int] = graph.m_node_sg_nidx
    natoms: int = subgraph.m_natoms

    # print(f"Number of feasible edges: {len(feasible_edges)}")
    print(f"Subgraph natoms: {subgraph.m_natoms}")

    subgraph_copy: mgraph.subgraph = mgraph.subgraph(subgraph)

    # first determine reference points
    reference_points: List[float]
    no_ref_points: int

    reference_points, no_ref_points = get_ref_points(graph, subgraph, target_size)

    # print(f"no_ref_points: {no_ref_points}")
    for iedge in feasible_edges:
        edge: Tuple[int, int] = edges[iedge]
        node1: int = edge[0]
        node2: int = edge[1]

        node1_idx: int = node_sg_nidx[node1]
        node2_idx: int = node_sg_nidx[node2]

        subgraph_copy.delete_edge(node1_idx, node2_idx)

    fragid: List[int] = [0 for _ in range(natoms)]
    nfrags: int = gtraverse.determine_fragid(fragid, subgraph_copy) 

    mon_atoms: List[List[int]] = [[] for _ in range(nfrags)]
    mon_adjacency: List[Set[int]] = [set() for _ in range(nfrags)]
    mon_hcaps_fragids: List[List[int]] = [[] for _ in range(nfrags)]
    mon_centroids: List[float] = [0.0 for _ in range(3 * nfrags)]
    mon_hcaps: List[int] = [0 for _ in range(nfrags)] # stores no of hcaps per monomer

    for iatom in range(0, natoms):
        fid: int = fragid[iatom]
        mon_atoms[fid - 1].append(iatom)

    for iedge in feasible_edges:
        edge: Tuple[int, int] = edges[iedge]
        node1: int = edge[0]
        node2: int = edge[1]

        node1_idx: int = node_sg_nidx[node1]
        node2_idx: int = node_sg_nidx[node2]

        fid1: int = fragid[node1_idx]
        fid2: int = fragid[node2_idx]

        if (fid1 != fid2):
            mon_adjacency[fid1 - 1].add(fid2 - 1)
            mon_adjacency[fid2 - 1].add(fid1 - 1)
            mon_atoms[fid1 - 1].append(node2_idx + natoms)
            mon_atoms[fid2 - 1].append(node1_idx + natoms)
            mon_hcaps[fid1 - 1] += 1
            mon_hcaps[fid2 - 1] += 1
            mon_hcaps_fragids[fid1 - 1].append(fid2 - 1)
            mon_hcaps_fragids[fid2 - 1].append(fid1 - 1)
    
    calc_mon_centroid(graph, subgraph, mon_centroids, mon_atoms, nfrags)

    initial_population: List[List[int]] = []
    for iref_point in range(0, no_ref_points):
        fragid_copy: List[int] = fragid.copy()

        reference_point: List[float] = [reference_points[3 * iref_point],
                                        reference_points[3 * iref_point + 1],
                                        reference_points[3 * iref_point + 2]]
        monomer_list: List[mon_sorter.monomer] = []
        mon_dist_ref_point(mon_centroids, nfrags, reference_point, monomer_list)
        monomer_list.sort(key=lambda x: x.distance)

        construct_fragments(monomer_list, mon_atoms, mon_hcaps, mon_hcaps_fragids, mon_adjacency,
                            nfrags, fragid_copy, natoms, target_size)

        initial_solution: List[int] = init_pop_solution(graph, feasible_edges, fragid_copy, natoms)
        initial_population.append(initial_solution)
    
    # print(f"feasible_edges: {feasible_edges}")
    # print(f"initial pop: ")
    # for blah in initial_population:
    #     print(blah)
    ngenes: int = len(feasible_edges)
    degenerate_solutions_idx: List[int] = [0 for _ in range(no_ref_points)]
    for isol in range(0, no_ref_points):
        for jsol in range(0, isol):
            degenerate_status: bool = degenerate_solutions(ngenes, initial_population[isol], initial_population[jsol])
            if degenerate_status:
                degenerate_solutions_idx[isol] = 1
                # get rid of ones with higher index
    
    final_initial_population: List[List[int]] = []

    for isol in range(0, no_ref_points):
        if degenerate_solutions_idx[isol] == 0:
            final_initial_population.append(initial_population[isol])

    return final_initial_population