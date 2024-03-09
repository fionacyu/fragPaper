import sys
import os
import json
from typing import List

current_dir = os.path.dirname(os.path.realpath(__file__))
charac_dir = os.path.join(current_dir, '..', 'charac')
sys.path.append(charac_dir)
sys.path.append(current_dir)

from mgraph import mgraph
import atomic_data


def read_input(graph: mgraph, input_file: str) -> None:
    with open(input_file, "r") as f:
        contents = json.load(f)
    f.close()

    graph.m_target_frag_size = float(contents["frag_size"])
    symbols: List[str] = contents["symbols"]
    coordinates: List[float] = contents["geometry"]
    charges: List[int] = contents["atom_charges"]

    graph.set_natoms(len(symbols))

    graph.m_coordinates = coordinates
    graph.m_charges = charges    
    graph.m_name = input_file.split(".json")[0]

    x_values: List[float] = []
    y_values: List[float] = []
    z_values: List[float] = []
    for iatom in range(0, graph.m_natoms):
        symbol: str = symbols[iatom]
        atomic_number: int = atomic_data.atomic_number_map[symbol]

        x_values.append(coordinates[3 * iatom])
        y_values.append(coordinates[3 * iatom + 1])
        z_values.append(coordinates[3 * iatom + 2])

        graph.m_elements.append(atomic_number)

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    z_min = min(z_values)
    z_max = max(z_values)
    
    graph.set_range(x_min, x_max, y_min, y_max, z_min, z_max)
    
    