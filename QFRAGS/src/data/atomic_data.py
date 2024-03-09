from typing import Dict, List

atomic_number_map: Dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "Br": 35,
    "I": 53
}

bo_counter_map: Dict[float, int] = {
    1.0: 0,
    1.5: 1,
    2.0: 2,
    3.0: 3
}

atomic_masses: List[float] = [1.008, 4.003, 6.941, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
                                 22.99, 24.31, 26.98, 28.09, 30.97, 32.06, 35.45, 39.95, 39.10, 40.08,
                                 44.96, 47.87, 50.94, 52.00, 54.94, 55.85, 58.93, 58.69, 63.55, 65.38,
                                 69.72, 72.63, 74.92, 78.96, 79.90, 83.80, 85.47, 87.62, 88.91, 91.22,
                                 92.91, 95.94, 98.00, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,
                                 121.76, 127.60, 126.90, 131.29]

atom_symbols: List[str] = ["H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
        "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I"]

def get_valence(atomic_no: int) -> int:
    match atomic_no:
        case 1:
            return 1
        case 5:
            return 3
        case 6:
            return 4
        case 7:
            return 5
        case 8:
            return 6
        case 9:
            return 7
        case 15:
            return 5
        case 16:
            return 6
        case 17:
            return 7
        case 35:
            return 7
        case 53:
            return 7
        case _:
            raise SystemExit(f"Valence info for {atomic_no} not present")

def get_radii(atomic_no: int) -> float:
    match atomic_no:
        case 1:
            return 1.20

        case 6:
            return 1.70
        
        case 7:
            return 1.55

        case 8:
            return 1.52
        
        case 9:
            return 1.47

        case 15:
            return 1.80

        case 16:
            return 1.80
        
        case 17:
            return 1.75
        
        case 35:
            return 1.85
        
        case 53:
            return 1.98
        
        case _:
            raise SystemExit(f"No radii info for atomic no {atomic_no}")