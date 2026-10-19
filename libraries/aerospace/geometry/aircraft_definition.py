from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Literal, Tuple, Union


@dataclass
class WingXsec:
    span: float
    root_chord: float
    tip_chord: float
    twist_deg: float
    twist_loc: float
    sweep_deg: float
    dihedral_deg: float

@dataclass
class Airfoil:
    thick_chord:float
    camber:float
    camber_loc: float


@dataclass
class Wing:
    name: str
    wing_sections: List[WingXsec]
    airfoils: List[Airfoil]
    x_loc: float
    y_loc: float
    z_loc: float
    x_rot: float
    y_rot: float
    z_rot: float

@dataclass
class Aircraft:
    name: str
    wing_list: List[Wing]
