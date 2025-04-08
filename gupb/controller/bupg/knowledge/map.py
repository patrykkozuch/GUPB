import dataclasses
from typing import Optional

from gupb.model.arenas import Terrain


@dataclasses.dataclass
class MapKnowledge:
    terrain: Optional[Terrain] = None
    menhir_location = None
    