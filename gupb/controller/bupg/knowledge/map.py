from typing import Optional

import numpy as np

from gupb.model import characters
from gupb.model.arenas import Terrain, terrain_size
from gupb.model.coordinates import Coords
from gupb.model.effects import EffectDescription
from gupb.model.games import MIST_TTH_PER_CHAMPION


class MapKnowledge:
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.menhir_location: Optional[Coords] = None
        self.mnist_moved = False
        self.estimated_menhir_location: np.ndarray | None = None
        self._total_weight: float = 0
        self._mist_radius: int | None = None
        self._episode: int = 0

        # All these attributes MIGHT BE OUTDATED
        self.weapons = {
            coords: tile.loot.description()
            for coords, tile in self.terrain.items()
            if tile.loot
        }

        self.consumables = {
            coords: tile.consumable.description()
            for coords, tile in self.terrain.items()
            if tile.consumable
        }

        self.effects: dict[Coords, EffectDescription] = {}
        self.mist: set[Coords] = set()
        self.fires: set[Coords] = set()

    @property
    def mist_radius(self):
        if self._mist_radius is None:
            size = terrain_size(self.terrain)
            self._mist_radius = int(size[0] * 2 ** 0.5) + 1
        return self._mist_radius

    def update_terrain(self, knowledge: characters.ChampionKnowledge):
        for coords, tile in knowledge.visible_tiles.items():
            # Update weapons
            if coords in self.weapons and tile.loot is None:
                del self.weapons[coords]

            if tile.loot:
                self.weapons[coords] = tile.loot

            # Update consumables
            if coords in self.consumables and tile.consumable is None:
                del self.consumables[coords]

            if tile.consumable:
                del self.consumables[coords]

            # Update fires
            if "fire" in [ef for ef in tile.effects]:
                self.fires.add(coords)

            # Update mists
            if "mist" in [ef for ef in tile.effects]:
                self.mist.add(coords)

            if self.terrain[coords].description() == "menhir":
                self.menhir_location = coords

    def episode_tick(self):
        if self._mist_radius is None:
            size = terrain_size(self.terrain)
            self._mist_radius = int(size[0] * 2 ** 0.5) + 1

        self._episode += 1
        self.mnist_moved = self._episode % MIST_TTH_PER_CHAMPION == 0
        if self.mnist_moved:
            self._mist_radius -= 1

    def update_menhir_location(self, new_estimate: np.ndarray, weight: float = 1.0) -> None:
        if self.estimated_menhir_location is None:
            self.estimated_menhir_location = np.zeros((2,))
        self._total_weight += weight
        self.estimated_menhir_location += weight * (new_estimate - self.estimated_menhir_location) / self._total_weight
