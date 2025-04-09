from typing import Optional

from gupb.model import characters
from gupb.model.arenas import Terrain
from gupb.model.coordinates import Coords
from gupb.model.effects import EffectDescription


class MapKnowledge:
    def __init__(self, terrain: Terrain):
        self.terrain = terrain
        self.menhir_location: Optional[Coords] = None

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
