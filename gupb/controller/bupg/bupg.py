import random

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from gupb import controller
from gupb.controller.bupg.knowledge.map import MapKnowledge
from gupb.controller.bupg.utils import position_change_to_move
from gupb.model import arenas
from gupb.model import characters
from gupb.model.arenas import Arena
from gupb.model.characters import Facing, Action
from gupb.model.coordinates import Coords

POSSIBLE_ACTIONS = [
    characters.Action.TURN_LEFT,
    characters.Action.TURN_RIGHT,
    characters.Action.STEP_FORWARD,
    characters.Action.ATTACK,
]


# noinspection PyUnusedLocal
# noinspection PyMethodMayBeStatic
class BUPGController(controller.Controller):
    def __init__(self, first_name: str):
        self.first_name: str = first_name
        self.map_knowledge = None
        self.grid = None
        self.weapon = None
        self.health = None
        self.facing = None
        self.pathfinder = AStarFinder()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BUPGController):
            return self.first_name == other.first_name
        return False

    def __hash__(self) -> int:
        return hash(self.first_name)

    def update_knowledge(self, knowledge: characters.ChampionKnowledge):
        self.map_knowledge.update_terrain(knowledge)

        my = knowledge.visible_tiles[knowledge.position].character
        self.weapon = my.weapon
        self.health = my.health
        self.facing = my.facing

    def decide(self, knowledge: characters.ChampionKnowledge) -> characters.Action:
        self.update_knowledge(knowledge)

        if action := self.go(knowledge.position, Coords(4, 7), self.facing):
            return action

        # Just Dance
        return characters.Action.TURN_LEFT if random.random() > 0.5 else characters.Action.TURN_RIGHT

    def go(self, start: Coords, end: Coords, facing: Facing) -> Action | None:
        """
        Args:
            start (Coords): The current position (x, y)
            end (Coords): The target position (x, y)
            facing (Facing): The current facing direction of the champion.
        """
        self.grid.cleanup()

        start = self.grid.node(*start)
        end = self.grid.node(*end)

        path, runs = self.pathfinder.find_path(start, end, self.grid)

        if len(path) > 1:
            return position_change_to_move(
                (path[1].y, path[1].x),
                (start.y, start.x),
                facing
            )

    def praise(self, score: int) -> None:
        pass

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        self.map_knowledge = MapKnowledge(terrain=Arena.load(arena_description.name).terrain)
        self.create_grid()

    def create_grid(self):
        W = max(self.map_knowledge.terrain, key=lambda x: x[0])[0] + 1
        H = max(self.map_knowledge.terrain, key=lambda x: x[1])[1] + 1
        self.grid = np.zeros(shape=(H, W))

        for (x, y), tile in self.map_knowledge.terrain.items():
            if tile.terrain_passable():
                self.grid[y, x] = 1

        self.grid = Grid(matrix=self.grid)

    @property
    def name(self) -> str:
        return f'BUPG {self.first_name}'

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.MINION


POTENTIAL_CONTROLLERS = [
    BUPGController("Minion")
]
