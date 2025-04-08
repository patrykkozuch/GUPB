import random

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from gupb import controller
from gupb.controller.bupg.knowledge.map import MapKnowledge
from gupb.model import arenas
from gupb.model import characters
from gupb.model.arenas import Arena
from gupb.model.characters import Facing

POSSIBLE_ACTIONS = [
    characters.Action.TURN_LEFT,
    characters.Action.TURN_RIGHT,
    characters.Action.STEP_FORWARD,
    characters.Action.ATTACK,
]


def position_change_to_move(current_pos: tuple, new_pos: tuple, facing: Facing):
    move = np.array(current_pos) - np.array(new_pos)
    print(current_pos, new_pos)
    def rotate_left(v):
        return -v[1], v[0]

    def rotate_right(v):
        return v[1], -v[0]

    def rotate_180(v):
        return -v[0], -v[1]

    if facing == Facing.RIGHT:
        move = rotate_left(move)
    elif facing == Facing.DOWN:
        move = rotate_180(move)
    elif facing == Facing.LEFT:
        move = rotate_right(move)

    print(move)

    match move:
        case (0, 0):
            return characters.Action.DO_NOTHING
        case (-1, 0):
            return characters.Action.STEP_LEFT
        case (1, 0):
            return characters.Action.STEP_RIGHT
        case (0, -1):
            return characters.Action.STEP_BACKWARD
        case (0, 1):
            return characters.Action.STEP_FORWARD

    print("???")

# noinspection PyUnusedLocal
# noinspection PyMethodMayBeStatic
class BUPGController(controller.Controller):
    def __init__(self, first_name: str):
        self.first_name: str = first_name
        self.map_knowledge = MapKnowledge()
        self.grid = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BUPGController):
            return self.first_name == other.first_name
        return False

    def __hash__(self) -> int:
        return hash(self.first_name)

    def decide(self, knowledge: characters.ChampionKnowledge) -> characters.Action:

        mist_tiles = {
            coords: tile
            for coords, tile in knowledge.visible_tiles.items()
            if "mist" in [eff.type for eff in tile.effects]
        }

        # if len(mist_tiles) > 3:
        #     t_1, t_2, t_3 = list(mist_tiles)[:3]
        #     self.map_knowledge.menhir_location = Coords(*circle_from_points(t_1, t_2, t_3))

        own_x, own_y = knowledge.position

        start = self.grid.node(own_x, own_y)
        end = self.grid.node(5, 5)

        finder = AStarFinder()
        path, runs = finder.find_path(start, end, self.grid)

        if path:
            new_pos = (path[1].x, path[1].y)
            action=  position_change_to_move(knowledge.position, new_pos, knowledge.visible_tiles[knowledge.position].character.facing)
            print("ACTION: ", action)
            return action
        print(knowledge.position)
        print("DO NOTHING")

        return characters.Action.DO_NOTHING

    def praise(self, score: int) -> None:
        pass

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        self.map_knowledge.terrain = Arena.load(arena_description.name).terrain

        W = max(self.map_knowledge.terrain, key=lambda x: x[0])[0]
        H = max(self.map_knowledge.terrain, key=lambda x: x[1])[1]

        self.grid = np.zeros(shape=(W, H))

        for (x, y), tile in self.map_knowledge.terrain.items():
            if tile.terrain_passable():
                self.grid[x, y] = 1

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
