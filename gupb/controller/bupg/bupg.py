import random
import traceback
from threading import Event

import cv2
import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from scipy.ndimage import label, gaussian_filter

from gupb import controller
from gupb.controller.bupg.knowledge.map import MapKnowledge
from gupb.controller.bupg.strategies.find_menhir import MenhirEstimator
from gupb.controller.bupg.utils import position_change_to_move
from gupb.model import arenas
from gupb.model import characters
from gupb.model.characters import Facing, Action
from gupb.model.coordinates import Coords
from gupb.model.weapons import Axe, Bow, Sword, Knife, Scroll, Amulet, PropheticWeapon

POSSIBLE_ACTIONS = [
    characters.Action.TURN_LEFT,
    characters.Action.TURN_RIGHT,
    characters.Action.STEP_FORWARD,
    characters.Action.STEP_BACKWARD,
    characters.Action.STEP_LEFT,
    characters.Action.STEP_RIGHT,
    characters.Action.ATTACK,
]

FACINGS = [
    (0, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
]


# noinspection PyUnusedLocal
# noinspection PyMethodMayBeStatic
class BUPGController(controller.Controller):
    WEAPON_PRIORITY = ["knife", "axe", "sword", "bow_unloaded", "bow_loaded", "amulet", "scroll", "propheticweapon"]

    train_step = Event()
    env = None

    def __init__(self, first_name: str):
        self.first_name: str = first_name
        self.map_knowledge: MapKnowledge | None = None
        self.menhir_estimator = None
        self.grid = None
        self.weapon = None
        self.health = None
        self.facing = None
        self.pathfinder = AStarFinder()
        self.position = None
        self.tries = 0
        self.ticks = 0
        self.me = None
        # Notify the environment that the game has started
        self.env.attach_controller(self)
        self.died = False
        self.score = None
        self.champions_alive = None
        self.knowledge = None
        self.arena_size = None

    @classmethod
    def assign_env(cls, env):
        cls.env = env

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BUPGController):
            return self.first_name == other.first_name
        return False

    def __hash__(self) -> int:
        return hash(self.first_name)

    def estimate_menhir(self, knowledge: characters.ChampionKnowledge):
        menhir_estimate, weight = self.menhir_estimator.estimate_menhir(knowledge)

        if menhir_estimate is not None:
            self.map_knowledge.update_menhir_location(menhir_estimate, weight)
            print(f"Estimated menhir location: {self.map_knowledge.estimated_menhir_location}")

    def update_knowledge(self, knowledge: characters.ChampionKnowledge):
        self.map_knowledge.update_terrain(knowledge)
        self.knowledge = knowledge
        self.champions_alive = knowledge.no_of_champions_alive
        # self.map_knowledge.episode_tick()
        # self.estimate_menhir(knowledge)

        self.me = knowledge.visible_tiles[knowledge.position].character
        self.weapon = self.me.weapon
        self.health = self.me.health
        self.facing = self.me.facing

    def facing_enemy(self, knowledge: characters.ChampionKnowledge):
        tile_in_front = knowledge.visible_tiles[self.position + self.facing.value]
        return tile_in_front.character is not None and tile_in_front.character != self.me

    @staticmethod
    def weapon_class(weapon_name: str):
        if weapon_name == "axe":
            return Axe
        if weapon_name == "sword":
            return Sword
        if weapon_name == "bow_unloaded" or weapon_name == "bow_loaded":
            return Bow
        if weapon_name == "knife":
            return Knife
        if weapon_name == "scroll":
            return Scroll
        if weapon_name == "amulet":
            return Amulet
        if weapon_name == "propheticweapon":
            return PropheticWeapon
        return None

    @property
    def enemy_in_range(self):
        wpn_class = self.weapon_class(self.weapon.name)

        if wpn_class is None:
            return False

        coords = wpn_class.cut_positions(self.map_knowledge.terrain, self.position, self.facing)

        all_coords = set(coords) & set(self.knowledge.visible_tiles.keys())

        for coord in all_coords:
            tile = self.knowledge.visible_tiles[coord]
            if tile.character is not None and tile.character != self.me:
                return True

        return False

    def find_best_weapon(self):
        for weapon in self.WEAPON_PRIORITY:
            # we already have the best available weapon
            if weapon == self.weapon.name:
                return None

            weapon_coords = self.map_knowledge.find_closest_weapon(self.position, weapon)
            if weapon_coords is not None:
                return weapon_coords

    def decide(self, knowledge: characters.ChampionKnowledge) -> characters.Action:
        try:
            self.ticks += 1
            position_changed = knowledge.position != self.position

            self.update_knowledge(knowledge)
            self.position = knowledge.position

            if not self.env.game_started.is_set():
                self.env.game_started.set()
            #
            #     most_unknown_point = self.map_knowledge.get_most_unknown_point()
            #
            #     dist_to_potion, point = self.map_knowledge.distance_to_potion(self.position)
            #     if dist_to_potion <= 3:
            #         point_to_go = point
            #     else:
            #         if weapon_coords := self.find_best_weapon():
            #             point_to_go = weapon_coords
            #         elif self.map_knowledge.menhir_location:
            #             dist_to_mist = self.map_knowledge.distance_to_mist(self.position)
            #             tree_coord = self.map_knowledge.find_closest_tree(self.map_knowledge.menhir_location)
            #
            #             if dist_to_mist > 5 and tree_coord and abs(self.map_knowledge.menhir_location[0] - tree_coord[0]) + abs(self.map_knowledge.menhir_location[1] - tree_coord[1]) <= 8:
            #                 point_to_go = tree_coord
            #
            #                 if self.position == point_to_go:
            #                     if self.enemy_in_range(knowledge):
            #                         return characters.Action.ATTACK
            #             else:
            #                 point_to_go = self.map_knowledge.menhir_location
            #         else:
            #             point_to_go = most_unknown_point
            #
            #     if not position_changed and self.tries <= 1:
            #         self.tries += 1
            #         if action := self.go(knowledge.position, Coords(*point_to_go), self.facing):
            #             return action
            #
            #     if not position_changed and self.tries <= 3:
            #         if self.facing_enemy(knowledge):
            #             return characters.Action.ATTACK
            #         else:
            #             self.tries += 1
            #             return characters.Action.TURN_LEFT
            #
            #     self.tries = 0

            # Mark the current turn as finished
            self.env.turn_event.set()

            # Wait for the training step to finish
            self.train_step.wait()
            self.train_step.clear()

            return POSSIBLE_ACTIONS[self.env.action]
        except:
            print(traceback.print_exc())
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
                (start.y, start.x),
                (path[1].y, path[1].x),
                facing,
                "yx"
            )

    def praise(self, score: int) -> None:
        self.died = True
        self.score = score
        self.env.turn_event.set()
        self.train_step.wait()
        self.train_step.clear()
        print(score)

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        arena = arenas.Arena.load(arena_description.name)
        self.map_knowledge = MapKnowledge(terrain=arena.terrain)
        self.arena_size = arena.size
        self.menhir_estimator = MenhirEstimator(self.map_knowledge)
        self.ticks = 0
        self.create_grid()
        self.died = False

    def create_grid(self):
        W, H = self.arena_size

        self.grid = np.zeros(shape=(H + 1, W + 1))

        for (x, y), tile in self.map_knowledge.terrain.items():
            if tile.terrain_passable():
                self.grid[y, x] = 1

        def find_largest_blob(arr):
            # Label connected components (4-connectivity by default)
            labeled_array, num_features = label(arr)

            # Count sizes of all blobs (excluding background label 0)
            sizes = np.bincount(labeled_array.ravel())
            sizes[0] = 0  # ignore background

            # Get label of largest blob
            max_label = sizes.argmax()
            max_size = sizes[max_label]

            # Create a mask for the largest blob
            largest_blob = (labeled_array == max_label)

            return largest_blob.astype(np.uint8)

        self.grid = find_largest_blob(self.grid)

        self.map_knowledge.looked_at = self.grid
        self.map_knowledge.last_looked_at = np.zeros(shape=self.grid.shape)
        self.map_knowledge.initial_seen = np.sum(self.grid)
        self.map_knowledge.remove_unreachable_weapons()

        for (x, y), tile in self.map_knowledge.terrain.items():
            if tile.loot and self.grid[y, x] > 0:
                self.grid[y, x] = 1

        self.map_knowledge.walkable = np.copy(self.grid)
        self.grid = Grid(matrix=self.grid)

    @property
    def maps(self):
        MAP_SIZE = 11
        MAP_HALF = MAP_SIZE // 2

        opponents_map = np.zeros(shape=(2, MAP_SIZE, MAP_SIZE))
        terrain_map = np.zeros(shape=(1, MAP_SIZE, MAP_SIZE))
        weapons_map = np.zeros(shape=(1, MAP_SIZE, MAP_SIZE))
        effects_map = np.zeros(shape=(2, MAP_SIZE, MAP_SIZE))
        my_map = np.zeros(shape=(2, MAP_SIZE, MAP_SIZE))

        visible_coords, _ = zip(*self.knowledge.visible_tiles.items())
        range_coords = self.weapon_class(self.weapon.name).cut_positions(
            self.map_knowledge.terrain,
            self.position,
            self.facing
            )

        for x in range(self.position.x - MAP_HALF, self.position.x + MAP_HALF + 1):
            for y in range(self.position.y - MAP_HALF, self.position.y + MAP_HALF + 1):
                if (x, y) not in self.map_knowledge.terrain:
                    continue

                walkable_index = (0, y - (self.position.y - MAP_HALF), x - (self.position.x - MAP_HALF))

                if self.map_knowledge.terrain[x, y].description().type in ['land', 'forest']:
                    if self.map_knowledge.terrain[x, y].description().type == 'land':
                        terrain_map[walkable_index] = 0.5
                    else:
                        terrain_map[walkable_index] = 1.0
                else:
                    if self.map_knowledge.terrain[x, y].description().type == 'sea':
                        terrain_map[walkable_index] = -0.5
                    else:
                        terrain_map[walkable_index] = -1.0

                type_index = (0, y - (self.position.y - MAP_HALF), x - (self.position.x - MAP_HALF))

                if (x, y) in self.map_knowledge.weapons:
                    weapon = self.map_knowledge.weapons[x, y]
                    weapons_map[type_index] = (self.WEAPON_PRIORITY.index(weapon.name) + 1) / (
                            len(self.WEAPON_PRIORITY) + 1)

                consumable_index = (0, y - (self.position.y - MAP_HALF), x - (self.position.x - MAP_HALF))
                hazardous_index = (1, y - (self.position.y - MAP_HALF), x - (self.position.x - MAP_HALF))

                if (x, y) in self.map_knowledge.consumables:
                    effects_map[consumable_index] = 1.0

                if self.map_knowledge.mist[y, x] > 0.5 or (x, y) in self.map_knowledge.fires:
                    effects_map[hazardous_index] = 1.0

                seen_index = (0, y - (self.position.y - MAP_HALF), x - (self.position.x - MAP_HALF))
                my_weapon_index = (1, y - (self.position.y - MAP_HALF), x - (self.position.x - MAP_HALF))

                age = self.map_knowledge.timestamp - self.map_knowledge.last_looked_at[y, x] + 1
                my_map[seen_index] = np.exp(-age / 80)

                if (x, y) in range_coords:
                    wpn_dmg = (3 if self.weapon.name in ['bow_loaded', 'bow_unloaded', 'axe'] else 2) / 3
                    my_map[my_weapon_index] = wpn_dmg

        # Gaussian blur my_map
        gaussian_filter(my_map[1], sigma=1, output=my_map[1])

        for (x, y), op in self.map_knowledge.opponents.items():
            op_weapon = self.weapon_class(op.weapon.name)
            cut_positions = op_weapon.cut_positions(self.map_knowledge.terrain, Coords(x, y), op.facing)

            for (cut_x, cut_y) in cut_positions:
                if (cut_x, cut_y) not in self.map_knowledge.terrain:
                    continue

                idx_y = cut_y - (self.position.y - MAP_HALF)
                idx_x = cut_x - (self.position.x - MAP_HALF)
                if 0 <= idx_y < MAP_SIZE and 0 <= idx_x < MAP_SIZE:
                    wpn_dmg = (3 if op.weapon.name in ['bow_loaded', 'bow_unloaded', 'axe'] else 2) / 3
                    opponents_map[0, idx_y, idx_x] = wpn_dmg

            op_y = y - (self.position.y - MAP_HALF)
            op_x = x - (self.position.x - MAP_HALF)

            if 0 <= op_y < MAP_SIZE and 0 <= op_x < MAP_SIZE:
                opponents_map[1, op_y, op_x] = op.health / 20

        # Gaussian blur opponents_map
        gaussian_filter(opponents_map[0], sigma=1, output=opponents_map[0])
        gaussian_filter(opponents_map[1], sigma=1, output=opponents_map[1])

        menhir_path = np.zeros(shape=(1, MAP_SIZE, MAP_SIZE))

        if self.map_knowledge.menhir_location is not None:
            path = self.path_to_menhir()
            for (y, x) in path:
                m_y = y - (self.position.y - MAP_HALF)
                m_x = x - (self.position.x - MAP_HALF)
                if 0 <= m_y < MAP_SIZE and 0 <= m_x < MAP_SIZE:
                    menhir_path[0, m_y, m_x] = 1.0

        # Combine maps into one
        return np.concatenate(
            (opponents_map, terrain_map, menhir_path, weapons_map, effects_map, my_map), axis=0
        )

    def distance_to_menhir(self):
        if self.map_knowledge.menhir_location is None:
            return 10000

        self.grid.cleanup()

        start = self.grid.node(self.position[0], self.position[1])
        end = self.grid.node(self.map_knowledge.menhir_location[0], self.map_knowledge.menhir_location[1])

        path, runs = self.pathfinder.find_path(start, end, self.grid)

        return len(path) - 1 if len(path) > 1 else 10000

    def path_to_menhir(self):
        if self.map_knowledge.menhir_location is None:
            return []

        self.grid.cleanup()

        start = self.grid.node(self.position[0], self.position[1])
        end = self.grid.node(self.map_knowledge.menhir_location[0], self.map_knowledge.menhir_location[1])

        path, runs = self.pathfinder.find_path(start, end, self.grid)

        return path

    def is_in_tree(self):
        return self.map_knowledge.terrain[self.position].description().type == 'forest'

    @property
    def name(self) -> str:
        return f'BUPG {self.first_name}'

    @property
    def preferred_tabard(self) -> characters.Tabard:
        return characters.Tabard.MINION
