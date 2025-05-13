import numpy as np

from gupb.controller import keyboard
from gupb.controller import random
from gupb.controller.camperbot import camperbot
from gupb.controller.neat import kim_dzong_neat_jr
from gupb.controller import kirby_learning
from gupb.controller import norgul
from gupb.controller import reinforced_rogue
from gupb.controller import garek
from gupb.controller import rustler
from gupb.controller.bupg import bupg
from gupb.controller.pirat import pirat
from gupb.controller import roomba
from gupb.controller import Keramzytowy_mocarz

keyboard_controller = keyboard.KeyboardController()
bupg_controller = bupg.BUPGController("BUPG")

arenas = [
    # 'archipelago',
    # 'dungeon',
    # 'fisher_island',
    # 'isolated_shrine',
    # 'lone_sanctum',
    # 'mini',
    'ordinary_chaos',
    # 'wasteland'
]

np.random.shuffle(arenas)

CONFIGURATION = {
    'arenas': arenas,
    'controllers': [
        bupg_controller,
        rustler.Rustler("Rustler A"),
        garek.GarekController("Garek A"),
        norgul.NorgulController("Norgul A"),
        reinforced_rogue.ReinforcedRogueController("Rogue")
    ],
    'start_balancing': False,
    'visualise': True,
    'show_sight': bupg_controller,
    'runs_no': 10000,
    'profiling_metrics': [],
}
