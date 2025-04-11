from gupb.controller import random
from gupb.controller.bupg import bupg
from gupb.controller.keyboard import KeyboardController

minion = bupg.BUPGController("Minion")
keyboard = KeyboardController()

CONFIGURATION = {
    'arenas': [
        'ordinary_chaos'
    ],
    'controllers': [
        minion,
        random.RandomController("Bob1"),
        random.RandomController("Bob2"),
        random.RandomController("Bob3"),
        random.RandomController("Darius"),
    ],
    'start_balancing': False,
    'visualise': False,
    'show_sight': False,
    'runs_no': 10,
    'profiling_metrics': [],
}
