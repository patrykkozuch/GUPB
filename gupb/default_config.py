from gupb.controller import random
from gupb.controller.bupg import bupg


minion = bupg.BUPGController("Minion")

CONFIGURATION = {
    'arenas': [
        'ordinary_chaos'
    ],
    'controllers': [
        minion,
        random.RandomController("Alice"),
        random.RandomController("Bob"),
        random.RandomController("Cecilia"),
        random.RandomController("Darius"),
    ],
    'start_balancing': False,
    'visualise': True,
    'show_sight': minion,
    'runs_no': 10,
    'profiling_metrics': [],
}
