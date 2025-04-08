from gupb.controller.bupg.strategies.base import BaseStrategy
from gupb.model import characters


class FindMenhirStrategy(BaseStrategy):
    INITIAL_MOVES = [
        characters.Action.TURN_LEFT for _ in range(4)
    ]

    menhir_found = False

    def apply(self) -> characters.Action:
        q = self.INITIAL_MOVES

        if not self.menhir_found and q:
            return q.pop(0)

        return characters.Action.DO_NOTHING