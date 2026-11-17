"""Search productions derived from the operator registry's single authority."""
from foundry.ops.registry import COST, OPS, SEARCH_CONSTANTS, SEARCH_PRODUCTIONS, WINDOW_LADDER, Slot


def grammar(fields: list[str], window_caps: tuple[int, int]) -> str:
    """window_caps = (largest time-series window, largest pair window) the panel depth allows."""
    rows = [f'{name} {int(Slot.PANEL)} {COST["operand"]} 0' for name in sorted(fields)]
    rows += [f'{p.operator} {int(Slot.PANEL)} {COST[OPS[p.operator].kind]} {len(p.slots)} '
             + ' '.join(str(int(slot)) for slot in p.slots) for p in SEARCH_PRODUCTIONS]
    # A window is an attribute of its operator's node, so it adds no cost of its own.
    rows += [f'{w} {int(Slot.WINDOW)} 0 0' for w in WINDOW_LADDER if w <= window_caps[0]]
    rows += [f'{w} {int(Slot.PAIR_WINDOW)} 0 0' for w in WINDOW_LADDER if w <= window_caps[1]]
    rows += [f'{k} {int(Slot.CONST)} {COST["const_add"]} 0' for k in SEARCH_CONSTANTS]
    return '\n'.join(rows)
