"""Allocation semantics shared by formulas, arrays and portfolio weighting."""


def signal_to_book(values, universe, slots, allocation, xp):
    lower = -1. if allocation.allow_short else 0.
    return xp.where(universe & xp.isfinite(values), xp.clip(values, lower, 1.), 0.) / slots


def gross_exposure(book, allocation, xp):
    values = xp.abs(book) if allocation.allow_short else book
    return values.sum(-1, keepdims=True)


def cap_book(book, allocation, xp):
    gross = gross_exposure(book, allocation, xp)
    if allocation.max_gross == 1.:
        # Preserve the cash-book operation order and rounding.
        return book / xp.maximum(1, gross)
    return book / xp.maximum(1., gross / allocation.max_gross)
