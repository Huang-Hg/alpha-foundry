"""Shared exposure arithmetic for account execution and portfolio construction."""
from types import ModuleType

from foundry.types import Allocation, FloatArray


def gross_exposure(book: FloatArray, allocation: Allocation, xp: ModuleType) -> FloatArray:
    """Return (...,T,1) gross exposure from resident (...,T,S) books."""
    values = xp.abs(book) if allocation.allow_short else book
    return values.sum(-1, keepdims=True)


def cap_book(book: FloatArray, allocation: Allocation, xp: ModuleType) -> FloatArray:
    """Cap resident book exposure without changing cash-book operation order."""
    gross = gross_exposure(book, allocation, xp)
    if allocation.max_gross == 1.:
        # Preserve the cash-book operation order and rounding.
        return book / xp.maximum(1, gross)
    return book / xp.maximum(1., gross / allocation.max_gross)
