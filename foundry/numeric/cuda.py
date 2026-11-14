"""Paired return moments; resources belong to the prepared array backend."""
from pathlib import Path

import cupy as cp
import numpy as np

from foundry.contracts import ReturnMoments
from foundry.numeric.cuda_source import paired_source


class Moments:
    def __init__(self):
        self.module = cp.RawModule(code=paired_source() + '\n' +
            Path(__file__).with_name('moments.cu').read_text(encoding='utf-8'),
            options=('--std=c++17', '--fmad=false'))
        self.kernel = self.module.get_function('return_moments')

    def measure(self, returns, weights, ppy):
        shape = returns.shape[:-1]
        days = returns.shape[-1]
        rows = cp.ascontiguousarray(returns.reshape(-1, days))
        output = cp.empty((len(rows), 4), cp.float64)
        self.kernel((len(rows),), (128,),
            (rows, cp.ascontiguousarray(weights), np.int32(days), np.float64(ppy), output))
        return ReturnMoments(*(output[:, i].reshape(shape) for i in range(4)))

    def close(self):
        del self.kernel, self.module
