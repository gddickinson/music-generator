"""Scale-degree grids of the fractal melody generator, computed by the repo's own code.

FractalMelodyGenerator.generate_melody reads, for every Mandelbrot row, int(savgol(row) % 14) at every 4th column.
`python_degrees` reproduces exactly that with the generator's mandelbrot() and scipy; `exact_degrees` does the same
smoothing in exact integer arithmetic (what static/fractal_engine.js does). `quirks` lists the cells where SciPy's
floating-point result lands just below a whole number and so gives a different degree.
"""
from fractions import Fraction

import numpy as np
from scipy.signal import savgol_filter

from fractal_melody_generator import EmotionalStyle, FractalMelodyGenerator, MelodyParams, MusicalMode

RES, STEP, N_DEGREES = 200, 4, 14


def _divtime(emotion: str) -> np.ndarray:
    gen = FractalMelodyGenerator(MelodyParams(mode=MusicalMode.IONIAN, emotion=EmotionalStyle(emotion)))
    return gen.mandelbrot(RES, RES, 100)


def python_degrees(emotion: str):
    div = _divtime(emotion)
    return [[int(v) for v in (savgol_filter(div[i, :], 15, 3) % N_DEGREES).astype(int)[::STEP]] for i in range(RES)]


def _weights():
    """Exact least-squares cubic weights: centre (window -7..7) and the interp-mode edge rows SciPy uses."""
    def row(ts, t):
        a = [[Fraction(x) ** p for p in range(4)] for x in ts]
        ata = [[sum(a[k][i] * a[k][j] for k in range(len(ts))) for j in range(4)] for i in range(4)]
        inv = _inverse(ata)
        tv = [Fraction(t) ** p for p in range(4)]
        coef = [sum(tv[i] * inv[i][j] for i in range(4)) for j in range(4)]
        return [sum(coef[j] * a[k][j] for j in range(4)) for k in range(len(ts))]
    return {"centre": row(range(-7, 8), 0), 0: row(range(15), 0), 4: row(range(15), 4), RES - 4: row(range(15), 11)}


def _inverse(m):
    n = len(m)
    a = [list(r) + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(m)]
    for c in range(n):
        p = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[p] = a[p], a[c]
        a[c] = [x / a[c][c] for x in a[c]]
        for r in range(n):
            if r != c:
                a[r] = [x - a[r][c] * y for x, y in zip(a[r], a[c])]
    return [r[n:] for r in a]


def exact_degrees(emotion: str):
    div, w = _divtime(emotion), _weights()
    out = []
    for i in range(RES):
        row = [int(v) for v in div[i, :]]
        degs = []
        for col in range(0, RES, STEP):
            wts = w.get(col, w["centre"])
            start = (0 if col < RES // 2 else RES - 15) if col in w else col - 7
            value = sum(wt * row[start + k] for k, wt in enumerate(wts))
            degs.append(int(value % N_DEGREES))
        out.append(degs)
    return out


def quirks(emotion: str):
    """[(row, column, scipy_degree)] where SciPy's degree differs from the exact one."""
    py, ex = python_degrees(emotion), exact_degrees(emotion)
    return [(i, j * STEP, py[i][j]) for i in range(RES) for j in range(RES // STEP) if py[i][j] != ex[i][j]]


def near_integer_cells(emotion: str, tol: float = 1e-9):
    """Cells whose SciPy value is within tol of a whole number (where platforms may round differently)."""
    div = _divtime(emotion)
    cells = set()
    for i in range(RES):
        s = savgol_filter(div[i, :], 15, 3)[::STEP] % N_DEGREES
        for j, v in enumerate(s):
            if abs(v - round(v)) < tol:
                cells.add((i, j * STEP))
    return cells
