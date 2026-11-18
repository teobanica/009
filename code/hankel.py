"""Exact inertia of finite symmetric matrices and Hankel truncations.

All arithmetic is rational. Congruence elimination uses a nonzero diagonal
pivot when possible. If every remaining diagonal entry vanishes, a nonzero
off-diagonal entry gives a 2 by 2 block of inertia (1, 1, 0). A completely
zero residual matrix contributes only null directions. In particular a zero
Gaussian pivot never causes an indefinite matrix to be declared positive.
"""

from fractions import Fraction


def inertia(matrix):
    """Return the numbers of positive, negative, and zero eigenvalues."""
    residual = [[Fraction(value) for value in row] for row in matrix]
    size = len(residual)
    if any(len(row) != size for row in residual):
        raise ValueError("The matrix must be square")
    if any(residual[i][j] != residual[j][i]
           for i in range(size) for j in range(i)):
        raise ValueError("The matrix must be symmetric")
    positive = negative = zero = 0
    while residual:
        size = len(residual)
        pivot = next((i for i in range(size) if residual[i][i]), None)
        if pivot is not None:
            diagonal = residual[pivot][pivot]
            positive += diagonal > 0
            negative += diagonal < 0
            rest = [i for i in range(size) if i != pivot]
            residual = [
                [residual[i][j] - residual[i][pivot] * residual[pivot][j] / diagonal
                 for j in rest]
                for i in rest
            ]
            continue
        pair = next(((i, j) for i in range(size) for j in range(i)
                     if residual[i][j]), None)
        if pair is None:
            zero += size
            break
        first, second = pair
        off_diagonal = residual[first][second]
        positive += 1
        negative += 1
        rest = [i for i in range(size) if i not in pair]
        residual = [
            [residual[i][j]
             - (residual[i][first] * residual[second][j]
                + residual[i][second] * residual[first][j]) / off_diagonal
             for j in rest]
            for i in rest
        ]
    return positive, negative, zero


def hankel_inertia(sequence, size=None):
    """Test H=(c[i+j]); the default is the largest available truncation."""
    if size is None:
        size = (len(sequence) + 1) // 2
    if size < 1 or 2 * size - 1 > len(sequence):
        raise ValueError("Insufficient moments for this Hankel size")
    return inertia([[sequence[i + j] for j in range(size)] for i in range(size)])


def first_non_psd_size(sequence):
    """Return the first available non-PSD leading Hankel size, or None."""
    for size in range(1, (len(sequence) + 1) // 2 + 1):
        if hankel_inertia(sequence, size)[1]:
            return size
    return None


def hankel_summary(sequence):
    """Describe only the computed finite matrix, without an infinite claim."""
    positive, negative, zero = hankel_inertia(sequence)
    status = "NOT PSD" if negative else "PSD" if zero else "PD"
    return f"size {positive + negative + zero}: {status}, inertia {(positive, negative, zero)}"
