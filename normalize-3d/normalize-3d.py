import numpy as np

def normalize_3d(v: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as v.
    """
    if isinstance(v[0], list):
        answer = []

        for i in v:
            i = np.array(i, dtype=float)

            result = 0
            for j in i:
                result += j**2

            norms = np.sqrt(result)

            normalized = np.divide(
                i,
                norms,
                out=np.zeros_like(i, dtype=float),
                where=norms != 0.0
            )

            answer.append(normalized)

        return np.array(answer)

    else:
        v = np.array(v, dtype=float)

        result = 0
        for i in v:
            result += i**2

        norms = np.sqrt(result)

        return np.divide(
            v,
            norms,
            out=np.zeros_like(v, dtype=float),
            where=norms != 0.0
        )