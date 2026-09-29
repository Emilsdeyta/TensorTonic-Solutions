import numpy as np

def apply_homogeneous_transform(T: list, points: list) -> np.ndarray:
    """
    Returns transformed points with shape (3,) or (N, 3).
    """
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    single = points.ndim == 1

    if single:
        points_h = np.append(points, 1.0)
        result = T @ points_h
        return result[:3]

    points_h = np.hstack([
        points,
        np.ones((points.shape[0], 1))
    ])

    result = points_h @ T.T

    return result[:, :3]