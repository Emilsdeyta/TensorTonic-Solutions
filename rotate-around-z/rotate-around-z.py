import numpy as np

def rotate_around_z(points: list, theta: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as points.
    """
    result1 = []
    if isinstance(points[0], list):
        # row = []
        for point in points:
            x1 = point[0]*np.cos(theta)  - point[1]*np.sin(theta)
            y1 = point[0]*np.sin(theta) + point[1]*np.cos(theta)
            z1 = point[2]
            row = (x1, y1, z1)
        
            result1.append(row)
        return np.array(result1)

        
    else:
        x1 = points[0]*np.cos(theta)  - points[1]*np.sin(theta)
        y1 = points[0]*np.sin(theta) + points[1]*np.cos(theta)
        z1 = points[2]
    
        result = [x1, y1, z1]
        return np.stack(result, axis=-1)