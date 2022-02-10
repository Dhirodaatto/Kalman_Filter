import numpy as np
import math
from scipy.linalg import expm
class Physics_Model():

    F = None
    G = None
    dt = 0
    model_type = None

    def __init__(self, dt, model_type) -> None:
        if model_type is None:
            self.F = np.array(
                [[1, 1 * dt, .5 * dt**2, 0, 0, 0],
                [0, 1, 1 * dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1 * dt, .5 * dt**2],
                [0, 0, 0, 0, 1, 1 * dt],
                [0, 0, 0, 0, 0, 1]]
                )    
    
    def __generate_F_matrix(A, B):
        return expm(A * dt)
    
    def __generate_G_matrix(A, B):
        return expm(A * dt * np.linalg.pinv(A)) @ B
    
if __name__ == "__main__":
    pass
