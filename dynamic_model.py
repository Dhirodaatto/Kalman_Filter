import numpy as np
import math
from scipy.linalg import expm
class Physics_Model():

    F = None
    G = None
    dt = 0
    model_type = None

    def __init__(self, dt, model_type) -> None:
        self.dt = dt
        if model_type is None:
            self.F = np.array(
                [[1, 1 * dt, .5 * dt**2, 0, 0, 0],
                [0, 1, 1 * dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1 * dt, .5 * dt**2],
                [0, 0, 0, 0, 1, 1 * dt],
                [0, 0, 0, 0, 0, 1]]
                )    
        if model_type == "bicycle":
            # cornering stiffness
            cf, cr = 0, 0
            m = 0 # vehicle mass
            u = 0 # velocity in direction along the length of the vehicle
            v = 0 # velocity in direction perpendicular to v
            a1, a2 = 0, 0 # distance from ventre of mass to front and rear tires respectively
            I = 0 # vehicular moment of inertia about z axis
            r = 0 # angular velocity in clockwise direction i.e. <0,0,r> -> angular velocity vector

            
            # vehicle dynamics https://thef1clan.com/2020/12/23/vehicle-dynamics-the-dynamic-bicycle-model/
            A = np.array(
                    [
                        [ (cr + cf) / (m * u), (cf * a1 - cr * a2) / (m * u) + u],
                        [ (cf * a1 - cr * a2) / (u * I), (cf * a1**2 + cr * a2**2) / (I * u)]
                    ]
                    )
            
            B = np.array(
                    [
                        [(cf / m)],
                        [(cf * a1) / I]
                    ]
                    )

            self.F = self.__generate_F_matrix(A, B)
            self.G = self.__generate_G_matrix(A, B)
    
    def __generate_F_matrix(A, B):
        return expm(A * self.dt)
    
    def __generate_G_matrix(A, B):
        return expm(A * self.dt * np.linalg.pinv(A)) @ B
    
if __name__ == "__main__":
    pass
