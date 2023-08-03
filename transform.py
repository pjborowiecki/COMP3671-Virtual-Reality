import numpy as np
from quaternion import Quaternion

class Transform:

    # translation
    @staticmethod
    def Translation(pos):
        x,y,z = np.squeeze(np.asarray(pos))[:3]
        return np.matrix([
            [1,0,0,x],
            [0,1,0,y],
            [0,0,1,z],
            [0,0,0,1]])
    
    # rotation about axis given by [x,y,z]
    @staticmethod
    def Rotation(v, angle):
        R = np.asmatrix(np.eye(4))
        R[:3,:3] = Quaternion(angle, np.squeeze(np.asarray(v)[:3]), True).toRotationMatrix()
        return R

    # scaling in axes [sx,sy,sz]
    @staticmethod
    def Scaling(s):
        sx,sy,sz = s
        return np.matrix([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ])