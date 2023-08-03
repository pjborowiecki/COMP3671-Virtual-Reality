import numpy as np
from transform import Transform
from quaternion import Quaternion

class Camera:
    def __init__(
            self, 
            ratio, 
            fov=np.pi/2, 
            near=0.01, 
            far=100,
            position = np.array([.0, .0, .0]),
            orientation = Quaternion(1)
        ):
        self.position = position
        self.orientation = orientation
        self.ratio = ratio
        self.fov = fov
        self.near = near
        self.far = far

    def getForward(self):
        return np.squeeze(np.asarray(self.orientation.toRotationMatrix()[:,2]))

    def lookAt(self, target):
        currentForward = self.getForward()
        newForward = target - self.position
        newForward /= np.linalg.norm(newForward)
        axis = np.cross(currentForward, newForward)
        if not any(axis):
            axis = np.array([0,1,0])
        dot = np.dot(currentForward, newForward)
        angle = np.arccos(dot)
        self.orientation = self.orientation * Quaternion.fromAngleAxis(angle, axis)

    def ViewMatrix(self):
        R = np.asmatrix(np.eye(4))
        R[:3,:3] = self.orientation.inv().toRotationMatrix()
        return R @ Transform.Translation(-self.position)
    
    def ViewMatrix_inv(self):
        R = np.asmatrix(np.eye(4))
        R[:3,:3] = self.orientation.toRotationMatrix()
        return Transform.Translation(-self.position) @ R    

    def ProjectionMatrix(self):
        f = 1 / np.tan(self.fov / 2)
        ratio,far,near = self.ratio, self.far, self.near
        return np.matrix([
            [f/ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near)/(far - near), 2*far*near/(far - near)],
            [0,0,-1,0]
        ])

    def ProjectionMatrix_inv(self):
        f = 1 / np.tan(self.fov / 2)
        ratio,far,near = self.ratio, self.far, self.near
        return np.matrix([
            [ratio/f, 0, 0, 0],
            [0, 1/f, 0, 0],
            [0, 0, 0, -1],
            [0, 0, (far-near)/(2*far*near), (far+near)/(2*far*near)]
        ])