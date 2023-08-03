from model import Model
from transform import Transform
from quaternion import Quaternion

import numpy as np


class Object:
    def __init__(
            self,
            model,
            scaling = np.array([1.0, 1.0, 1.0]),
            mass = 1.0,
            position = np.array([0.0, 0.0, 0.0]),
            orientation = Quaternion(1),
            velocity = np.array([0.0, 0.0 ,0.0]),
            angularVelocity = np.array([0.0, 0.0, 0.0])                
        ):
        self.model = model
        self.scaling = scaling
        self.mass = mass
        self.position = position
        self.orientation = orientation

        self.force = np.array([.0, .0, .0])
        self.torque = np.array([.0, .0, .0])

        self.updateWorldCoords()
        self.setVelocities(velocity, angularVelocity)


    """ computing world coordinates is required for air friction calculation """
    def updateWorldCoords(self):
        self.ModelMatrix = self.computeModelMatrix()
        self.NormalMatrix = self.computeNormalMatrix()
        self.Inertia = self.computeInertia()
        self.Inertia_inv = self.computeInertia_inv()

        self.Vertices = np.matmul(self.ModelMatrix, self.model.Vertices)
        self.com = np.squeeze(np.asarray(np.matmul(self.ModelMatrix, np.array([.0, .0, .0, 1.0]).T)))[:3]

        # axis aligned bounding box
        self.AABB =  [
            np.squeeze(np.asarray(np.min(self.Vertices[:3,], 1))),  # min vertex
            np.squeeze(np.asarray(np.max(self.Vertices[:3,], 1)))   # max vertex
        ]

        self.FaceMidpoints = np.matmul(self.ModelMatrix, self.model.FaceMidpoints)

        self.FaceNormals = np.matmul(self.NormalMatrix, self.model.NormalAreas)
        self.FaceAreas = np.linalg.norm(np.asarray(self.FaceNormals) * (self.scaling[:,np.newaxis] ** 2), axis=0)
        self.FaceNormals /= np.linalg.norm(self.FaceNormals, axis=0)
    

    def computeBoundingSphere(self):
        #return (self.com, (2**0.5)*np.max(np.abs(self.scaling)))

        # Ritter's algorithm
        x = self.Vertices[:3,0]
        y = self.Vertices[:,np.argmax(np.linalg.norm(self.Vertices[:3,]-x, axis=0))][:3,0]
        z = self.Vertices[:,np.argmax(np.linalg.norm(self.Vertices[:3,]-y, axis=0))][:3,0]
        c = 0.5*(y+z)
        r = 0.5*np.linalg.norm(y-z)
        outside = self.Vertices[:,np.linalg.norm(self.Vertices[:3,] - c, axis=0) > r]
        for i in range(len(outside[0,:])):
            pt = outside[:3,i]
            dist = np.linalg.norm(pt-c)
            if dist > r:
                r = dist
            
        ret = (np.squeeze(np.asarray(c.T)), r)
        return ret
        
        
    def clearForces(self):
        self.force = np.array([.0, .0, .0])
        self.torque = np.array([.0, .0, .0])

    def applyForce(self, force, position):
        r = position - self.com
        self.force += force
        self.torque += np.cross(r, force)

    def applyImpulse(
            self,
            r,   # vector from centre of mass to where the impulse is applied
            impulse
    ):
        self.momentum += impulse
        self.angularMomentum += np.cross(r,impulse)

    def setVelocities(
            self,
            velocity = None,
            angularVelocity = None
    ):
        self.momentum = self.mass * velocity if velocity is not None else self.momentum
        self.angularMomentum = self.Inertia @ angularVelocity if angularVelocity is not None else self.angularMomentum


    def getVelocities(self):
        R = self.orientation.toRotationMatrix()
        velocity = self.momentum / self.mass
        angularVelocity = np.squeeze(np.asarray(self.Inertia_inv @ self.angularMomentum.T).T)
        return velocity, angularVelocity

    def computeModelMatrix(self):
        S = Transform.Scaling(self.scaling)
        R = self.orientation.toRotationMatrix(homogeneous = True)
        T = Transform.Translation(self.position)
        return T @ R @ S


    def computeNormalMatrix(self):
        #return np.linalg.inv(self.computeModelMatrix()).T[:3,:3]
        return np.linalg.inv(self.ModelMatrix[:3,:3]).T
    
    def computeInertia(self):
        J = self.model.Inertia
        S = Transform.Scaling(self.scaling)[:3,:3]
        S_inv = Transform.Scaling(np.reciprocal(self.scaling))[:3,:3]
        R = self.orientation.toRotationMatrix(homogeneous = True)[:3,:3]
        R_inv = R.T
        return self.mass * (S_inv @ R_inv @ J @ R @ S)
    
    def computeInertia_inv(self):
        J = self.model.Inertia_inv
        S = Transform.Scaling(self.scaling)[:3,:3]
        S_inv = Transform.Scaling(np.reciprocal(self.scaling))[:3,:3]
        R = self.orientation.toRotationMatrix(homogeneous = True)[:3,:3]
        R_inv = R.T
        return (1.0/self.mass) * (S_inv @ R_inv @ J @ R @ S)