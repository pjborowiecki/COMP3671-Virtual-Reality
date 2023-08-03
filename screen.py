import numpy as np
import pygame as pg
from math import atan2
from quaternion import Quaternion
from image import Color
from shape import Point, Triangle

class Screen:
    def __init__(self, camera, surface):
        self.camera = camera
        self.surface = surface
        self.fillTriangles = True
        self.shading = True
        self.fast = False
        self.clear()


    def clear(self):
        w,h = self.surface.get_size()
        self.zbuffer = [np.inf] * w * h
        self.surface.fill((0,0,0,0))


    
    def drawObject(self, obj):
        ProjectionViewModelMatrix = self.camera.ProjectionMatrix() @ self.camera.ViewMatrix() @ obj.ModelMatrix
        projectedVertexMatrix = np.matmul(ProjectionViewModelMatrix, obj.model.Vertices)
        projectedVertexMatrix_scr = projectedVertexMatrix[:2,:]/np.clip(projectedVertexMatrix[2,:], 1e-10, None)  # coordinates divided by z

        # skip if off-screen
        if np.min(np.max(projectedVertexMatrix_scr,axis=1)) <= -1.0 or np.max(np.min(projectedVertexMatrix_scr,axis=1)) >= 1.0: 
            return

        # TODO: avoid taking the inverse
        NormalTransform = np.linalg.inv(ProjectionViewModelMatrix[:3,:3]).T

        projectedVertexMatrix_normalised = projectedVertexMatrix / np.linalg.norm(projectedVertexMatrix[:3,], axis=0)
        projectedVertexNormalMatrix = np.matmul(NormalTransform, obj.model.VertexNormals)
        projectedVertexNormalMatrix /= np.linalg.norm(projectedVertexNormalMatrix, axis=0)
        colors = np.clip(
            np.rint(np.einsum('ij,ij->j', projectedVertexMatrix_normalised[:3,:], projectedVertexNormalMatrix)*255),
            0, 255
        )


        projectedFaceNormalMatrix = np.matmul(NormalTransform, obj.model.NormalAreas)
        projectedFaceNormalMatrix /= np.linalg.norm(projectedFaceNormalMatrix, axis=0)
        
        width, height = self.surface.get_size()
        for i in range(len(obj.model.faces)):
            face = obj.model.faces[i]
            if np.max(projectedVertexMatrix[2,face]) < 0: continue
            
            face_intensity = np.matmul(projectedVertexMatrix_normalised[:3,face[0]].T, projectedFaceNormalMatrix[:,i])[0,0]
            if face_intensity < 0: continue
            
            c = colors[face]
            p = [
                Point(
                    x = int(0.5*width*(1 - projectedVertexMatrix_scr[0,face[k]])), 
                    y = int(0.5*height*(1 - projectedVertexMatrix_scr[1,face[k]])),
                    z = projectedVertexMatrix[2,face[k]],
                    color = Color(c[k],c[k],c[k],c[k])
                ) for k in range(3)
            ]
            
            # faster but no z-buffer
            if self.fast:
                if np.min(projectedVertexMatrix[2,face]) < 0: continue
                points = [
                    (point.x, point.y) for point in p
                ]
                c = min(max(int(face_intensity * 255), 0), 255)
                color = (c,c,c,c)
                pg.draw.polygon(self.surface, color, points, 1 - self.fillTriangles)
            else:
                Triangle(p[0], p[1], p[2]).draw(self.surface, self.zbuffer)
                # pg.display.update()
            
            
    def correctDistortion(self):
        c1 = 5.0
        c2 = 5.0
        width, height = self.surface.get_size()
        surfCopy = self.surface.copy()
        self.clear()
        for px in range(0, width):
            x = 2*(px - 0.5*width)/width
            for py in range(0, height):
                y = 2*(py - 0.5*height)/height     
                r = (x**2 + y**2)**0.5
                if r == 0: continue
                cosTheta = x/r
                sinTheta = y/r
                #theta = atan2(y,x)
                rd = (c1 * r**2 + c2 * r**4 + c1**2 * r**4 + c2**2 * r**8 + 2*c1*c2*r**6)/(1 + 4*c1*r**2 + 6*c2*r**4)
                xd = rd * cosTheta
                yd = rd * sinTheta
                pxd = int((xd + 1.0)*width/2.0)
                pyd = int((yd + 1.0)*height/2.0)
                if pxd < 0 or pxd >= width or pyd < 0 or pyd >= height: continue
                self.surface.set_at((px, py), surfCopy.get_at((pxd,pyd)))
