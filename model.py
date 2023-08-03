import numpy as np
from transform import Transform

class OBJFile:
    def __init__(self, file):
        self.vertices = []
        self.faces = []

        # Read in the file
        f = open(file, 'r')
        for line in f:
            if line.startswith('#'): continue
            segments = line.split()
            if not segments: continue

            # Vertices
            if segments[0] == 'v':
                self.vertices.append(([float(i) for i in segments[1:4]]))

            # Faces
            elif segments[0] == 'f':
                # Support models that have faces with more than 3 points
                # Parse the face as a triangle fan
                for i in range(2, len(segments)-1):
                    corner1 = int(segments[1].split('/')[0])-1
                    corner2 = int(segments[i].split('/')[0])-1
                    corner3 = int(segments[i+1].split('/')[0])-1
                    self.faces.append([corner1, corner2, corner3])


class Model:
    def __init__(
            self, 
            vertices, 
            faces, 
            proportionalNormalise = True
        ):
        self.faces = faces
        self.Vertices = np.matrix(np.ones((4,len(vertices))))
        self.Vertices[:3,] = np.matrix(vertices).T

        # compute face normal areas matrix (magnitudes are the areas of triangular faces)
        self.NormalAreas = self.computeNormalAreasMatrix(self.Vertices, self.faces)

        faceAreas = np.linalg.norm(self.NormalAreas, axis=0)

        # set the local origin to be the body's centre of mass (assuming uniform mass distribution)
        totalFaceArea = np.sum(faceAreas)
        #print("totalArea = ", totalArea)
        com = np.matrix([.0, .0, .0]).T
        for i in range(len(self.faces)):
            face = self.faces[i]
            midpoint = self.Vertices[:,face].mean(1)[:3]
            mass = faceAreas[i]/totalFaceArea
            com += midpoint * mass

        self.Vertices[:3,] -= com
        
        if proportionalNormalise:
            self.Vertices[:3,] /= np.max(np.abs(self.Vertices[:3,]))
        else: self.Vertices[:3,:3] /= np.max(np.abs(self.Vertices[:3,3]))

        
        # compute vertex normal matrix
        self.VertexNormals = self.computeVertexNormalMatrix(self.Vertices, self.NormalAreas, self.faces)

        # compute the inertia tensor and its inverse, and btw face midpoints
        self.FaceMidpoints = np.ones((4,len(faces)))

        V = np.zeros((3,len(faces)))
        for i in range(len(self.faces)):
            face = self.faces[i]
            midpoint = np.squeeze(np.asarray(self.Vertices[:,face].mean(1)[:3]))
            self.FaceMidpoints[:3,i] = np.asarray(midpoint).T
            mass = faceAreas[i]/totalFaceArea
            V[:,i] =  (mass ** 0.5) * midpoint.T

        C = np.matmul(V, V.T)
        self.Inertia = np.trace(C) * np.eye(3) - C 
        self.Inertia_inv = np.linalg.inv(self.Inertia)

        self.BoundingSphere = self.computeBoundingSphere()

        

    # direction vectors have 3 coordinates so matrix is 3xN
    def computeNormalAreasMatrix(self, VertexMatrix, faces):
        NormalAreas = np.matrix(np.zeros((3,len(faces))))
        for i in range(len(faces)):
            p0, p1, p2 = VertexMatrix[:3,faces[i]].T
            NormalAreas[:,i] = 0.5 * np.cross(p2-p0, p1-p0).T
        return NormalAreas
    

    def computeVertexNormalMatrix(self, VertexMatrix, NormalMatrix, faces):
        facesContainingVertex = lambda v: list(filter(lambda f: v in faces[f], range(len(faces))))
        VertexNormals = np.matrix(np.zeros((3,VertexMatrix.shape[1])))
        for i in range(VertexMatrix.shape[1]):
            VertexNormals[:,i] = NormalMatrix[:,facesContainingVertex(i)].mean(axis=1)
        return VertexNormals / np.linalg.norm(VertexNormals, axis=0)


    def computeBoundingSphere(self):
        # Ritter's algorithm
        x = self.Vertices[:,0]
        y = self.Vertices[:,np.argmax(np.linalg.norm(self.Vertices[:,]-x, axis=0))]
        z = self.Vertices[:,np.argmax(np.linalg.norm(self.Vertices[:,]-y, axis=0))]

        c = 0.5*(y+z)
        r = 0.5*np.linalg.norm(y-z)
        
        for pt in np.asarray(self.Vertices[:,np.linalg.norm(self.Vertices[:,] - c, axis=0) > r].T):
            dist = np.linalg.norm(pt-c)
            
            if dist > r:
                r = dist

        return (c, r)

        

    @staticmethod
    def fromOBJFile(filename):
        data = OBJFile(filename)
        return Model(
            vertices = data.vertices, 
            faces = data.faces
        )