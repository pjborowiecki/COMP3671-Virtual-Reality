from object import Object
from quaternion import Quaternion
import numpy as np
import time

class Collisions:
    def find(objects):
        ret = []

        # broad stage
        possibleCollisions = Collisions.sweepAndPrune(objects)
        
        # narrow stage - get colliding pair as well as minimum translation vector (penetration depth, collision normal)
        for (obj1, obj2) in possibleCollisions:
            c1,r1 = obj1.computeBoundingSphere()
            c2,r2 = obj2.computeBoundingSphere()
            v = c2-c1
            d = np.linalg.norm(v)
            if d < r1+r2:
                midpoint = 0.5*(c2 + c1 + (r1-r2)/d * (c2-c1))
                penetration = r1 + r2 - d
                direction = v / d
                ret.append([obj1, obj2, penetration, np.squeeze(np.asarray(direction.T))[:3], np.squeeze(np.asarray(midpoint.T))[:3]])   # note: direction obj1 -> obj2
        
        return ret

    # ---------------------------------------------------------
    # Broad stage
    # ---------------------------------------------------------
    # broad search for collisions using sweep and prune algorithm
    @staticmethod
    def sweepAndPrune(objects):
        overlappingList = objects
        overlappingPairs = set()
        axis = 0
        while axis <= 2 and len(overlappingList) > 0:
            overlappingPairs, overlappingList = Collisions.sweepAxis(overlappingList, axis)
            axis += 1
        return overlappingPairs


    # get pairs of objects whose projections to {axis} overlap
    @staticmethod
    def sweepAxis(objects, axis=0):
        axisList = sorted(objects, key=lambda x: x.AABB[0][axis])

        overlappingPairs = set()
        activeList = []
        overlappingList = set()

        i = 0
        while i < len(axisList):
            j = 0
            while j < len(activeList):
                if (axisList[i].AABB[0][axis] > activeList[j].AABB[1][axis]):
                    activeList.pop(j)
                    j -= 1
                else:
                    overlappingPairs.add(frozenset([axisList[i], activeList[j]]))
                    overlappingList.add(axisList[i])
                    overlappingList.add(activeList[j])
                j += 1
            activeList.append(axisList[i])
            i += 1
        return overlappingPairs, overlappingList




class Physics:
    def __init__(self,
                 dt,
                 objects = [],
                 g = np.array([.0, -1.0, .0]),
                 airDensity = 1.0
    ):
        self.objects = objects
        self.dt = dt
        self.g = g
        self.airDensity = airDensity


    def applyAirResistance(self, obj):
        velocity, angularVelocity = obj.getVelocities()
        v = np.linalg.norm(velocity)

        # r = midpoint - obj.com
        rs = obj.FaceMidpoints[:3,] - np.asmatrix(obj.com).T
        
        # v_face = velocity + angularVelocity x r
        v_faces = np.asmatrix(velocity).T + np.asmatrix(np.cross(angularVelocity, rs.T)).T
        
        # vA = max(v_face dot FaceNormal * FaceArea, 0.0)
        vA = np.clip(np.multiply(np.sum(np.multiply(v_faces, obj.FaceNormals), axis=0), np.asmatrix(obj.FaceAreas)), 0, None)
        
        # force: in direction -v_face, magnitude v_face^2 * area cross section * density
        forces = -np.multiply(v_faces, vA) * 0.5 * self.airDensity
        
        # torque = r x force
        torques = np.asmatrix(np.cross(rs.T, forces.T)).T
        
        totalForce = np.squeeze(np.asarray(np.sum(forces, axis=1).T))
        totalTorque = np.squeeze(np.asarray(np.sum(torques, axis=1).T))

        """
        Note: bound on force due to friction
            It can happen that for large v, v^2 is even bigger and force due to drag can also get very large
            If F * dt is large enough it can reverse direction of velocity (and hence momentum p), which is unrealistic for frictional force
            and can cause the solution to diverge.
            To mitigate this we can bound |F| by -M|V|/(dt * <f|v>)  where f, v are directions of F and V 
            (and <f|v> < 0 - at least it should be, there could be some weird effect where a rotating face is pushing against the wind? so we check that just in case)
            The bound is obtained by insisting that P_new = P + Fdt satisfies <P_new | P> >= 0
        """
        if np.dot(totalForce, velocity) < 0:
            Fmag = np.linalg.norm(totalForce)
            Fdir = totalForce / Fmag
            Vdir = velocity / np.linalg.norm(velocity)
            Fmag = min(Fmag, obj.mass/(self.dt * (-1) * np.dot(Fdir,Vdir)))
            totalForce = Fmag * Fdir
        
        obj.force += totalForce
        obj.torque += totalTorque

        
    def applyGravity(self, obj):
        obj.force += obj.mass * self.g

            

    def step(self):
        for obj in self.objects:
            self.applyGravity(obj)
            self.applyAirResistance(obj)
            self.updateBodyState_Euler(obj)

            obj.updateWorldCoords()
            obj.clearForces()


        self.resolveCollisions()
            



    def integrateFrame(self, frameTime):
        t = 0
        while (t < frameTime):
            self.step()
            t += self.dt


    def updateBodyState_Euler(self, obj):
        dt = self.dt

        """
        Note: bound on torque
            Would like angular velocity w := J^-1 * L (where J = Inertia tensor) to not change too much between frames (|dw| < C, say)
            |T|dt = |Jdw| <= |J||dw| < |J|C  => |T| < |J|*C / dt (|J| = l2 operator norm)
            e.g. C = 2pi, so we won't do full revolution between frames, which seems reasonable
            probably much less is enough            
        """
        if np.any(obj.torque):
            Tmag = np.linalg.norm(obj.torque)
            Tdir = obj.torque / Tmag
            #C = 2*np.pi
            C = np.pi/2
            #C = 0.1
            Tmag = min(Tmag, np.linalg.norm(obj.Inertia) * C/self.dt)
            obj.torque = Tmag * Tdir


        obj.momentum += obj.force * dt
        obj.angularMomentum += obj.torque * dt



        # limit momentum and angularMomentum so the simulation doesn't blow up
        pmag = np.linalg.norm(obj.momentum) 
        pdir = obj.momentum / pmag
        obj.momentum = min(10e3 * obj.mass, pmag) * pdir


        obj.position += obj.momentum / obj.mass * dt
        _, angularV = obj.getVelocities()
        obj.orientation += 0.5 * Quaternion(0, angularV) * obj.orientation * dt
        obj.orientation.normalise()


    def resolveCollisions(self):
        collisions = Collisions.find(self.objects)

        for obj1, obj2, penetration, n, contact in collisions:
            ma = obj1.mass / len(obj1.model.Vertices[:,0])
            mb = obj2.mass / len(obj2.model.Vertices[:,0])
            
            xa = obj1.com
            xb = obj2.com
            va,angva = obj1.getVelocities()
            vb,angvb = obj2.getVelocities()

            Ia_inv = obj1.Inertia_inv
            Ib_inv = obj1.Inertia_inv


            ra = contact - xa
            rb = contact - xb
            vpa = va + np.cross(angva,ra)
            vpb = vb + np.cross(angvb,rb)
            vrel = np.dot(n, vpa-vpb)

            #print(n @ Ia_inv @ np.cross(ra,n))
            #j = -np.dot((vb + np.cross(angvb,rb) - va - np.cross(angva,ra)),n) / (1.0/ma + 1.0/mb + n * np.cross(Ia_inv @ np.cross(ra,n), ra) + n * np.cross(Ib_inv @ np.cross(rb,n),rb))
            j = -1.0 * vrel*n / (1.0/ma + 1.0/mb + np.dot(n, np.squeeze(np.asarray(np.cross(Ia_inv @ np.cross(ra,n), ra) + np.cross(Ib_inv @ np.cross(rb,n),rb)))))
            J = np.squeeze(np.asarray(j*n))
            #J /= len(contactPoints)
            #print("impulse = ", J)
            obj1.applyImpulse(ra, J)
            obj2.applyImpulse(rb, -J)                