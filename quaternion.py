import numpy as np
import numbers

class Quaternion:
    def __init__(self, w = 0, V = np.zeros(3)):
        self.w = float(w)
        self.V = np.squeeze(np.asarray(V.astype(float)))


    @staticmethod
    def fromAngleAxis(angle, axis):
        return Quaternion(np.cos(angle/2), np.sin(angle/2) * axis / np.linalg.norm(axis))


    @staticmethod
    def fromEuler(alpha, beta, gamma, rotationOrder="xyz", extrinsic=True):
        """Convert Euler angles to a quaternion.
        Parameters:
            alpha (float): rotation angle (in radians) around the first axis in the provided sequence.
            beta (float): rotation angle (in radians) around the second axis in the provided sequence.
            gamma (float): rotation angle (in radians) around the third axis in the provided sequence.
            rotationOrder (str): order of rotations, defaults to "xyz".
            extrinsic (bool): if True, the rotations are applied to a fixed frame of reference. If False, the rotations are applied to a rotating frame of reference.
        Returns:
            Quaternion: a Quaternion object (following the x, y, z, w convention), representing the input Euler angles.
        Notes:
            - Sarabandi and Thomas (2019) give a good overview of existing conversion methods: 
                [https://asmedigitalcollection.asme.org/mechanismsrobotics/article-abstract/11/2/021006/472377/A-Survey-on-the-Computation-of-Quaternions-From]
            - This implementation attempts to follow the Rotation Matrix to Quaternion conversion method.
        """

        rotationMatrix = np.eye(3)

        for axis in reversed(rotationOrder) if extrinsic else rotationOrder:
            x, y, z = (axis == a for a in "xyz")

            if x: axisRotationMatrix = [[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]]
            if y: axisRotationMatrix = [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
            if z: axisRotationMatrix = [[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]]

            rotationMatrix = np.dot(rotationMatrix, axisRotationMatrix)

        w = np.sqrt(1 + np.trace(rotationMatrix)) * 0.5
        x = (rotationMatrix[2, 1] - rotationMatrix[1, 2]) / (4 * w)
        y = (rotationMatrix[0, 2] - rotationMatrix[2, 0]) / (4 * w)
        z = (rotationMatrix[1, 0] - rotationMatrix[0, 1]) / (4 * w)

        return Quaternion(w, np.array([x, y, z]))


    def toEuler(self, rotationOrder="xyz", extrinsic=True):
        """Convert a quaternion to Euler angles in radians using the specified rotation order.
        Args:
            rotationOrder (str): A string specifying the order of rotations. Default is "xyz".
            extrinsic (bool): A boolean indicating whether to use extrinsic rotations. Default is True.
        Returns:
            A tuple of three floats representing the Euler angles in radians.
        Raises:
            ValueError: If the specified rotation order is not valid.
            TypeError: If the input is not a Quaternion object.
        Notes:
            It is assumed that the imput quaternion object is of the form (x, y, z, w).

            The methods accepts the following rotation orders: "zxz", "xyx", "yzy", "zyz", "xzx", "yxy", "xyz", "yzx", "zxy", "xzy", "zyx", "yxz".
            
            The conversion from a quaternion to Euler angles involves singularities where the output is ambiguous.
            When using extrinsic rotations, the first and third angles are between -pi/2 and pi/2, and the second angle is between 0 and pi.
            When using intrinsic rotations, the first and third angles are between 0 and pi, and the second angle is between -pi/2 and pi/2.
        
            The implementation accounts for the gimbal lock problem and handles such cases appropriately.
            
            The function was implemented based on the algorithm proposed by Bernrdes and Viollet in "Quaternion to Euler Angles Conversion" (2022).
            The authors claim around 30x speedup compared with other the Shuster method from 1993 (http://www.ladispe.polito.it/corsi/Meccatronica/02JHCOR/2011-12/Slides/Shuster_Pub_1993h_J_Repsurv_scan.pdf)
            See https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302 for details.
        """

        if rotationOrder.upper() not in ["ZXZ", "XYX", "YZY", "ZYZ", "XZX", "YXY", "XYZ", "YZX", "ZXY", "XZY", "ZYX", "YXZ"]:
            raise ValueError("Invalid rotation order. Must be one of 'zxz', 'xyx', 'yzy', 'zyz', 'xzx', 'yxy', 'xyz', 'yzx', 'zxy', 'xzy', 'zyx', 'yxz'.")

        rotationOrder = rotationOrder.upper() if extrinsic else rotationOrder[::-1].upper()

        axes = {"X": 0, "Y": 1, "Z": 2}
        i, j, k = map(axes.get, rotationOrder)
        k = (3 - i - j) if (proper := i == k) else k

        coefficient = int((i - j) * (j - k) * (k - i) * 0.5)

        x,y,z = self.V
        q = np.array([x, y, z, self.w])
        eulerAngles = np.empty(3)

        a = q[3] if proper else q[3] - q[j]
        b = q[i] if proper else q[i] + q[k] * coefficient
        c = q[j] if proper else q[j] + q[3]
        d = q[k] * coefficient if proper else q[k] * coefficient - q[i]

        normSquared = a**2 + b**2 + c**2 + d**2
        halfSum = np.arctan2(b, a)
        halfDifference = np.arctan2(-d, c)

        eulerAngles[1] = np.arccos(2*(a**2 + b**2) / normSquared - 1)

        epsilon = 1e-7
        farEnoughFromZero = abs(eulerAngles[1]) >= epsilon
        farEnoughFromPi = abs(eulerAngles[1] - np.pi) >= epsilon
        noSingularities = np.logical_and(farEnoughFromZero, farEnoughFromPi)

        if noSingularities:
            eulerAngles[0] = halfSum + halfDifference
            eulerAngles[2] = halfSum - halfDifference
        else:
            if extrinsic:
                eulerAngles[2] = 0 if not noSingularities else eulerAngles[2]
                eulerAngles[0] = 2 * halfSum if not farEnoughFromZero else eulerAngles[0]
                eulerAngles[0] = 2 * halfDifference if not farEnoughFromPi else eulerAngles[0]
            else:
                eulerAngles[0] = 0 if not noSingularities else eulerAngles[0]
                eulerAngles[2] = 2 * halfSum if not farEnoughFromZero else eulerAngles[2]
                eulerAngles[2] = 2 * halfDifference if not farEnoughFromPi else eulerAngles[2]
                
        eulerAngles = np.mod(eulerAngles+np.pi, 2*np.pi) - np.pi
        
        eulerAngles[2] *= coefficient if not proper else eulerAngles[2]
        eulerAngles[1] -= np.pi * 0.5 if not proper else eulerAngles[1]

        if not extrinsic:
            eulerAngles[0], eulerAngles[2] = eulerAngles[2], eulerAngles[0]

        return eulerAngles[0], eulerAngles[1], eulerAngles[2]


    def toRotationMatrix(self, homogeneous = False):
        R = np.asmatrix(np.eye(3 if homogeneous == False else 4))
        norm = self.norm()
        w = self.w/norm
        x,y,z  = self.V/norm
        R[:3,:3] = np.matrix([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return R

    def inv(self):
        return self.conj() * (1.0/self.norm())

    def norm(self):
        return (self.w ** 2 + np.sum(self.V ** 2)) ** 0.5

    def normalised(self):
        norm = self.norm()
        return Quaternion(self.w / norm, self.V / norm)


    def normalise(self):
        norm = self.norm()
        self.w /= norm
        self.V /= norm

    def conj(self):
        return Quaternion(self.w, -self.V)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w * other.w - np.dot(self.V, other.V),
                self.w * other.V + other.w * self.V + np.cross(self.V, other.V)
            )
        elif isinstance(other, numbers.Number):
            return Quaternion(self.w * other, self.V * other)
        
    def __rmul__(self, other):
        return self * other
    
    def __add__(self, other):
        return Quaternion(self.w + other.w, self.V + other.V)
    
    def __str__(self):
        return "(%s, %s)" % (self.w, self.V)