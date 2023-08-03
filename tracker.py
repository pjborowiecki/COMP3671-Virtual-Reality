from quaternion import Quaternion
import csv
import numpy as np

class Tracker:
    def __init__(self, filename, dt = 1.0/256, alpha = 0.01):
        self.csvfile = open(filename, newline='')
        self.datareader = csv.reader(self.csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        
        self.orientation = Quaternion(1)
        self.q_gyro = Quaternion(1)

        self.dt = dt
        self.alpha = alpha

    def integrateFrame(self, frameTime):
        t = 0

        while t < frameTime:
            row = next(self.datareader, None)
            if row is None:
                return False

            _,gyroX,gyroY,gyroZ,accX,accY,accZ,_,_,_ = row

            # dead reckoning filter
            gyro = np.array([gyroY,gyroZ,gyroX]) * (np.pi / 180)
            gyro_norm = np.linalg.norm(gyro)
            dq_gyro = Quaternion.fromAngleAxis(self.dt * gyro_norm, gyro / gyro_norm) if gyro_norm != 0 else Quaternion(1)
            self.q_gyro = self.q_gyro * dq_gyro 

            # tilt correction
            acc = np.array([accY,accZ,accX])
            acc = acc / np.linalg.norm(acc) if any(acc) else np.array([.0, 1.0, .0])
            q_acc_body = Quaternion(0, acc).normalised()
            q_acc_world = (self.q_gyro * q_acc_body * self.q_gyro.inv()).normalised()
            ax,ay,az = q_acc_world.V
            tiltAxis = np.array([az, 0, -ax])
            tiltAxis /= np.linalg.norm(tiltAxis)
            cosphi = np.dot(q_acc_world.V / np.linalg.norm(q_acc_world.V), np.array([.0, 1.0, .0]))
            phi = np.arccos(cosphi)
            
            self.orientation = self.q_gyro * Quaternion.fromAngleAxis(-self.alpha * phi, tiltAxis)

            t += self.dt

        return True