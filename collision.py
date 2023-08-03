from model import Model
from object import Object
from camera import Camera
from screen import Screen
from quaternion import Quaternion
from physics import Physics

import numpy as np
import pygame as pg
from pygame.locals import *

np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))


fps = 60
width  = 1000
height = 600
ratio = width / height

surface = pg.display.set_mode((width, height))

camera = Camera(
    ratio=ratio,
    position = np.array([.0, .0, -2.0])

)

screen = Screen(camera, surface)
screen.fillTriangles = False
screen.shading = True

#quit()
headset = Model.fromOBJFile('data/headset.obj')
#quit()
#cube = Model.fromOBJFile('data/cube.obj')
#big = Model.fromOBJFile('data/whale.obj')

objects = [

    Object(
        model = headset,
        position = np.array([.0, 1.0, 0.0]),
        velocity = np.array([.0, 0.0, .0]),
        #orientation = Quaternion.fromAngleAxis(np.pi/7, np.array([2.0, 1.0, 3.0])),
        mass = 1.0,
        scaling = np.array([1.0, 1.0, 1.0])
    )
]
#quit()

physics = Physics(
    dt = 1.0/fps,
    objects = objects,
    airDensity = 0.1,
    g = np.array([.0, -1.0, .0])
)

camRightAngle = 0.0
camUpAngle = 0.0
camRoll = 0.0
screen.fast = True
while True:
    e = pg.event.get()
    keys = pg.key.get_pressed()
    if keys[K_LEFT]:
        camRightAngle += 0.1
    if keys[K_RIGHT]:
        camRightAngle += -0.1
    if keys[K_UP]:
        camUpAngle += -0.1
    if keys[K_DOWN]:
        camUpAngle += 0.1
    if keys[K_PERIOD]:
        camRoll += -0.1
    if keys[K_COMMA]:
        camRoll += 0.
    
    camera.orientation = Quaternion.fromAngleAxis(camRightAngle, np.array([0, 1, 0])) \
                * Quaternion.fromAngleAxis(camUpAngle, np.array([1, 0, 0])) \
                * Quaternion.fromAngleAxis(camRoll, np.array([0, 0, 1]))
    #print(camera.getForward())
    physics.integrateFrame(frameTime = 1.0/fps)
    screen.clear()
    for object in objects:
        screen.drawObject(object)
    pg.display.flip()
    pg.time.wait(int(1000.0/fps))