from model import Model
from object import Object
from camera import Camera
from screen import Screen
from tracker import Tracker
from physics import Physics
from quaternion import Quaternion

import numpy as np
import pygame as pg
from pygame.locals import *

import sys

import time
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

fps = 60
hz = 256
frametime = 1.0/fps

np.random.seed(42)
    
tracker = Tracker('data/IMUData.csv', dt = 1.0/hz)

skipFrames = int(hz / fps)
dt = 1.0/hz
frameTime = int(1000/fps)

width  = 1000
height = 600
ratio = width / height

surface = pg.display.set_mode((width, height))

camera = Camera(ratio=ratio, near=0.0)
camera.position = np.array([.0, .0, .0])

screen = Screen(camera, surface)
#screen.fillTriangles = True
#screen.shading = True
#screen.fast = True
correctDistortion = True
saveOutput = True


headset = Model.fromOBJFile('data/headset.obj')

def addRandomObject(objects, h = 5.0):
    theta = np.random.rand() * 2*np.pi
    r = np.random.rand()*10 + 5
    objects.append(
        Object(
            model = headset,
            position = r*np.array([np.sin(theta), .0, np.cos(theta)] + h * np.array([.0, 1.0, .0])),
            velocity = np.random.rand(3) * 20,
            angularVelocity = (np.random.rand(3)-0.5*np.array([1.0, 1.0, 1.0])) * 2*np.pi,
            #scaling = 3 * np.random.rand() * np.array([1.0, 1.0, 1.0]),
            orientation = Quaternion.fromAngleAxis(np.random.rand() * np.pi, np.random.rand(3))
        )
    )

objects = [
    Object(
        model = headset,
        position = np.array([.0, .0, 4.0]),
        velocity = np.array([.0, 5.0, .0]),
        mass = 1.0
    ),
    Object(
        model = headset,
        position = np.array([.5, 2.0, 3.0])
    )
]

#for i in range(10):
#	addRandomObject(objects, 1.0)


physics = Physics(
    dt = 0.5/fps,
    objects = objects,
    airDensity = 0.001,
    g = np.array([.0, -9.8, .0])
)



frameTime = 1.0/fps

startTime = time.time()

spawnTimer = 0.0
frame = 0
while True:
    # input
    for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN:
                    correctDistortion = not correctDistortion
                if event.key == pg.K_f:
                    screen.fillTriangles = not screen.fillTriangles
                if event.key == pg.K_g:
                    screen.shading = not screen.shading
                if event.key == pg.K_p:
                    screen.fast = not screen.fast
                if event.key == pg.K_ESCAPE:
                    quit()


    # scene
    spawnTimer += frameTime
    if (spawnTimer > 0.1):
        addRandomObject(objects, 1.0)
        if len(objects) > 1000:
            objects.pop(0)
        spawnTimer = 0.0


    # physics
    physics.integrateFrame(frameTime)

    # camera
    tracker.integrateFrame(frameTime)
    camera.orientation = tracker.orientation

    # draw
    screen.clear()
    #screen.drawObject(central)
    for obj in objects:
        screen.drawObject(obj)

    if correctDistortion: screen.correctDistortion()
    pg.display.update()

    timeTaken = time.time() - startTime
    pg.time.wait(int((frameTime - timeTaken)*1000))
    startTime = time.time()
    if saveOutput: pg.image.save(screen.surface, "output/" + str(frame) + ".png")
    frame += 1