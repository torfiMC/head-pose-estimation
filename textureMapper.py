"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect human face in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

To find more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""


#!/usr/bin/env python

# Author: Ryan Myers
# Models: Jeff Styers, Reagan Heller
#
# Last Updated: 2015-03-13
#
# This tutorial provides an example of creating a character
# and having it walk around on uneven terrain, as well
# as implementing a fully rotatable camera.

from pose_estimator import PoseEstimator
from mark_detector import MarkDetector
import cv2
from argparse import ArgumentParser
from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerQueue, CollisionRay
from panda3d.core import Filename, AmbientLight, DirectionalLight
from panda3d.core import PandaNode, NodePath, Camera, TextNode, CardMaker, Texture, Vec3, OrthographicLens, Point3, TexProjectorEffect, LensNode, TextureStage
from panda3d.core import CollideMask, PerspectiveLens
from direct.task.Task import Task
from direct.gui.OnscreenText import OnscreenText
from direct.actor.Actor import Actor
import random
import sys
import os
import math


print(__doc__)
# print("OpenCV version: {}".format(cv2.__version__))

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


# Function to put instructions on the screen.
def addInstructions(pos, msg):
    return OnscreenText(text=msg, style=1, fg=(1, 1, 1, 1), scale=.05,
                        shadow=(0, 0, 0, 1), parent=base.a2dTopLeft,
                        pos=(0.08, -pos - 0.04), align=TextNode.ALeft)

# Function to put title on the screen.


def addTitle(text):
    return OnscreenText(text=text, style=1, fg=(1, 1, 1, 1), scale=.07,
                        parent=base.a2dBottomRight, align=TextNode.ARight,
                        pos=(-0.1, 0.09), shadow=(0, 0, 0, 1))


class RoamingRalphDemo(ShowBase):
    def __init__(self):
        # Set up the window, camera, etc.
        ShowBase.__init__(self)

        # Set the background color to black
        self.win.setClearColor((0, 0, 0, 1))

        # This is used to store which keys are currently pressed.
        self.keyMap = {
            "left": 0, "right": 0, "forward": 0, "cam-left": 0, "cam-right": 0, "up": 0, "down": 0, "back": 0}

        self.offset = [0.0, 0.0, 0.0]
        self.lastPos = [0.0, 0.0, 0.0]

        card_maker = CardMaker("quad")
        # Define the size of the quad (full screen)
        card_maker.setFrame(-1, 1, -1, 1)
        self.quad = self.render.attachNewNode(card_maker.generate())

        texture = self.loader.loadTexture("models/test2.png")
        texture.setFormat(Texture.F_rgb)
        self.texture = texture
        self.quad.setTexture(texture)
        # card_maker.setNumVertices(100)
        card_maker.setUvRange((0, 0), (1, 1))

        self.quad.setLightOff()

        # Create a lens for the texture projection
        proj_lens = PerspectiveLens()
        proj_lens.setFov(40, 40)

        projector = self.render.attachNewNode(LensNode("projector"))
        # projector.setPos(10, -10, 5)  # Example position
        # projector.lookAt(self.quad)
        print(projector.node)
        projector.node().setLens(proj_lens)
        self.projector = projector

        ts = TextureStage('ts')
        ts.setSort(1)
        ts.setMode(TextureStage.MDecal)
        self.ts = ts
        # projector.setPos(10, -10, 5)  # Example position
        # projector.lookAt(self.quad)
        self.quad.projectTexture(self.ts, self.texture, self.projector)

        lens = OrthographicLens()
        lens.setFilmSize(2, 2)  # Adjust if needed, based on the quad's size
        self.cam.node().setLens(lens)
        self.cam.setPos(0, -10, 0)  # Camera position
        self.cam.lookAt(self.quad)  # Look straight at the quad
        self.gameTask = taskMgr.add(self.gameLoop, "gameLoop")

        # Accept the control keys for movement and rotation

        self.accept("escape", sys.exit)
        self.accept("space", self.reset)

        self.accept("a", self.setKey, ["left", True])
        self.accept("d", self.setKey, ["right", True])
        self.accept("w", self.setKey, ["up", True])
        self.accept("s", self.setKey, ["down", True])
        self.accept("q", self.setKey, ["forward", True])
        self.accept("e", self.setKey, ["back", True])

        self.accept("a-up", self.setKey, ["left", False])
        self.accept("d-up", self.setKey, ["right", False])
        self.accept("w-up", self.setKey, ["up", False])
        self.accept("s-up", self.setKey, ["down", False])
        self.accept("q-up", self.setKey, ["forward", False])
        self.accept("e-up", self.setKey, ["back", False])

        # Set up the camera
        self.disableMouse()

        # 1. Setup the video source from webcam or video file.
        self.video_src = args.cam if args.cam is not None else args.video
        if self.video_src is None:
            print("Video source not assigned, default webcam will be used.")
            self.video_src = 0

        self.cap = cv2.VideoCapture(self.video_src)

        # Get the frame size. This will be used by the pose estimator.
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 2. Introduce a pose estimator to solve pose.
        self.pose_estimator = PoseEstimator(img_size=(self.height, self.width))

        # 3. Introduce a mark detector to detect landmarks.
        self.mark_detector = MarkDetector()

        # 4. Measure the performance with a tick meter.
        self.tm = cv2.TickMeter()

        self.translation_vector = [[0, 0, 0]]
        self.x = 0
        self.y = 0
        self.z = 0

        self.xoffset = 0.0
        self.yoffset = 0.0
        self.zoffset = 45.0

    # Records the state of the arrow keys

    def gameLoop(self, task):
        # Get the time elapsed since the next frame.  We need this for our
        # distance and velocity calculations.
        dt = globalClock.getDt()
        print(dt)
        # If the ship is not alive, do nothing.  Tasks return Task.cont to
        # signify that the task should continue running. If Task.done were
        # returned instead, the task would be removed and would no longer be
        # called every frame.
        # if not self.alive:
        #    return Task.cont
        self.captureFrame()

        if self.keyMap["left"]:
            self.xoffset -= 1.0
            print("xoffset ", self.xoffset)
        if self.keyMap["right"]:
            self.xoffset += 1.0
            print("xoffset ", self.xoffset)
        if self.keyMap["up"]:
            self.zoffset += 1.0
            print("zoffset ", self.zoffset)
        if self.keyMap["down"]:
            self.zoffset -= 1.0
            print("zoffset ", self.zoffset)

        if self.keyMap["forward"]:
            self.yoffset += 10.0
            print("yoffset ", self.yoffset)
        if self.keyMap["back"]:
            self.yoffset -= 10.0
            print("yoffset ", self.yoffset)

        #    self.ralph.setY(self.ralph, -25 * dt)

        return Task.cont
     # Since every return is Task.cont, the task will
        # continue indefinitely

    def captureFrame(self):
        # Read a frame.
        frame_got, frame = self.cap.read()
        if frame_got is False:
            return

        # If the frame comes from webcam, flip it so it looks like a mirror.
        # if self.video_src == 0:
        #    frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = self.mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            # self.tm.start()
            marks = self.mark_detector.detect_marks(face_img)
            # self.tm.stop()

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Try pose estimation with 68 points.
            pose = self.pose_estimator.solve_pose_by_68_points(marks)

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            # frame.
            # self.pose_estimator.draw_annotation_box(
            #    frame, pose[0], pose[1], color=(0, 255, 0))

            self.translation_vector = pose[1]
            # print(self.translation_vector[0])
            scale = 0.1

            # Calculate the magnitude of the vector
            magnitude = math.sqrt(
                self.translation_vector[0]**2 + self.translation_vector[1]**2 + self.translation_vector[2]**2)

            # print(self.translation_vector)

            magnitude = 1500.0

            x = (self.xoffset + self.translation_vector[0]) / magnitude
            # exchange z and y since it's left handed vs right handed coordinates system
            y = (self.yoffset + self.translation_vector[2]) / magnitude
            z = (self.zoffset + self.translation_vector[1]) / magnitude
            # print(x, y, z)

            self.x = x
            self.y = y
            self.z = z

            self.projector.setPos(x, y, z)  # Example position
            # self.projector.lookAt(self.quad)
            # print(self.translation_vector)
            # self.quad.projectTexture(self.ts, self.texture, self.projector)

            # Do you want to see the head axes?
            # self.pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            # self.mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # self.mark_detector.draw_box(frame, [facebox])

        # Show preview.
        # cv2.imshow("Preview", frame)

    def reset(self):
        self.xoffset = -self.x
        self.yoffset = -self.y

    def setKey(self, key, value):
        self.keyMap[key] = value


demo = RoamingRalphDemo()
demo.run()
