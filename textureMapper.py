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

from direct.showbase.ShowBase import ShowBase
from panda3d.core import CollisionTraverser, CollisionNode
from panda3d.core import CollisionHandlerQueue, CollisionRay
from panda3d.core import Filename, AmbientLight, DirectionalLight
from panda3d.core import PandaNode, NodePath, Camera, TextNode
from panda3d.core import CollideMask
from direct.task.Task import Task
from direct.gui.OnscreenText import OnscreenText
from direct.actor.Actor import Actor
import random
import sys
import os
import math


from argparse import ArgumentParser

import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

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

        # Set up the environment
        #
        # This environment model contains collision meshes.  If you look
        # in the egg file, you will see the following:
        #
        #    <Collide> { Polyset keep descend }
        #
        # This tag causes the following mesh to be converted to a collision
        # mesh -- a mesh which is optimized for collision, not rendering.
        # It also keeps the original mesh, so there are now two copies ---
        # one optimized for rendering, one for collisions.

        self.environ = loader.loadModel("models/world")
        self.environ.reparentTo(render)

        # Create the main character, Ralph

        ralphStartPos = self.environ.find("**/start_point").getPos()
        self.ralph = Actor("models/ralph",
                           {"run": "models/ralph-run",
                            "walk": "models/ralph-walk"})
        self.ralph.reparentTo(render)
        self.ralph.setScale(.2)
        self.ralph.setPos(ralphStartPos + (0, 0, 0.5))

        # Create a floater object, which floats 2 units above ralph.  We
        # use this as a target for the camera to look at.

        self.floater = NodePath(PandaNode("floater"))
        self.floater.reparentTo(self.ralph)
        self.floater.setZ(2.0)

        # Accept the control keys for movement and rotation

        self.accept("escape", sys.exit)
        # self.accept("arrow_left", self.setKey, ["left", True])
        # self.accept("arrow_right", self.setKey, ["right", True])
        # self.accept("arrow_up", self.setKey, ["forward", True])
        self.accept("a", self.setKey, ["cam-left", True])
        self.accept("d", self.setKey, ["cam-right", True])
        self.accept("w", self.setKey, ["forward", True])
        self.accept("s", self.setKey, ["back", True])
        self.accept("r", self.setKey, ["up", True])
        self.accept("f", self.setKey, ["down", True])
        # self.accept("arrow_left-up", self.setKey, ["left", False])
        # self.accept("arrow_right-up", self.setKey, ["right", False])
        # self.accept("arrow_up-up", self.setKey, ["forward", False])
        self.accept("a-up", self.setKey, ["cam-left", False])
        self.accept("d-up", self.setKey, ["cam-right", False])
        self.accept("w-up", self.setKey, ["forward", False])
        self.accept("s-up", self.setKey, ["back", False])
        self.accept("r-up", self.setKey, ["up", False])
        self.accept("f-up", self.setKey, ["down", False])
        taskMgr.add(self.move, "moveTask")

        # Game state variables
        self.isMoving = False

        # Set up the camera
        self.disableMouse()
        self.camera.setPos(self.ralph.getX(), self.ralph.getY() + 10, 2)

        # We will detect the height of the terrain by creating a collision
        # ray and casting it downward toward the terrain.  One ray will
        # start above ralph's head, and the other will start above the camera.
        # A ray may hit the terrain, or it may hit a rock or a tree.  If it
        # hits the terrain, we can detect the height.  If it hits anything
        # else, we rule that the move is illegal.
        self.cTrav = CollisionTraverser()

        self.ralphGroundRay = CollisionRay()
        self.ralphGroundRay.setOrigin(0, 0, 9)
        self.ralphGroundRay.setDirection(0, 0, -1)
        self.ralphGroundCol = CollisionNode('ralphRay')
        self.ralphGroundCol.addSolid(self.ralphGroundRay)
        self.ralphGroundCol.setFromCollideMask(CollideMask.bit(0))
        self.ralphGroundCol.setIntoCollideMask(CollideMask.allOff())
        self.ralphGroundColNp = self.ralph.attachNewNode(self.ralphGroundCol)
        self.ralphGroundHandler = CollisionHandlerQueue()
        self.cTrav.addCollider(self.ralphGroundColNp, self.ralphGroundHandler)

        self.camGroundRay = CollisionRay()
        self.camGroundRay.setOrigin(0, 0, 9)
        self.camGroundRay.setDirection(0, 0, -1)
        self.camGroundCol = CollisionNode('camRay')
        self.camGroundCol.addSolid(self.camGroundRay)
        self.camGroundCol.setFromCollideMask(CollideMask.bit(0))
        self.camGroundCol.setIntoCollideMask(CollideMask.allOff())
        self.camGroundColNp = self.camera.attachNewNode(self.camGroundCol)
        self.camGroundHandler = CollisionHandlerQueue()
        self.cTrav.addCollider(self.camGroundColNp, self.camGroundHandler)

        # Uncomment this line to see the collision rays
        # self.ralphGroundColNp.show()
        # self.camGroundColNp.show()

        # Uncomment this line to show a visual representation of the
        # collisions occuring
        # self.cTrav.showCollisions(render)

        # Create some lighting
        ambientLight = AmbientLight("ambientLight")
        ambientLight.setColor((.3, .3, .3, 1))
        directionalLight = DirectionalLight("directionalLight")
        directionalLight.setDirection((-5, -5, -5))
        directionalLight.setColor((1, 1, 1, 1))
        directionalLight.setSpecularColor((1, 1, 1, 1))
        render.setLight(render.attachNewNode(ambientLight))
        render.setLight(render.attachNewNode(directionalLight))

        self.camLens.setFov(120)
        # Before estimation started, there are some startup works to do.

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

        self.gameTask = taskMgr.add(self.gameLoop, "gameLoop")

        self.translation_vector = [[0, 0, 0]]
        self.x = 0
        self.y = 0
        self.z = 0

    # Records the state of the arrow keys

    def gameLoop(self, task):
        # Get the time elapsed since the next frame.  We need this for our
        # distance and velocity calculations.
        dt = globalClock.getDt()

        # If the ship is not alive, do nothing.  Tasks return Task.cont to
        # signify that the task should continue running. If Task.done were
        # returned instead, the task would be removed and would no longer be
        # called every frame.
        # if not self.alive:
        #    return Task.cont
        self.captureFrame()

        return Task.cont    # Since every return is Task.cont, the task will
        # continue indefinitely

    def captureFrame(self):
        # Read a frame.
        frame_got, frame = self.cap.read()
        if frame_got is False:
            return

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if self.video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
        facebox = self.mark_detector.extract_cnn_facebox(frame)

        # Any face found?
        if facebox is not None:

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            x1, y1, x2, y2 = facebox
            face_img = frame[y1: y2, x1: x2]

            # Run the detection.
            self.tm.start()
            marks = self.mark_detector.detect_marks(face_img)
            self.tm.stop()

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
            # self.camera.setX(self.translation_vector[0] * scale)
            # self.camera.setY(self.translation_vector[2] * scale)
            # self.camera.setZ(self.translation_vector[1] * scale + 2)

            self.camera.setPosHpr(self.translation_vector[0] * scale * -1 + self.x,
                                  self.translation_vector[2] * scale + self.y,
                                  self.translation_vector[1] *
                                  scale + 20 + self.z,
                                  0.0,
                                  0.0,
                                  0.0)

            # self.camera.lookAt(self.floater)

            # Do you want to see the head axes?
            # self.pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            # self.mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Do you want to see the facebox?
            # self.mark_detector.draw_box(frame, [facebox])

        # Show preview.
        # cv2.imshow("Preview", frame)

    def setKey(self, key, value):
        self.keyMap[key] = value

    # Accepts arrow keys to move either the player or the menu cursor,
    # Also deals with grid checking and collision detection
    def move(self, task):

        # Get the time that elapsed since last frame.  We multiply this with
        # the desired speed in order to find out with which distance to move
        # in order to achieve that desired speed.
        dt = globalClock.getDt()

        # If the camera-left key is pressed, move camera left.
        # If the camera-right key is pressed, move camera right.

        step = 100

        if self.keyMap["cam-left"]:
            self.x -= step * dt
            print((self.x, self.y, self.z))
        if self.keyMap["cam-right"]:
            self.x += step * dt
            print((self.x, self.y, self.z))

        if self.keyMap["forward"]:
            self.y += step * dt
            print((self.x, self.y, self.z))

        if self.keyMap["back"]:
            self.y -= step * dt
            print((self.x, self.y, self.z))

        if self.keyMap["down"]:
            self.z -= step * dt
            print((self.x, self.y, self.z))
        if self.keyMap["up"]:
            self.z += step * dt
            print((self.x, self.y, self.z))

        #    self.camera.setX(self.camera, +20 * dt)

        # save ralph's initial position so that we can restore it,
        # in case he falls off the map or runs into something.

        startpos = self.ralph.getPos()

        # If a move-key is pressed, move ralph in the specified direction.

        # if self.keyMap["left"]:
        #    self.ralph.setH(self.ralph.getH() + 300 * dt)
        # if self.keyMap["right"]:
        #    self.ralph.setH(self.ralph.getH() - 300 * dt)
        # if self.keyMap["forward"]:
        #    self.ralph.setY(self.ralph, -25 * dt)

        # If ralph is moving, loop the run animation.
        # If he is standing still, stop the animation.

        # if self.keyMap["forward"] or self.keyMap["left"] or self.keyMap["right"]:
        #    if self.isMoving is False:
        # self.ralph.loop("run")
        #        self.isMoving = True
        # else:
        #    if self.isMoving:
        #        self.ralph.stop()
        #        self.ralph.pose("walk", 5)
        #        self.isMoving = False

        # If the camera is too far from ralph, move it closer.
        # If the camera is too close to ralph, move it farther.

        camvec = self.ralph.getPos() - self.camera.getPos()
        # camvec.setZ(0)
        # camdist = camvec.length()
        # camvec.normalize()
        # if camdist > 10.0:
        #    self.camera.setPos(self.camera.getPos() + camvec * (camdist - 10))
        #    camdist = 10.0
        # if camdist < 5.0:
        #    self.camera.setPos(self.camera.getPos() - camvec * (5 - camdist))
        #    camdist = 5.0

        # Normally, we would have to call traverse() to check for collisions.
        # However, the class ShowBase that we inherit from has a task to do
        # this for us, if we assign a CollisionTraverser to self.cTrav.
        # self.cTrav.traverse(render)

        # Adjust ralph's Z coordinate.  If ralph's ray hit terrain,
        # update his Z. If it hit anything else, or didn't hit anything, put
        # him back where he was last frame.

        entries = list(self.ralphGroundHandler.entries)
        entries.sort(key=lambda x: x.getSurfacePoint(render).getZ())

        if len(entries) > 0 and entries[0].getIntoNode().name == "terrain":
            self.ralph.setZ(entries[0].getSurfacePoint(render).getZ())
        else:
            self.ralph.setPos(startpos)

        # Keep the camera at one foot above the terrain,
        # or two feet above ralph, whichever is greater.

        return task.cont


demo = RoamingRalphDemo()
demo.run()
