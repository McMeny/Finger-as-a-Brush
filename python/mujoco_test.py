
import mujoco.viewer

import matplotlib.pyplot as plt
# Other imports and helper functions
import time
import numpy as np

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

xml = """
<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
    rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"
    reflectance=".2"/>
  </asset>
  
  <worldbody>
    <light name = "top" pos = "0 0 1" mode = "trackcom"/>
    <geom name="floor" type="plane" pos="0 0 0" size="2 2 .02" material="grid"/>
    <body name = "finger" mocap = "true">
      <geom name="green_cylinder" type = "cylinder" pos="0 0 .002" size=".025 .001" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# Make model and data_m
model = mujoco.MjModel.from_xml_string(xml)
data_m = mujoco.MjData(model)

def add_disc(scene, pos, radius, rgba):
  # for memory allocation
  
  if scene.ngeom >= scene.maxgeom:
    return
  
  scene.ngeom += 1
  
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                      mujoco.mjtGeom.mjGEOM_CYLINDER,
                      np.array([radius, radius, 0.001]),
                      np.array(pos, dtype = float),
                      np.eye(3).flatten(),
                      rgba.astype(np.float32))
  
def add_visual_capsule(scene, point1, point2, radius, rgba):
  if scene.ngeom >= scene.maxgeom:
    return
  
  scene.ngeom += 1
  
  # Initialize a new capsule, and add it to the scene using mjv_connector
  
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  
  mujoco.mjv_connector(scene.geoms[scene.ngeom - 1],
                       mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                       point1, point2)

def modify_scene(scn, pts, stroke_starts):
  if len(pts) < 2:
    return
  
  if len(pts) - 1 in stroke_starts:
    return
  
  rgba = np.array([0, 0, 1, 1])
  radius = 0.02
  
  # for drawing capsules: add_visual_capsule(scn, pts[-2], pts[-1], radius, rgba)
  for p in np.linspace(pts[-2], pts[-1], 4):
    add_disc(scn, p, radius, rgba)


def add_box(scene, pos, half_size, angle, rgba):
  # flat square lying on the floor, rotated by angle (radians) around the z axis
  if scene.ngeom >= scene.maxgeom:
    return

  scene.ngeom += 1

  c, s = np.cos(angle), np.sin(angle)
  rot = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])

  mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                      mujoco.mjtGeom.mjGEOM_BOX,
                      np.array([half_size, half_size, 0.002]),
                      np.array(pos, dtype = float),
                      rot.flatten(),
                      rgba.astype(np.float32))


def order_outline(points):
  # Sort points by their angle around the middle point, so consecutive points
  # are neighbours around the shape (no diagonals). Works for convex shapes
  # like a rectangle of markers. Only the first two columns (x, y) are used.
  points = np.asarray(points, dtype = float)
  middle = points.mean(axis = 0)
  angles = np.arctan2(points[:, 1] - middle[1], points[:, 0] - middle[0])
  return points[np.argsort(angles)]


def draw_ground_truth(scn, markers):
  # markers: list of (pos, half_size, angle), one per reference ArUco marker.
  # Clears the user scene and draws the marker boxes + outline once. The path
  # discs from modify_scene are then added on top of these geoms.
  scn.ngeom = 0

  box_rgba = np.array([0.1, 0.1, 0.1, 1])
  outline_rgba = np.array([1, 0, 0, 1])

  for pos, half_size, angle in markers:
    add_box(scn, pos, half_size, angle, box_rgba)

  # outline: connect each point to the next, and the last back to the first
  if len(markers) >= 2:
    ordered = order_outline([m[0] for m in markers])
    for i in range(len(ordered)):
      p1 = ordered[i] + [0, 0, 0.003]
      p2 = ordered[(i + 1) % len(ordered)] + [0, 0, 0.003]
      add_visual_capsule(scn, p1, p2, 0.008, outline_rgba)


# reset state and time

mujoco.mj_resetData(model, data_m)
mujoco.mj_forward(model, data_m)
