import os
import subprocess
import mujoco.viewer

import matplotlib.pyplot as plt
# Other imports and helper functions
import time
import itertools
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
      

# reset state and time

mujoco.mj_resetData(model, data_m)
mujoco.mj_forward(model, data_m)
    
if __name__ == "__main__":
  points = []
  stroke_starts = set()
  
  with mujoco.viewer.launch_passive(model, data_m) as viewer:
    while viewer.is_running():
      t = time.time()
      
      # make point from t: 0.3 * np.cost(t), 0.3 * np.sin(t), 0.005
      # this is recalculated and gives a new position each time since 
      # it is recalculated on every pass of the loop.
      
      point = [0.3 * np.cos(t), 0.3 * np.sin(t), .001]
      
      # moves sphere to that point
      
      data_m.mocap_pos[0] = point
      
      points.append(point)
      modify_scene(viewer.user_scn, points, stroke_starts)
      
      mujoco.mj_forward(model, data_m)
      
      
      viewer.sync()
      time.sleep(0.01)
    

