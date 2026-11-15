from mujoco_test import model, data_m, modify_scene

import os
import subprocess
import mujoco.viewer

import matplotlib.pyplot as plt
# Other imports and helper functions
import time
import itertools
import numpy as np

import cv2
#import open3d as o3d
import serial

ser = serial.Serial("COM15", 115200, timeout = 0.05)



d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

data = np.load("calibration.npz")
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

detector = cv2.aruco.ArucoDetector(d)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

marker_lengths = {
    0: 0.032,
    1: 0.1,
    2: 0.032,
    3: 0.032,
    4: 0.032,
    5: 0.032,
}

TRACKED_ID = 2
trail = []
touching = False

#vis = o3d.visualization.Visualizer()
#vis.create_window()
#pcd = o3d.geometry.PointCloud()

#border = o3d.geometry.LineSet(
#    points = o3d.utility.Vector3dVector([[0,0,0], [640, 0, 0], [640, -480, 0], [0, -480, 0]]),
#    lines = o3d.utility.Vector2iVector([[0,1], [1,2], [2,3], [3,0]]))
#border.paint_uniform_color([0.8, 0.8, 0.8])
#vis.add_geometry(border)

#line_set = o3d.geometry.LineSet()
#vis.add_geometry(line_set, reset_bounding_box=False)
#vis.add_geometry(pcd, reset_bounding_box=False)
was_touching = False
stroke_starts = set()

viewer = mujoco.viewer.launch_passive(model, data_m)

viewer.cam.lookat[:] = [0, 0, 0]
viewer.cam.distance = 2
viewer.cam.elevation = -89.9
viewer.cam.azimuth = 90

while(True):
    
    ret, frame = cap.read()
    
    old_len = len(trail)
    
    if not ret:
        break
    
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith(">fsr:"):
            value = line.split(":")[1]
            if value.isdigit():
                touching = int(value) > 1000
                #print(value, "TOUCHING" if touching else "")
                
    if touching and not was_touching:
        stroke_starts.add(len(trail))
    was_touching = touching
    
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for (marker_corner, marker_id) in zip(corners, ids):
            part = marker_corner.reshape((4, 2)) # each marker piece with 4 corners
            #print(corners)
            #print(marker_id)
            (top_left, top_right, bottom_right, bottom_left) = part
            
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            #print(top_left, top_right, bottom_right, bottom_left)
            
            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
            
            
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            if int(marker_id == TRACKED_ID) and touching:
                trail.append([center_x, center_y])

            cv2.putText(frame, str(marker_id),
                        (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                        )
        
        poses = {}
        # go through every detected marker, where marker_corner is its 4 corner pixels and marker_id is the id number of marker
        for marker_corner, marker_id in zip(corners, ids.flatten()): # flatten collapses ids shape from (N,1) to (N,)
               
            # if ID isnt in your marker_lengths dictionary, it skips the marker.
            if marker_id not in marker_lengths: 
                continue
                
            # for describing the physical marker size
            # for picturing the marker lying flat with its centre at (0,0,0).
            length = marker_lengths[marker_id]
            half = length / 2
            marker_points = np.array([[-half, half, 0], [half, half, 0],
                                          [half, -half, 0], [-half, -half, 0]], dtype = np.float32)
                
            # ok: if it found a solution
            # rvec: marker's rotation as an axis-angle vector
            # tvec: position of the marker's centre in metres, measured from the camera. x is right, y is down, z points straight out of the lens.
            ok, rvec, tvec = cv2.solvePnP(marker_points, marker_corner, 
                                              camera_matrix, dist_coeffs,
                                              flags = cv2.SOLVEPNP_IPPE_SQUARE)
            
            if ok:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, half)
                poses[int(marker_id)] = (rvec, tvec)
                visible_ids = sorted(poses)
                #print('print', visible_ids)
                                
                if TRACKED_ID in visible_ids:
                    _, tvec2 = poses[2]
                    
                    for brick_id in visible_ids:
                        if brick_id == 1:
                            continue
                        _, tvec_b = poses[brick_id]
                        dist = np.linalg.norm(tvec_b - tvec2)
                        #print(f"marker 1 -> marker {brick_id}: {dist:.3f} m")
    
    for i in range(1, len(trail)):
        if i in stroke_starts:
            continue
        cv2.line(frame, trail[i-1], trail[i], (0,0,255), 3)
                
    if trail:
        points = []
        
        for x,y in trail:
            points.append([(x - 320) * 0.002, (240 - y) * 0.002, 0.001])
        
        data_m.mocap_pos[0] = points[-1]
        mujoco.mj_forward(model, data_m)
        
                
        if len(trail) > old_len:
            modify_scene(viewer.user_scn, points, stroke_starts)
         
        lines = []
        for i in range(1, len(points)):
            if i in stroke_starts:
                continue
            lines.append([i-1, i])
            
#        line_set.points = o3d.utility.Vector3dVector(points)
#        line_set.lines = o3d.utility.Vector2iVector(lines)
#        vis.update_geometry(line_set)
#        line_set.paint_uniform_color([0, 0, 1])
                
#    vis.poll_events()
    viewer.sync()
    
#    vis.update_renderer()
    
    cv2.imshow("ArUco detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

viewer.close()
#vis.destroy_window()

cap.release()
cv2.destroyAllWindows()
