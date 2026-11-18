# Accuracy test for the finger brush.
#
# The reference ArUco markers on the paper form the ground truth outline.
# The marker on the finger (TRACKED_ID) is the brush: while the FSR says the
# finger is touching, its centre is recorded as the drawn path.
# The RMSE is the root mean square of each drawn point's shortest distance to
# the ground truth outline, in millimetres.
#
# The ground truth is captured during the first ~2 seconds and then stays fixed.
#
# Keys (in the OpenCV window):  q = quit and save,  s = save,  c = clear the path (new trial),
#                               g = re-capture the ground truth

from mujoco_test import model, data_m, modify_scene, draw_ground_truth, order_outline

import os
import csv
import time
import mujoco
import mujoco.viewer
import numpy as np

import cv2
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
    1: 0.032,
    2: 0.032,
    3: 0.032,
    4: 0.032,
    5: 0.032,
}

TRACKED_ID = 2          # marker on the finger = the brush
FSR_THRESHOLD = 1000    # FSR reading above this counts as touching

MARKER_SPACING_MM = 150.0   # centre-to-centre distance between neighbouring reference markers
TOLERANCE_MM = 5.0          # a drawn point counts as "on the line" within this distance

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# camera pixels -> MuJoCo metres (same scale as the paint trail: 0.002 m per pixel,
# image centre (320, 240) at the origin, image y flipped so up is +y)
PX_TO_M = 0.002
def px_to_world(x, y, z = 0.002):
    return [(x - 320) * PX_TO_M, (240 - y) * PX_TO_M, z]


# ---------- RMSE ----------

def distances_to_outline(points, outline):
    # Shortest distance from each point to the closed outline (in pixels).
    # For every edge A->B, project the point onto the edge, clamp it between
    # A and B, and measure the distance; keep the smallest over all edges.
    points = np.asarray(points, dtype = float)
    best = np.full(len(points), np.inf)
    for i in range(len(outline)):
        a = outline[i]
        b = outline[(i + 1) % len(outline)]
        ab = b - a
        length_sq = np.dot(ab, ab)
        if length_sq == 0:
            t = np.zeros(len(points))
        else:
            t = np.clip((points - a) @ ab / length_sq, 0, 1)
        closest = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(points - closest, axis = 1))
    return best


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def accuracy_pct(errors_mm):
    # RMSE as a fraction of the marker spacing, turned into a score:
    # 100% = drawn exactly on the line, 0% = RMSE as big as a whole side (150 mm)
    return max(0.0, 100 * (1 - rmse(errors_mm) / MARKER_SPACING_MM))


def within_tolerance_pct(errors_mm):
    # percentage of drawn points within TOLERANCE_MM of the ground truth line
    return 100 * float(np.mean(np.asarray(errors_mm) <= TOLERANCE_MM))


def save_results(trail, stroke_ids, outline, mm_per_px):
    if not trail or outline is None:
        print("Nothing to save (need a drawn path and at least 2 reference markers).")
        return

    os.makedirs(RESULTS_DIR, exist_ok = True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    errors_mm = distances_to_outline(trail, outline) * mm_per_px

    # every drawn point with its error
    points_file = os.path.join(RESULTS_DIR, f"accuracy_{stamp}.csv")
    with open(points_file, "w", newline = "") as f:
        w = csv.writer(f)
        w.writerow(["stroke", "x_px", "y_px", "error_mm"])
        for (x, y), s, e in zip(trail, stroke_ids, errors_mm):
            w.writerow([s, round(x, 2), round(y, 2), round(e, 3)])

    # one summary row per trial, appended to the same file
    summary_file = os.path.join(RESULTS_DIR, "summary.csv")
    header = ["time", "points", "strokes", "accuracy_pct", f"within_{TOLERANCE_MM:g}mm_pct",
              "rmse_mm", "mean_mm", "max_mm", "marker_spacing_mm", "mm_per_px"]
    # if an older summary.csv has different columns, keep it under another name
    if os.path.exists(summary_file):
        with open(summary_file, newline = "") as f:
            old_header = next(csv.reader(f), None)
        if old_header != header:
            os.rename(summary_file, os.path.join(RESULTS_DIR, f"summary_old_{stamp}.csv"))
    new_file = not os.path.exists(summary_file)
    with open(summary_file, "a", newline = "") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow([stamp, len(trail), len(set(stroke_ids)),
                    round(accuracy_pct(errors_mm), 2), round(within_tolerance_pct(errors_mm), 1),
                    round(rmse(errors_mm), 3), round(float(errors_mm.mean()), 3),
                    round(float(errors_mm.max()), 3), MARKER_SPACING_MM, round(mm_per_px, 4)])

    print(f"Accuracy {accuracy_pct(errors_mm):.1f}% | "
          f"within {TOLERANCE_MM:g} mm: {within_tolerance_pct(errors_mm):.0f}% | "
          f"RMSE {rmse(errors_mm):.2f} mm | mean {errors_mm.mean():.2f} mm | "
          f"max {errors_mm.max():.2f} mm | {len(trail)} points | "
          f"markers {MARKER_SPACING_MM:g} mm apart")
    print(f"Saved {points_file}")


# ---------- state ----------

trail = []              # brush centre (pixels) while touching
stroke_ids = []         # which stroke each trail point belongs to
stroke_starts = set()   # trail indices where a new stroke begins
stroke_count = 0
touching = False
was_touching = False
unsaved = False         # path has points that haven't been saved yet

# Ground truth is captured once at start-up and then locked for the whole session:
# for the first GT_FRAMES frames every reference marker detection is collected,
# then each marker's position is averaged and the outline is frozen.
# Press "g" to re-capture it (e.g. after moving the paper).
GT_FRAMES = 60          # about 2 seconds at 30 fps
gt_samples = {}         # id -> list of (centre px, side px, angle) while capturing
gt_frames_seen = 0
gt_locked = False

reference = {}          # locked reference markers: id -> (centre px, side px, angle)
outline = None          # locked ground truth outline in pixels (ordered corners)
mm_per_px = None        # locked pixel -> mm scale
gt_ngeom = 0            # number of MuJoCo geoms used by the ground truth
live_errors = None      # per-point errors (mm) for the live RMSE display


def lock_ground_truth(samples, frames_seen):
    # Average every marker that was seen in at least half the capture frames
    # (ignores one-off false detections). Returns (reference, outline, mm_per_px),
    # or None if fewer than 2 markers qualify.
    ref = {}
    for marker_id, s in samples.items():
        if len(s) < frames_seen / 2:
            continue
        centres = np.array([c for c, _, _ in s])
        sides = np.array([side for _, side, _ in s])
        angles = np.array([a for _, _, a in s])
        # average angles via sin/cos so -179 deg and +179 deg don't average to 0
        mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
        ref[marker_id] = (centres.mean(axis = 0), float(sides.mean()), float(mean_angle))

    if len(ref) < 2:
        return None

    out = order_outline([c for c, _, _ in ref.values()])

    # pixel -> mm scale from the known marker spacing: every outline edge is
    # MARKER_SPACING_MM long in real life, so divide by its average length in pixels
    edges_px = np.linalg.norm(out - np.roll(out, -1, axis = 0), axis = 1)
    scale = MARKER_SPACING_MM / float(edges_px.mean())

    # sanity check: the same edges measured using the printed marker size instead
    marker_scale = float(np.mean([marker_lengths.get(i, 0.032) * 1000 / side
                                  for i, (_, side, _) in ref.items()]))
    print("Outline edges using the marker-size scale: "
          f"{np.round(edges_px * marker_scale, 1).tolist()} mm "
          f"(should all be close to {MARKER_SPACING_MM:g} mm)")
    return ref, out, scale

viewer = mujoco.viewer.launch_passive(model, data_m)

viewer.cam.lookat[:] = [0, 0, 0]
viewer.cam.distance = 2
viewer.cam.elevation = -89.9
viewer.cam.azimuth = 90

while(True):

    ret, frame = cap.read()
    if not ret:
        break

    # ---- FSR: is the finger touching the paper? ----
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith(">fsr:"):
            value = line.split(":")[1]
            if value.isdigit():
                touching = int(value) > FSR_THRESHOLD

    if touching and not was_touching:
        stroke_starts.add(len(trail))
        stroke_count += 1
    was_touching = touching

    old_len = len(trail)

    # ---- markers ----
    corners, ids, rejected = detector.detectMarkers(frame)

    brush_px = None
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for marker_corner, marker_id in zip(corners, ids.flatten()):
            part = marker_corner.reshape((4, 2))   # each marker piece with 4 corners
            (top_left, top_right, bottom_right, bottom_left) = part

            if marker_id == TRACKED_ID:
                # finger tracking, exactly as in paint_mujoco.py:
                # corners rounded to whole pixels, centre = midpoint of the
                # top-left -> bottom-right diagonal, point added while touching
                top_left_i = (int(top_left[0]), int(top_left[1]))
                bottom_right_i = (int(bottom_right[0]), int(bottom_right[1]))
                center_x = int((top_left_i[0] + bottom_right_i[0]) / 2.0)
                center_y = int((top_left_i[1] + bottom_right_i[1]) / 2.0)
                brush_px = (center_x, center_y)

                # only record the path once the ground truth is locked
                if gt_locked and touching:
                    trail.append([center_x, center_y])
                    stroke_ids.append(stroke_count)
                    unsaved = True

            elif not gt_locked:
                center = part.mean(axis = 0)
                side_px = np.linalg.norm(part - np.roll(part, -1, axis = 0), axis = 1).mean()
                dx, dy = top_right - top_left          # top edge
                angle = np.arctan2(-dy, dx)            # minus dy because image y points down
                gt_samples.setdefault(int(marker_id), []).append((center, side_px, angle))

        # pose of every marker (axes drawn on the image), once per frame
        for marker_corner, marker_id in zip(corners, ids.flatten()):
            if marker_id not in marker_lengths:
                continue
            half = marker_lengths[marker_id] / 2
            marker_points = np.array([[-half, half, 0], [half, half, 0],
                                      [half, -half, 0], [-half, -half, 0]], dtype = np.float32)
            ok, rvec, tvec = cv2.solvePnP(marker_points, marker_corner,
                                          camera_matrix, dist_coeffs,
                                          flags = cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, half)

    # ---- ground truth capture (start-up only) ----
    if not gt_locked:
        gt_frames_seen += 1
        if gt_frames_seen >= GT_FRAMES:
            result = lock_ground_truth(gt_samples, gt_frames_seen)
            if result is not None:
                reference, outline, mm_per_px = result
                gt_locked = True
                print(f"Ground truth locked: markers {sorted(reference)}, "
                      f"{mm_per_px:.4f} mm/px")

                # draw the locked ground truth into MuJoCo once; it never changes,
                # so the path can be added on top of it like in paint_mujoco.py
                mj_markers = [(px_to_world(c[0], c[1]), side / 2 * PX_TO_M, ang)
                              for c, side, ang in reference.values()]
                with viewer.lock():
                    draw_ground_truth(viewer.user_scn, mj_markers)
                    gt_ngeom = viewer.user_scn.ngeom
            else:
                # not enough markers seen yet: start the capture window again
                gt_samples.clear()
                gt_frames_seen = 0

    # ---- OpenCV drawing ----
    if outline is not None:
        cv2.polylines(frame, [outline.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 2)

    for i in range(1, len(trail)):
        if i in stroke_starts:
            continue
        cv2.line(frame, trail[i-1], trail[i], (255, 0, 0), 3)

    if brush_px is not None:
        color = (0, 255, 255) if touching else (200, 200, 200)
        cv2.circle(frame, brush_px, 6, color, -1)

    # live readings, only recomputed when new points were added
    if trail and outline is not None:
        if len(trail) != old_len or live_errors is None or len(live_errors) != len(trail):
            live_errors = distances_to_outline(trail, outline) * mm_per_px
        status_lines = [
            f"Accuracy {accuracy_pct(live_errors):.1f}%   "
            f"within {TOLERANCE_MM:g} mm: {within_tolerance_pct(live_errors):.0f}%",
            f"RMSE {rmse(live_errors):.2f} mm  max {live_errors.max():.2f} mm  n={len(trail)}",
            f"Ground truth: markers {MARKER_SPACING_MM:g} mm apart",
        ]
    elif not gt_locked:
        status_lines = [f"Capturing ground truth {gt_frames_seen}/{GT_FRAMES} - keep hand away "
                        f"(markers: {len(gt_samples)})"]
    else:
        status_lines = [f"Ground truth locked ({len(reference)} markers, "
                        f"{MARKER_SPACING_MM:g} mm apart) - draw!"]
    for row, text in enumerate(status_lines):
        y = 25 + row * 22
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ---- MuJoCo (same as paint_mujoco.py) ----
    if trail:
        points = []
        for x, y in trail:
            points.append([(x - 320) * PX_TO_M, (240 - y) * PX_TO_M, 0.001])

        data_m.mocap_pos[0] = points[-1]
        mujoco.mj_forward(model, data_m)

        # only the newest segment is added; earlier discs stay in the scene
        if len(trail) > old_len:
            with viewer.lock():
                modify_scene(viewer.user_scn, points, stroke_starts)

    viewer.sync()

    cv2.imshow("ArUco detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        save_results(trail, stroke_ids, outline, mm_per_px)
        unsaved = False
    if key == ord("g"):
        # re-capture the ground truth (also clears the path, since the old
        # path was measured against the old ground truth)
        gt_locked = False
        gt_samples.clear()
        gt_frames_seen = 0
        reference, outline, mm_per_px = {}, None, None
        gt_ngeom = 0
        key = ord("c")
        print("Re-capturing ground truth")
    if key == ord("c"):
        trail.clear()
        stroke_ids.clear()
        stroke_starts.clear()
        stroke_count = 0
        was_touching = False
        unsaved = False
        live_errors = None
        # remove the path discs from MuJoCo, keep the ground truth geoms
        with viewer.lock():
            viewer.user_scn.ngeom = gt_ngeom
        print("Path cleared")
    if not viewer.is_running():
        break

# save on exit, unless this path was already saved with "s"
if unsaved:
    save_results(trail, stroke_ids, outline, mm_per_px)

viewer.close()
cap.release()
cv2.destroyAllWindows()
