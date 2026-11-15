import cv2

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


marker0 = cv2.aruco.generateImageMarker(d, 0, 400)
marker1 = cv2.aruco.generateImageMarker(d, 1, 400)
marker2 = cv2.aruco.generateImageMarker(d, 2, 400)
marker3 = cv2.aruco.generateImageMarker(d, 3, 400)
#marker4 = cv2.aruco.generateImageMarker(d, 4, 400)
marker5 = cv2.aruco.generateImageMarker(d, 5, 400)
marker6 = cv2.aruco.generateImageMarker(d, 7, 400)

ok0 = cv2.imwrite("marker0.png", marker0)
ok1 = cv2.imwrite("marker1.png", marker1)
ok2 = cv2.imwrite("marker2.png", marker2)
ok3 = cv2.imwrite("marker3.png", marker3)
#ok4 = cv2.imwrite("marker4.png", marker4)
ok5 = cv2.imwrite("marker5.png", marker5)
ok6 = cv2.imwrite("marker6.png", marker6)
    
if not (ok0 and ok1 and ok2 and ok3 and ok5 and ok6):
    print("These markers have NOT been saved, try again.")
else:
    print("These markers have been saved.")

