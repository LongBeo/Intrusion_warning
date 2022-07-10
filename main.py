"""
Xay dung he thong canh bao xam nhap vung cam: YOLO + OpenCV
"""
import cv2
from imutils.video import VideoStream
# from matplotlib.colors import PowerNorm
import numpy as np
from yolodetect import YoloDetect


# video_cap = VideoStream(src=0).start()
video_cap = cv2.VideoCapture(0)
# w = 1280
# h = 920
# dim = (w,h)
points=[]
model = YoloDetect()
detect = False

def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

def draw_polygon(frame,points):
   for point in points:
      frame = cv2.circle(frame,(point[0],point[1]),5,(0,0,255),-1)
   # frame = cv2.polylines(frame,[np.int32(points)],False,(255,0,0))
   frame = cv2.polylines(frame,[np.int32(points)],False,(255,0,0))
   return frame

while True:
   res,frame = video_cap.read()
   # print(video_cap.get(3))
   # print(video_cap.get(4))
   frame = cv2.flip(frame,1)
   # frame = cv2.resize(frame,dim)
   frame = draw_polygon(frame,points)
   
   # model = YoloDetect()
   if detect:
      frame = model.detect(video_cap=video_cap,frame=frame,points=points)
   
   cv2.imshow("Intrusion Warning",frame)
   
   key = cv2.waitKey(1)
   if key == ord("q"):
      break
   elif key == ord("d"):
      points.append(points[0])
      detect = True
   cv2.setMouseCallback('Intrusion Warning', handle_left_click,points)

# video_cap.stop()   
video_cap.release()
cv2.destroyAllWindows()
   

