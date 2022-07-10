import cv2
import numpy as np
from pyparsing import line
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



def isInside(point,x2y2):
   poly = Polygon(point)
   x2y2 = Point(x2y2)
   print("isInside:",poly.contains(x2y2))
   return poly.contains(x2y2)

class YoloDetect():
   def __init__(self,detect_class = "person"): #,frame_width=640, frame_height=480.,video_cap=cv2.VideoCapture(0)
      self.classnames_file = "config_w\classnames.txt"
      self.weights_file = "config_w\yolov4-tiny.weights"
      self.config_file = "config_w\yolov4-tiny.cfg"
      self.conf_threshold = 0.5
      self.nms_threshold = 0.4
      # self.video_cap = video_cap
      self.detect_class = detect_class
      # self.frame_width = self.video_cap.get(3)
      # self.frame_height = self.video_cap.get(4)
      self.scale = 1 / 255
      self.model = cv2.dnn.readNet(self.weights_file, self.config_file)
      self.classes = None
      self.output_layers = None
      self.read_class_file()
      self.get_output_layers()
      self.last_alert = None
      # self.alert_telegram_each = 15  # seconds

   def read_class_file(self):
      with open(self.classnames_file,'r') as f:
         self.classes = [line.strip() for line in f.readlines()]
         
   def get_output_layers(self):
      layer_names = self.model.getLayerNames()
      self.output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]  
   
   
   def draw_pred(self,img,class_id,x1,y1,x2,y2,points):
      '''
      Ve bbox cho doi tuong.
      '''
      label = str(self.classes[class_id])
      color = (0,255.0)
      cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
      cv2.putText(img,label,(x1-10,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

      center_ = ((x1 + x2) // 2, (y1 + y2) // 2)
      cv2.circle(img,center_,5,(color),-1)
      if isInside(points,center_):
         img = cv2.putText(img, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      return isInside(points,center_)
   
   def detect(self,video_cap,frame,points):
      # Blob: duoc su dung de trich xuat feature from Image and resize them:
      #320×320 it’s small so less accuracy but better speed
      #609×609 it’s bigger so high accuracy and slow speed
      #416×416 it’s in the middle and you get a bit of both.
      blob = cv2.dnn.blobFromImage(frame,self.scale,(416,416),(0,0,0),True,crop=False)
      self.model.setInput(blob)
      outs = self.model.forward(self.output_layers)
      
      class_ids = []
      confidences = []
      boxes = []
      frame_width = video_cap.get(3)
      frame_height = video_cap.get(4)
      for out in outs:
         for detection in out:
            # print("Detection:",detection)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence >= self.conf_threshold) and (self.classes[class_id] == self.detect_class):
               center_x = int(detection[0] * frame_width)
               center_y = int(detection[1] * frame_height)
               w = int(detection[2] * frame_width)
               h = int(detection[3] * frame_height)
               x = center_x - w / 2
               y = center_y - h / 2
               class_ids.append(class_id)
               confidences.append(float(confidence))
               boxes.append([x, y, w, h])
      #su dung non-max de tranh hien tuong trong lan bbox
      indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

      for i in indices:
         box = boxes[i]
         print("box",box)
         x = box[0]
         y = box[1]
         w = box[2]
         h = box[3]
         # x, y, w, h = boxes[i]
         print("box:",box)
         self.draw_pred(frame, class_ids[i], round(x), round(y), round(x+w), round(y+h), points)
      return frame
       
       



