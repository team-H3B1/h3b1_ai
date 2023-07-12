import cv2
import mediapipe as mp
import numpy as np
import math
import knn.predict as predict

BG_COLOR = (192, 192, 192) # gray
class MediaPipe:

  def __init__(self) -> None:
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_drawing_styles = mp.solutions.drawing_styles
    self.mp_pose = mp.solutions.pose
    
  def calF1F2(self, results):
    nose_xyz = np.array([
      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y
    ])

    foot_xyz = np.array([
      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
    ])

    nose_x_foot_y = np.array(([
      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x,
      results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
    ])) 
    o1 = math.atan((nose_xyz[0]-foot_xyz[0]) / (nose_xyz[1]-foot_xyz[1]))
    o2 = math.atan((nose_xyz[0]-nose_x_foot_y[0]) / (nose_xyz[1]-nose_x_foot_y[1]))
    
    f1 = np.linalg.norm(nose_xyz - nose_x_foot_y) / np.linalg.norm(nose_xyz - foot_xyz)
    f2 = abs((o1 - o2) * 180 / math.pi) / 90
    
    return f1, f2

  def getPose(self, image):
      with self.mp_pose.Pose(
      static_image_mode=True,
      model_complexity=2,
      enable_segmentation=True,
      min_detection_confidence=0.5) as pose:
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks: return image, "no human"

        """f1, f2 cal"""
        f1, f2 = self.calF1F2(results)
        state = predict.knn_predict(f1, f2)
        
        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose landmarks on the image.
        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec= self.mp_drawing_styles.get_default_pose_landmarks_style())
        
        return annotated_image, state
        # result : annotated_image