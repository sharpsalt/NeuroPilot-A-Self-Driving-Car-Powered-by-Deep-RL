import os
import cv2
from src.models import model
from subprocess import call

from ultralytics import YOLO
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class SteeringAnglePrediction:
    def __init__(self,model_path):
        self.session=tf.InteractiveSession()
        self.saver=tf.train.Saver()
        self.saver.restore(self.session,model_path)
        self.smoothed_angle=0

    def predict_angle(self,image):
        degrees=model.y.eval(feed_dict={model,x:[image],model.keep_prob: 1.0})[0][0]
        return degrees
    def smooth_angle(self,predicted_angle):
        if self.smoothed_angle==0:
            self.smoothed_angle=predicted_angle
        else:
            self.smoothed_angle+=0.2*pow(abs(predicted_angle-self.smoothed_angle),
                                         predicted_angle-self.smoothed_angle)/abs(predicted_angle-self.)
        return self.smoothed_angle
    def close(self):
        