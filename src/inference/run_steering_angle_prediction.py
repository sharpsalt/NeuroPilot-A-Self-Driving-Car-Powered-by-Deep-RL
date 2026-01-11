import os
from turtle import degrees
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
        self.model=model

    def predict_angle(self,image):
        radians=model.y.eval(feed_dict={model,x:[image],model.keep_prob: 1.0})[0][0]
        degrees=radians*180.0/3.14159265
        return degrees
    

    def smooth_angle(self,predicted_angle):
        if self.smoothed_angle==0:
            self.smoothed_angle=predicted_angle
        else:
            difference=predicted_angle-self.smoothed_angle
            if difference!=0:
                abs_differebce=abs(difference)
                scaled_difference=pow(abs_differebce,2./3.0)
                self.smoothed_angle+=(0.2*scaled_difference*(difference/abs_differebce))
            return self.smoothed_angle 
        
    def close(self):
        self.session.close()

class DrivingSimulator:
    def __init__(self,predictor,data_dir,steering_image_path,is_windows=False):
        self.predictor=predictor
        self.data_dir=data_dir
        self.steering_image_path=cv2.imread(steering_image_path) #because i ahve to lead the steering wheel image
        self.is_windows=is_windows

        if self.steering_image is None:
            raise ValueError(f"Could not load Steering wheel image from {steering_image_path}")
        
        #now my moto is to make it square such that i can resize it 
        height,width=self.steering_image_path.shape[:2] #from starting 2nd index tk hoga so i will slice it 
        size=max(height,width)
        self.steering_image=cv2.resize(self.steering_image,(size,size))
    
    def start_simulation(self):
        i=0
        while cv2.waitKey(10)!=ord('q'):
            full_image=cv2.imread(os.path.join(self.data_dir,f"{i}.jpg"))
            if full_image is None:
                print(f"Image {i}.jpg not found in {self.data_dir}, ending simulation.")
                break
            resized_image=cv2.resize(full_image[-150:],(200,66))/255.0
            predicted_angle=self.predictor.predict_angle(resized_image)
            smoothed_angle=self.predictor.smooth_angle(predicted_angle)
            if not self.is_windows:
                # os.system('clear')
                call('clear')
            print(f"Predicted steering angle: {predicted_angle:.2f} degrees")
            self.display_frames(full_image,smoothed_angle)
            i+=1
        cv2.destroyAllWindows()
    
    