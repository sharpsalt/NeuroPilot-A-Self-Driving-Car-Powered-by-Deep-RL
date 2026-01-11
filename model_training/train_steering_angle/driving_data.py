import cv2
import os
import random
import numpy as np

# Base directory resolution (robust path handling)
BASE_PATH=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/driving_dataset"))

xs=[]
ys=[]

train_batch_pointer=0
val_batch_pointer=0

with open(os.path.join(BASE_PATH,"data.txt")) as f:
    for line in f:
        img_file=line.split()[0] 
        angle_str=line.split()[1].split(',')[0]
        xs.append(os.path.join(BASE_PATH, img_file))
        #the paper by Nvidia uses the inveses of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as outputs(pi/180)
        ys.append(float(angle_str) * np.pi /180)

#get number of images
num_images=len(xs)
#shuffle list of images
c=list(zip(xs, ys))
# random.shuffle(c)
xs,ys=zip(*c)

train_xs=xs[:int(len(xs)*0.8)]
train_ys=ys[:int(len(xs)*0.8)]

val_xs=xs[-int(len(xs)*0.2):]
val_ys=ys[-int(len(xs)*0.2):]

num_train_images=len(train_xs)
num_val_images=len(val_xs)



def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out=[]
    y_out=[]
    for i in range(batch_size):
        y_out.append([train_ys[(train_batch_pointer+i)%num_train_images]])
        image_path=train_xs[(train_batch_pointer+i)%num_train_images]
        img=cv2.imread(image_path)
    #    img=cv2.resize(img,(200,66))
        if img is None:
            print(f"Skkipping image {image_path} as it is missing")
            continue #we will skip this image 
        img=img[-150:] #crop the image(90 pixels from top)
        img=cv2.resize(img,(200,66))/255.0
        x_out.append(img)
        y_out.append([train_ys[(train_batch_pointer+i)%num_train_images]])

    train_batch_pointer+=batch_size
    return x_out,y_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        image_path = val_xs[(val_batch_pointer + i) % num_val_images]
        img = cv2.imread(image_path)
        if img is None:
            print(f"Skipping image {image_path} as it is missing")
            continue  # we will skip this image
        img = img[-150:]  # crop the image(90 pixels from top)
        img = cv2.resize(img, (200, 66)) / 255.0
        x_out.append(img)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
        #same hi krna hai ismein bhi
    val_batch_pointer += batch_size
    return x_out, y_out
