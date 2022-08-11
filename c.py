import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os,ssl,time
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")['labels']
print(pd.Series(y).value_counts() )
classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

nclasses=len(classes)

xtrain,xtest,ytrain,ytest=train_test_split(X,y,random_state=9,train_size=3500,test_size=1500)
xtrainScaled=xtrain/255.0
xtestScaled=xtest/255.0

clf=LogisticRegression(solver='saga',multi_class="multinomial").fit(xtrainScaled,ytrain)
y_predict=clf.predict(xtestScaled)
acc = accuracy_score(ytest,y_predict)
print(acc)
time.sleep(10)

capture=cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=capture.read()
        print("in the loop")
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY())
        height,width=gray.shape()
        upperLeft=(int(width/2-55),int(height/2-55))
        bottomRight=(int(width/2+55),int(height/2+55))
        cv2.rectangle(gray,upperLeft,bottomRight,(0,255,0),2)
        roi=gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]

        imagePIL=Image.fromarray(roi)
        image_w=imagePIL.convert('L')
        image_w_resized=image_w.resize((28,28),Image.ANTIALIAS)
        
        image_bw_resized_inverted = PIL.ImageOps.invert(image_w_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

  
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
capture.release()
cv2.destroyAllWindows()