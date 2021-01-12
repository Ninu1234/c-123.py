import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
if (not os.environ.get('PYTHONHTTPSVERIFY','')and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_default_https_context = ssl._create_unverified_context()
    x = np.load('image.npz')['arr_0']
    y = pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts())
classes = ['0','1','2','3','4','5','6','7','8','9']
nclasses = len(classes)
xtrain,xtest,ytrain,ytest = train_test_split(X,y,random_state = 9,train_size = 7500,test_size = 2500)
xtrainscale = xtrain/255
xtestscale = xtest/255
clf = LogisticRegression(solver='saga',multi_class='multinomial').fit(xtrainscale,ytrain)
ypredict = clf.predict(xtestscale)
accuracy = accuracy_score(ytest,ypredict)
print(accuracy)
cap = cv2.VideoCapture(0)
while (True):
    try:
        ret,frame = cap.read()
        gray = cv.cvtcolor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape
        upper_left = (int (width/2 -56),int (height/2 -56))
        botton_right = (int (width/2 +56),int (height/2 +56))
        roi = gray[upper_left[1]:botton_right[1],upper_left[0]:botton_right[0]]
        Image_PIL = Image.fromarray(roi)
        image_bw = Image_PIL.convert('L')
        image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_resize_inverter = PIL.ImageOps.invert(image_bw_resize)
        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resize_inverter,pixel_filter)
        image_bw_resize_inverter_scale = np.clip(image_bw_resize_inverter-min_pixel,0,255)
        max_pixel = np.max(image_bw_resize_inverter)
        image_bw_resize_inverter_scale = np.asarray(image_bw_resize_inverter_scale)/max_pixel
        test_sample = np.array(image_bw_resize_inverter_scale).reshape(1,784)
        test_prediction = clf.predict(test_sample)
        print(test_predict)
        cv2.imshow('frame',gray)
        if cv2.waitKey(1)&0xff == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()