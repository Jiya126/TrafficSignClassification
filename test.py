from pyexpat import model
from statistics import mode
import cv2
import numpy as np
import pickle

# parameters
brightness = 180
threshold = 0.75

# setup video
c = cv2.VideoCapture(0)
c.set(10, brightness)

# import trained model
pickle_in = open('model trained.p','rb')
model = pickle.load(pickle_in)
print(model)

# process image
def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img = cv2.equalizeHist(img)
    return img
def processing(img):
    img = gray(img)
    img = equalize(img)
    img = img/255
    return img

# get classes names
def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'


while True:
    success, img = c.read()

    img = np.asarray(img)
    img = processing(img)
    cv2.imshow("processed image", img)
    cv2.putText(img, 'CLASS: ', (20,35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 3, cv2.LINE_AA)
    cv2.putText(img, 'PROBABILITY: ', (20,75), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 3, cv2.LINE_AA)
    
    # predict image
    prediction = model.predict(img)
    classNo = model.predict_classes(img)
    probabValue = np.amax(prediction)
    
    if probabValue > threshold:
        cv2.putText(img, str(classNo)+"  "+str(getClassName(classNo)), (120,35), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, str(round(probabValue*100 + 2)+"  "+"%"), (180, 75), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break