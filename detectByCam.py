import cv2

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

thresholdVal=0.5

classNames=[] #burda 0.indexdeki coconames de 1 olarak geçer
classFile='coco.names' #algılanan nesnelerin isimleri
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configurationPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'



net = cv2.dnn_DetectionModel(weightsPath,configurationPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()

    classIds,confidents,boundingBoxs = net.detect(img,confThreshold=thresholdVal)

    if len(classIds) !=0:
        for classId,confidence,boundingBox in zip(classIds.flatten(),confidents.flatten(),boundingBoxs):
            cv2.rectangle(img,boundingBox,color=(0,255,0),thickness=5)
            cv2.putText(img,classNames[classId-1],
                        (boundingBox[0]+15,boundingBox[1]+20),
                        cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),
                        thickness=2)
            cv2.putText(img, str(confidence), (boundingBox[0] + 15, boundingBox[1] + 50),
                        cv2.FONT_HERSHEY_DUPLEX, .5, (0, 255, 0),
                        thickness=1)

    cv2.imshow('Nesne Algilama 18290050',img)
    cv2.waitKey(1)