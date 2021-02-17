import cv2

imgOriginal = cv2.imread('Resources/wick.jpg')


scale_percent=80
width=int(imgOriginal.shape[1]*scale_percent/100)
height=int(imgOriginal.shape[0]*scale_percent/100)
dim=(width,height)
img=cv2.resize(imgOriginal,dim,interpolation=cv2.INTER_AREA)


thresholdVal=0.6

classNames=[] #burda 0.indexdeki coconames de 1 olarak geçer
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configurationPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='frozen_inference_graph.pb'


#bounding box and name
net = cv2.dnn_DetectionModel(weightsPath,configurationPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


classIds,confidents,boundingBoxs = net.detect(img,confThreshold=thresholdVal)

if len(classIds) !=0:
    for classId,confidence,boundingBox in zip(classIds.flatten(),confidents.flatten(),boundingBoxs):
            cv2.rectangle(img,boundingBox,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1],(boundingBox[0]+15,boundingBox[1]+10),
                        cv2.FONT_HERSHEY_DUPLEX,.5,(0,255,0),thickness=1)
            cv2.putText(img, str(confidence), (boundingBox[0] + 15, boundingBox[1] + 50),
                        cv2.FONT_HERSHEY_DUPLEX, .5, (0, 255, 0), thickness=1)

cv2.putText(img,"Huseyin Semih Ozker",(0,50),cv2.FONT_ITALIC,1,(255,255,0),thickness=3)



cv2.imshow('Nesne Algilama 18290050',img)

cv2.waitKey(0)