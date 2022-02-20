import streamlit as st
from IPython.display import Image
from matplotlib import pyplot as plt
import glob
import cv2
import argparse
import sys
import numpy as np
import os
import os.path
from random import randint
from os import rename
from PIL import Image
import pandas as pd
import time
import pandas as pd
from PIL import  ImageDraw, ImageFont
import base64


confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4  #Non-maximum suppression threshold
inpWidth = 416      #Width of network's input image
inpHeight = 416     #Height of network's input image

Title_html = """
    <style>
        .title h1{
          user-select: none;
          font-size: 35px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
    </style> 
    
    <div class="title">
        <h1>Vehicle And License Plate Recognition With Novel Dataset For Toll Collection</h1>
    </div>
    """
st.markdown(Title_html, unsafe_allow_html=True) #Title rendering

#=====================  SPLIT CLASSES  ============================
@st.cache
def split_classes(classesFile):
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

#===================================================================

# ================= Get the names of the output layers =============
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#=======================================================================

#=====================  READING NETWORK  ============================
def network (modelConfiguration, modelWeights):
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

#======================================================================

def drawPred(classId, conf, left, top, right, bottom, frame,classes):
    # Draw a bounding box.
    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0 ), 12)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0 ), 7)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    top = max(top, labelSize[1])
    # cv2.rectangle(frame, (left, top - round(2*labelSize[1])), (left + round(2*labelSize[0]), top + baseLine), (255, 0, 0), cv2.FILLED)
    # cv2.rectangle(frame, (left, top-10 - round(3.5*labelSize[1])), (left + round(3.5*labelSize[0]), top + baseLine-15),    (255, 255, 255), cv2.FILLED)
    # cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 4)

    cv2.rectangle(frame, (left, top-10 - round(2*labelSize[1])), (left + round(2*labelSize[0]), top + baseLine-15),    (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)

    
#====================================================================

def postprocess(frame, outs,classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        
        for detection in out:
            #if detection[4]>0.001:
            scores = detection[5:]
            classId = np.argmax(scores)
            
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    labeled = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print(indices)
    if(len (indices)<=0):
        cropped = np.zeros(3)
        left=0
        top=0
        right=0
        bottom=0
        labeled="No Detection"

    if len(indices) > 0:
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            label = str.upper((classes[classIds[i]]))
            labeled.append( classes[classIds[i]])
            # calculate bottom and right
            bottom = top+height
            right = left+width
            print("Top:",top)
            print("Bottom: ",bottom)
            print("Let: ",left)
            print("Right: ",right)
            if top<0:
                top=0
            if left<0:
                left=0
            if right>frameWidth:
                right=frameWidth
            if bottom>frameHeight:
                bottom=frameHeight
            #crop the plate out
            cropped = frame[top:bottom, left:right].copy()
            # drawPred
            drawPred(classIds[i], confidences[i], left, top, right, bottom, frame,classes)

    # print(cropped,left, top, right, bottom)
    return cropped,left, top, right, bottom,labeled

# Give the configuration and weight files for the model and load the network using them.
# modelConfiguration_1 = "yolo_vehicle/yolov4-obj.cfg"
# modelWeights_1 = "yolo_vehicle/yolov4-obj_final_Vehicle.weights"
# classesFile_1 = "yolo_vehicle/obj.names"


# modelConfiguration_2 = "yolo_lp/yolov4-obj.cfg"
# modelWeights_2 = "yolo_lp/yolov4-obj_final_LPD.weights"
# classesFile_2 = "yolo_lp/obj.names"


# modelConfiguration_3 = "yolo_character/yolov4-obj.cfg"
# modelWeights_3 = "yolo_character/yolov4-obj_final_ch.weights"
# classesFile_3 = "yolo_character/obj.names"

@st.cache(allow_output_mutation=True)
def load_net():
    modelConfiguration_1 = "Weights/type/yolov4_custom.cfg"
    modelWeights_1 = "Weights/type/yolov4_custom_last.weights"
    classesFile_1 = "Weights/type/obj.names"

    
    modelConfiguration_2 = "Weights/lp/yolov4_custom.cfg"
    modelWeights_2 = "Weights/lp/yolov4_custom_last.weights"
    classesFile_2 = "Weights/lp/obj.names"


    modelConfiguration_3 = "Weights/char/yolov4_custom.cfg"
    modelWeights_3 = "Weights/char/yolov4_custom_last.weights"
    classesFile_3 = "Weights/char/obj.names"
    
    
    ################ TINY MODEL WEIGHTS
    # modelConfiguration_1 = "New_weights_tiny/type/yolov3-tiny_custom.cfg"
    # modelWeights_1 = "New_weights_tiny/type/yolov3-tiny_custom_last.weights"
    # classesFile_1 = "New_weights_tiny/type/obj.names"

    # modelConfiguration_2 = "New_weights_tiny/lp/yolov3-tiny_custom.cfg"
    # modelWeights_2 = "New_weights_tiny/lp/yolov3-tiny_custom_last.weights"
    # classesFile_2 = "New_weights_tiny/lp/obj.names"
    
    # modelConfiguration_3 = "New_weights_tiny/char/yolov3-tiny_custom.cfg"
    # modelWeights_3 = "New_weights_tiny/char/yolov3-tiny_custom_last.weights"
    # classesFile_3 = "New_weights_tiny/char/obj.names"
    

    print ("------------- DEFINING YOLO PATHS -----------------")

    #======================  split classes  ===========================
    classes_1 = split_classes(classesFile_1)
    classes_2 = split_classes(classesFile_2)
    classes_3 = split_classes(classesFile_3)

    print ("------------- SPLIT ALL CLASSES  -----------------")

    #=====================  LOADING YOLO ===========================

    net_1 = network(modelConfiguration_1, modelWeights_1)
    net_2 = network(modelConfiguration_2, modelWeights_2)
    net_3 = network(modelConfiguration_3, modelWeights_3)

    print ("------------- LOADING ALL YOLO's -----------------")
    return classes_1,classes_2,classes_3,net_1,net_2,net_3
    

# is_video = False
# inputFile = 'original_image/5.jpg'
# cap1 = cv2.VideoCapture(inputFile)
def detect_objects(our_image,log_file):

    while cv2.waitKey(1) < 0:
        # get frame from the video
        # hasFrame1, frame1 = cap1.read() #frame: an image object from cv2
        # if not hasFrame1:
        #     print("Done processing !!!")
        #     cv2.waitKey(3000)
        #     break

        new = np.array(our_image.convert('RGB'))
        frame1 = cv2.cvtColor(new,1)

        st.set_option('deprecation.showPyplotGlobalUse', False)

        col1, col2 = st.columns(2)

        col1.subheader("Original Image")
        st.text("")
        plt.figure(figsize = (15,15))
        plt.imshow(our_image)
        col1.pyplot(use_column_width=True)

        blob = cv2.dnn.blobFromImage(frame1, 1/255, (inpWidth, inpHeight), [0,0,0], True, crop=False)
        net_1.setInput(blob)
        t1 = time.time()
        outs_1 = net_1.forward(getOutputsNames(net_1))
        t2 = time.time()
        x1=t2-t1
        cropped1,left1, top1, right1, bottom1,label1 = postprocess(frame1, outs_1,classes_1)
        # print("cropped1 ",cropped1)
        # try:
        #     st.subheader("Detected objects: " + ''.join(label1))
        # except IndexError:
        #     st.write("Nothing detected")
        
        # ,channels='BGR'
        st.text("")
        col2.subheader("Detected objects: " + ''.join(label1))
        st.text("")
        plt.figure(figsize = (15,15))
        plt.imshow(frame1)
        col2.pyplot(use_column_width=True)
        f1 = cropped1.flatten()
        if (f1.any()<=0):
            break

        print("CLASSES 1: ",classes_1[0])
        print(type(label1))
        print(type(classes_1))
        print("LABEL 1: ", label1)

        str1 = "" 
        for ele in label1:
            str1 += ele

        print("String: ",str1)
        print(type(str1))

        if (str1=='Car'):
            toll = 30
        if (str1=='Bus'):
            toll = 100
        if (str1=='Van'):
            toll = 50
        if (str1=='Suzuki/Carry'):
            toll = 30
        if (str1=='Type 1 Truck'):
            toll = 120
        if (str1=='Type 2 Truck'):
            toll = 250

        cv2.imwrite('sample1.jpg', cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR))


        blob = cv2.dnn.blobFromImage(cropped1, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net_2.setInput(blob)

        # Runs the forward pass to get output of the output layers
        t3 = time.time()
        outs_2 = net_2.forward(getOutputsNames(net_2))
        t4 = time.time()
        x2=t4-t3
        cropped2,left2, top2, right2, bottom2,label2 = postprocess(cropped1, outs_2,classes_2)
        
        st.text("")
        col2.subheader("Detected objects: " + ''.join(label2))
        st.text("")
        plt.figure(figsize = (15,15))
        plt.imshow(cropped1)
        col2.pyplot(use_column_width=True)
        f2 = cropped2.flatten()
        if (f2.any()<=0):
            break
        cv2.imwrite('sample2.jpg', cv2.cvtColor(cropped1, cv2.COLOR_RGB2BGR))


        blob = cv2.dnn.blobFromImage(cropped2, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        net_3.setInput(blob)

        # Runs the forward pass to get output of the output layers
        t5 = time.time()
        outs_3 = net_3.forward(getOutputsNames(net_3))
        t6 = time.time()
        x3=t6-t5
        total_time=x1+x2+x3
        
        boxes, confidences, class_IDs = [], [], []
        H, W = cropped2.shape[:2]
        for output in outs_3:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype("int")
                    x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_IDs.append(classID)


        # Apply non-max suppression to identify best bounding box
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        xmin, xmax, ymin, ymax, labels = [], [], [], [], []
        xmi,xma,ymi,yma,lc=[],[],[],[],[]
        xc,xcm,yc,ycm,ln=[],[],[],[],[]
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                xmin.append(x)
                ymin.append(y)
                xmax.append(x+w)
                ymax.append(y+h)
                label = str.upper((classes_3[class_IDs[i]]))
                print("-------------")

#                 print('label ',label)
                if (label == str('A') or label == str('B')or label == str('C')
                or label == str('D')or label == str('E')or label == str('F')
                or label == str('G')or label == str('H')or label == str('I')
                or label == str('J')or label == str('K')or label == str('L')
                or label == str('M')or label == str('N')or label == str('O')
                or label == str('P')or label == str('Q')or label == str('R')
                or label == str('S')or label == str('T')or label == str('U')or label == str('V')
                or label == str('W')or label == str('X')or label == str('Y')or label == str('Z')):
                    xmi.append(x)
                    ymi.append(y)
                    xma.append(x+w)
                    yma.append(y+h)
                    lc.append(label)
#                     print("xmi ",xmi,ymi,xma,yma)

                if (label == str('0') or label == str('1')or label == str('2')or label == str('3')
                or label == str('4')or label == str('5')or label == str('6')or label == str('7')
                or label == str('8')or label == str('9')):
                    xc.append(x)
                    yc.append(y)
                    xcm.append(x+w)
                    ycm.append(y+h)
                    ln.append(label)
#                     print("xc ",xc,yc,xcm,ycm)

                
        boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        char = pd.DataFrame({"xmi": xmi, "ymi": ymi, "xma": xma, "yma": yma, "Label": lc})
        num = pd.DataFrame({"xmi": xc, "yc": yc, "xca": xcm, "yca": ycm, "Label": ln})

        char.sort_values(by=['xmi'], inplace=True)
        num.sort_values(by=['xmi'], inplace=True)

        a=char[char.columns[4]]

        b=num[num.columns[4]]
        res = a.append(b)

        # Add a layer on top on a detected object 
        LABEL_COLORS = [0, 255, 0]
        image_with_boxes = cropped2.astype(np.float64)
        characters = image_with_boxes.astype(np.uint8)

        d1 = pd.DataFrame({"label":res})
        d2 = d1.transpose()
        d3 = d2.values.tolist()
        flatten_mat=[]
        for sublist in d3:
            for val in sublist:
                flatten_mat.append(val)


        lp_num = ''.join(map(str,flatten_mat))
        print("LP NUM: ",lp_num)
        print("Total time: ",total_time)
        for _, (xmin, ymin, xmax, ymax) in boxes.iterrows():
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += (np.random.randint(0, 255, size=(3),dtype="uint8"))
            cv2.rectangle(image_with_boxes,(xmin,ymin),(xmax,ymax),(0,0,0),1)
            # cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine-10),    (255, 255, 255), cv2.FILLED)
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2
  

        img_pil = Image.fromarray(image_with_boxes.astype(np.uint8))
        
        cv2.putText(image_with_boxes, lp_num, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1,cv2.LINE_AA)        
        # Display the final image
        # st.image(image_with_boxes.astype(np.uint8), width = 300)
        st.text("")
        col2.subheader("Detected objects: " + ''.join(res))
        st.text("")
        plt.figure(figsize = (15,15))
        plt.imshow(image_with_boxes.astype(np.uint8))
        col2.pyplot(use_column_width=True)
        cv2.imwrite('sample3.jpg', cv2.cvtColor(image_with_boxes.astype(np.uint8), cv2.COLOR_RGB2BGR))
        with open(log_file, 'r') as f:

            dataf= pd.DataFrame({"Vehicle Type":label1,"LP Number":lp_num,"Inference Time":total_time,"Toll Amount":toll})
            dataf.to_csv(log_file,index=False,mode='a',header=False)


        break



classes_1,classes_2,classes_3,net_1,net_2,net_3 = load_net()
# images_path = glob.glob(r"original_image/check\*.jpg")
# for img_path in images_path:
#     our_image = Image.open(img_path) 
#     detect_objects(our_image,"new.csv")

st.set_option('deprecation.showfileUploaderEncoding', False)
image_file = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
if image_file is not None:
    our_image = Image.open(image_file) 
    detect_objects(our_image,"prediction_result.csv")

