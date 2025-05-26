import cv2
import argparse
import numpy as np
from gtts import gTTS
import pyglet
import os, time


audio_timestamp = time.time()
audio_duration = 0


def speakOut(message):
    global audio_timestamp
    global audio_duration
    path_of_files = "assets/"
    tts = gTTS(text=message, lang='en')
    ttsname=(path_of_files+"message.mp3")
    tts.save(ttsname)
    music = pyglet.media.load(ttsname, streaming = False)
    music.play()
    audio_timestamp = time.time()
    audio_duration = music.duration
    #time.sleep(music.duration)
    #os.remove(ttsname)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)





#cap = cv2.VideoCapture("animals.mp4")
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, image = cap.read()
    #for x in range(0,50):
    #   ret, image = cap.read()
    
    image = cv2.resize(image,(640,360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    if audio_timestamp+audio_duration >= time.time():
        cv2.imshow('live',image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
   



    classes = None

    with open("yolov3.txt", 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                print("Class id:",end="")
                print(class_id)
                print(classes[class_id])
                
               
                
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])



    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
   


    cv2.imshow('detection',image)
    
    text = ""
    for i in indices:
        i = i[0]
        print(class_ids[i])
        if not text == "":
            text += ", "+classes[class_ids[i]]
        else:
            text += classes[class_ids[i]]
            
            
    print("text:",text)
    if not text == "": 
        os.remove("assets/message.mp3")
        speakOut(text)
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
