import os
import warnings
#warnings.filterwarnings('ignore')

from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K

from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from tools.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path='models/yolo.h5'
        self.score=0.5
        self.iou=0.5
        self.class_names=['person','bicycle','car','motorbike','aeroplane','bus','train',
                          'truck','boat','traffic light','fire hydrant','stop sign',
                          'parking meter','bench','bird','cat','dog','horse','sheep',
                          'cow','elephant','bear','zebra','giraffe','backpack','umbrella',
                          'handbag','tie','suitcase','frisbee','skis','snowboard',
                          'sports ball','kite','baseball bat','baseball glove','skateboard',
                          'surfboard','tennis racket','bottle','wine glass','cup','fork',
                          'knife','spoon','bowl','banana','apple','sandwich','orange',
                          'broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
                          'pottedplant','bed','diningtable','toilet','tvmonitor','laptop',
                          'mouse','remote','keyboard','cell phone','microwave','oven',
                          'toaster','sink','refrigerator','book','clock','vase','scissors',
                          'teddy bear','hair drier','toothbrush']

        self.anchors=np.array([[10,13],
                               [16,30],
                               [33,23],
                               [30,61],
                               [62,45],
                               [59,119],
                               [116,90],
                               [156,198],
                               [373,326]])
        self.sess=K.get_session()
        self.model_image_size=(416, 416) # fixed size or (None, None)
        self.is_fixed_size=self.model_image_size!=(None, None)
        self.boxes,self.scores,self.classes=self.generate()


    def generate(self):
        model_path=os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        #Loading the model
        self.yolo_model=load_model(model_path,compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape=K.placeholder(shape=(2,))
        boxes,scores,classes=yolo_eval(self.yolo_model.output,self.anchors,
                len(self.class_names),self.input_image_shape,
                score_threshold=self.score,iou_threshold=self.iou)

        return boxes,scores,classes

    def detect_image(self,image):

        if self.is_fixed_size:
            assert self.model_image_size[0]%32==0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32==0, 'Multiples of 32 required'
            boxed_image=letterbox_image(image,tuple(reversed(self.model_image_size)))
        else:
            #Converting size to multiple of 32
            new_image_size=(image.width-(image.width%32),image.height-(image.height%32))

            if new_image_size==(image.width,image.height): #if condition added
                boxed_image=image
            else:
                boxed_image=letterbox_image(image,new_image_size)

        image_data=np.array(boxed_image,dtype='float32')

        #print(image_data.shape)
        image_data/=255.
        image_data=np.expand_dims(image_data,0)  # Add batch dimension.
        
        out_boxes,out_scores,out_classes=self.sess.run(
            [self.boxes,self.scores,self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1],image.size[0]],
                K.learning_phase(): 0
            })

        return_boxs=[]
        for i,c in reversed(list(enumerate(out_classes))):
            predicted_class=self.class_names[c]
            if predicted_class!='person':
                continue
            box=out_boxes[i]
           # score = out_scores[i]  
            x=int(box[1])  
            y=int(box[0])  
            w=int(box[3]-box[1])
            h=int(box[2]-box[0])
            if x<0:
                w=w+x
                x=0
            if y<0:
                h=h+y
                y=0 
            return_boxs.append([x,y,w,h])

        return return_boxs

    def close_session(self):
        self.sess.close()
