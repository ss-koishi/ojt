#!usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
import sys
import time
import threading
import os
import gc
from datetime import datetime as dt

from pymodbus.client.sync import ModbusTcpClient as mtc

import platform
from smb.SMBConnection import SMBConnection as samba

labels = []
model_pred = None

def start():
    send_prediction(prediction(get_image()))
    #prediction(get_image())


def triming(img):
    height, width = img.shape[:2]
    x = 1329
    y = 575
    w = 200
    h = 200

    img = img[y:y + h, x:x + w]
    return img
   


def send_error(res):
    MODBUS_IP = '192.168.2.3' # modbus ip address
    destination = 5
    unit = 2

    client = mtc(MODBUS_IP)
    client.connect()

    data = None
    if res:
        data = 1
    else: 
        data = 0

    client.write_register(destination, data, unit=unit)

def send_prediction(pred):
    MODBUS_IP = '192.168.2.3' # modbus ip address
    destination = 1
    unit = 2

    client = mtc(MODBUS_IP)
    client.connect()

    data = 0
    if pred == 'OK':
        data = 0
    elif pred == 'NG':
        data = 1
    else:
        data = 3

    client.write_register(destination, data, unit=unit)


prev_image = None
def get_image():
    user = 'nishio'
    password = 'decs'
    server = 'FvIoT'
    server_ip = '192.168.2.9'
    home = 'Camera'
    path = 'TEST1'
    global prev_image

    pipe = samba(user, password, platform.node(), server)
    pipe.connect(server_ip, 139)

    dirc = None 
    ret = None # stored Sharedfile class

    dirs = pipe.listPath(home, path)
    dirc = next(reversed(dirs))
    items = reversed(pipe.listPath(home, os.path.join(path, dirc.filename)))
    next(items)
    ret = next(items)
    
    send_error(prev_image == ret.filename)
    prev_image = ret.filename
    
    # download image
    local_path = './cached/images'
    with open(os.path.join(local_path, 'img.jpg'), 'wb') as f:
        pipe.retrieveFile(home, os.path.join(path, os.path.join(dirc.filename, ret.filename)), f)

    return ((os.path.join(home, os.path.join(path, dirc.filename)), ret.filename), os.path.join(local_path, 'img.jpg'))



def write_log(context, filename):
    user = 'nishio'
    password = 'decs'
    server = 'FvIoT'
    server_ip = '192.168.2.9'
    home = 'Jetson'
    path = 'EW01/RES01'

    pipe = samba(user, password, platform.node(), server)
    pipe.connect(server_ip, 139)

    isExist = False
    files = pipe.listPath(home, path)
    for name in files:
        if filename == name.filename:
            isExist = True

    if isExist:
        with open(os.path.join('./cached/logs', 'logs.log'), 'wb') as f:
            pipe.retrieveFile(home, os.path.join(path, filename), f)
    
    with open(os.path.join('./cached/logs', 'logs.log'), 'a') as f:
        f.write(context)
         
         
    with open(os.path.join('./cached/logs', 'logs.log'), 'rb') as f:
        pipe.storeFile(home, os.path.join(path, filename), f)
    


def prediction(data):

    global pred_model
    img = cv2.imread(data[1])
    img = triming(img)
    cv2.imwrite("hoge.png", img)
    X = []
    img = cv2.resize(img, (224, 224)) # VGG16's input images size is (224, 224, 3)

    # convert grayscale
    # gamma = np.array([pow(x/255.0, 2.2) for x in range(256)], dtype='float32')
    # img = cv2.LUT(img, gamma)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = pow(img, 1.0/2.2) * 255

    img = img_to_array(img)
    img = img / 255
    X.append(img)
    X = np.asarray(X)
    start = time.time()
    preds = model_pred.predict(X)
    elapsed_time = time.time() - start

    pred_label = ""

    label_num = 0
    tmp_max_pred = 0
    for i in preds[0]:
        if i > tmp_max_pred:
            pred_label = labels[label_num]
            tmp_max_pred = i
        label_num += 1

    log_text = dt.now().strftime('%Y/%m/%d-%H:%M:%S') + ',' + pred_label + ',' + data[0][0]  + ',' + data[0][1] + '\n'
    log_file = dt.now().strftime('%Y%m%d') + '.log'
    write_log(log_text, log_file)
        
    
    print(pred_label)
    return pred_label


def scheduler(interval, target, wait = True):
    base_time = time.time()
    next_time = 0
    while True:
        t = threading.Thread(target = target)
        t.start()
        if wait:
            t.join()
        next_time = ((base_time - time.time()) % interval) or intertval
        time.sleep(next_time)


def main():
    # parse options
    parser = argparse.ArgumentParser(description='detection')
#   parser.add_argument('-m', '--model', default='./model/mnist_deep_model.json')
    parser.add_argument('-m', '--model', default='./model/model.h5')
    parser.add_argument('-l', '--labels', default='./model/labels.txt')
    args = parser.parse_args()

    gc.collect()
    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    print(labels)
    
    global model_pred
    model_pred = load_model(args.model)
    scheduler(60, start)



if __name__ == "__main__":
    main()
