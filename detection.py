#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
import sys
import time
import threading

from pymodbus.client.sync import ModbusTcpClient as mtc

import platform
from smb.SMBConnection import SMBConnection as samba

labels = []

def start():
    send_prediction(prediction(get_image()))


def send_prediction(pred):
    MODBUS_IP = '192.168.1.1' # modbus ip address
    destination = '12345' # register address

    client = mtc(MODBUS_IP)
    client.connect()

    data = 0
    if pred == 'OK':
        data = 0
    else if pred == 'NG':
        data = 1
    else:
        data = 3

    client.write_register(destination, data)


def get_image():
    user = 'hoge'
    password = 'hoge'
    server = 'hoge'
    server_ip = '192.168.1.1'
    home = 'temp'
    path = 'hoge/hoge'

    pipe = samba(user, password, platform.node(), server)
    pipe.connect(server_ip, 139)

    ret = None # stored Sharedfile class
    items = pipe.listPath(home, path)
    for item in items:
        # any process

    # download image
    local_path = './cached/images'
    with open(local_path, 'wb') as f:
        pipe.retrieveFile(ret, os.path.join(home, path), f)


    return cv2.imread(os.path.join(local_path, ret.filename))


def prediction(img):
    X = []
    img = cv2.resize(img, (150, 150))

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

    return pred_label


def sheduler(interval, target, wait = True):
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
    parser.add_argument('-m', '--model', default='./model/mnist_deep_model.json')
    parser.add_argument('-w', '--weights', default='./model/model.h5')
    parser.add_argument('-l', '--labels', default='./model/labels.txt')
    args = parser.parse_args()

    with open(args.labels,'r') as f:
        for line in f:
            labels.append(line.rstrip())
    print(labels)

    model_pred = load_model(args.weights)

    scheduler(60, start)



if __name__ == "__main__":
    main()
