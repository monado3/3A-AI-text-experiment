#!/usr/bin/python
#
# train the model
import numpy as np
import chainer 
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import os
import sys
import time

def get_dataset(dim=25):
    x = []
    y = []
    datalist = []
    with open("conf/train.list","r") as f:
        for line in f:
            line = line.rstrip()
            datalist.append(line)

    for d in datalist:
        print(d)
        with open("data/SF/data/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            x.append(dat.reshape(int(len(dat)/dim),dim))
        with open("data/TF/data/{}.dat".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            y.append(dat.reshape(int(len(dat)/dim),dim))
    return x,y

class VCDNN(Chain):
        def __init__(self, dim=25, n_units=256):
            super(VCDNN, self).__init__(
                    l1=L.Linear(dim, n_units),
                    l2=L.Linear(n_units,n_units),
                    l3=L.Linear(n_units,dim))

        def __call__(self, x_data, y_data,dim=25):
                x = Variable(x_data.astype(np.float32).reshape(len(x_data),dim))
                y = Variable(y_data.astype(np.float32).reshape(len(y_data),dim))

                return F.mean_squared_error(self.predict(x), y)
        
        def predict(self, x):
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))
            h3 = self.l3(h2)
            return h3

        def get_predata(self, x):
            return self.predict(Variable(x.astype(np.float32))).data


if __name__ == "__main__":
    x_train, y_train = get_dataset()
    # parameters for training
    #batchsize = 64 
    n_epoch = 50 
    dim = 25
    #n_units = 256
    n_units = 128
    N = len(x_train)
    
    model = VCDNN(dim,n_units)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # loop
    losses = []
    sum_loss = 0
    for epoch in range(1, n_epoch + 1):
        sum_loss = 0

        for i in range(0, N):
            x_batch = x_train[i]
            y_batch = y_train[i]
            model.zerograds()
            loss = model(x_batch,y_batch,dim)
            sum_loss += loss.data
            loss.backward()
            optimizer.update()
            average_loss = sum_loss / N
            losses.append(average_loss)

            print("epoch: {}/{}  loss: {}".format(epoch, n_epoch, average_loss))

    model.to_cpu()
    if not os.path.isdir("model"):
        os.mkdir("model")
    serializers.save_npz("model/vcmodel.npz",model)
