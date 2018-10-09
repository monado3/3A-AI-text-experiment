#!/usr/bin/python
#
# convert features by DNN 
import numpy as np
import chainer 
from chainer import Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import pysptk as sptk
import pyworld as pw
from scipy.io import wavfile
import os
import sys
import time


class VCDNN(Chain):
        def __init__(self, dim=24, n_units=256):
            super(VCDNN, self).__init__(
                    l1=L.Linear(dim, n_units),
                    l2=L.Linear(n_units,n_units),
                    l3=L.Linear(n_units,dim))

        def __call__(self, x_data, y_data,dim=24):
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
    # x_train, y_train = get_dataset()
    # parameters for training
    #batchsize = 64 
    dim = 25
    n_units = 128
    
    model = VCDNN(dim,n_units)
    serializers.load_npz("model/vcmodel.npz",model)

    # test data
    x = []
    datalist = []
    with open("conf/eval.list","r") as f:
        for line in f:
            line = line.rstrip()
            datalist.append(line)

    for d in datalist:
        with open("data/SF/mgc/{}.mgc".format(d),"rb") as f:
            dat = np.fromfile(f,dtype="<f8",sep="")
            x.append(dat.reshape(int(len(dat)/dim),dim))

    if not os.path.isdir("result"):
        os.mkdir("result")
    if not os.path.isdir("result/wav"):
        os.mkdir("result/wav")

    fs = 16000
    fftlen = 512
    alpha = 0.42
    for i in range(0,len(datalist)):
        outfile = "result/wav/{}.wav".format(datalist[i])
        with open("data/SF/f0/{}.f0".format(datalist[i]),"rb") as f:
            f0 = np.fromfile(f, dtype="<f8", sep="")
        with open("data/SF/ap/{}.ap".format(datalist[i]),"rb") as f:
            ap = np.fromfile(f, dtype="<f8", sep="")
            ap = ap.reshape(int(len(ap)/(fftlen+1)),fftlen+1)
        y = model.get_predata(x[i])
        y = y.astype(np.float64)
        sp = sptk.mc2sp(y, alpha, fftlen*2)
        owav = pw.synthesize(f0, sp, ap, fs)
        owav = np.clip(owav, -32768, 32767)
        wavfile.write(outfile, fs, owav.astype(np.int16))

