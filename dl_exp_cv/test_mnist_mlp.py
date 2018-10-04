#!/usr/bin/env python
try:
    import matplotlib

    matplotlib.use('Agg')
except ImportError:
    pass
import argparse

import chainer
from PIL import Image
from chainer import serializers
from dl_exp_cv.net import MLP


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--image', '-i', type=str, default="",
                        help='pass to input image')
    parser.add_argument('--model', '-m', default='my_mnist.model',
                        help='path to the training model')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()
    model = MLP(args.unit, 10)
    if args.gpu >= 0:
        model.to_gpu(chainer.cuda.get_device_from_id(args.gpu).use())
    serializers.load_npz(args.model, model)
    try:
        img = Image.open(args.image).convert("L").resize((28, 28))
    except:
        print("invalid input")
        return
    img_array = model.xp.asarray(img, dtype=model.xp.float32).reshape(1, 784)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        result = model.predict(img_array)
    # print("predict:", model.xp.argmax(result.data))
    pred_num_by_prob = model.xp.argsort(-result.data)
    print('the   most likely predicted number : {}'.format(pred_num_by_prob[0]))
    print('the second likely predicted number : {}'.format(pred_num_by_prob[1]))
    print('the  third likely predicted number : {}'.format(pred_num_by_prob[2]))

if __name__ == '__main__':
    main()
