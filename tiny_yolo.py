import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from cfg import *
from darknet import MaxPoolStride1, Darknet
from region_loss import RegionLoss

class TinyYoloNet(nn.Module):
    def __init__(self):
        super(TinyYoloNet, self).__init__()
        self.seen = 0
        self.num_classes = 80
        self.anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.1682]
        self.num_anchors = len(self.anchors)/2
        num_output = int((5+self.num_classes)*self.num_anchors)
        self.width = 416
        self.height = 416

        self.loss = RegionLoss(self.num_classes, self.anchors, self.num_anchors)
        self.cnn = nn.Sequential(OrderedDict([
            # conv1
            ('conv1', nn.Conv2d( 3, 16, 3, 1, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(16)),
            ('leaky1', nn.LeakyReLU(0.1, inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            # conv2
            ('conv2', nn.Conv2d(16, 32, 3, 1, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(32)),
            ('leaky2', nn.LeakyReLU(0.1, inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),

            # conv3
            ('conv3', nn.Conv2d(32, 64, 3, 1, 1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('leaky3', nn.LeakyReLU(0.1, inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),

            # conv4
            ('conv4', nn.Conv2d(64, 128, 3, 1, 1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('leaky4', nn.LeakyReLU(0.1, inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),

            # conv5
            ('conv5', nn.Conv2d(128, 256, 3, 1, 1, bias=False)),
            ('bn5', nn.BatchNorm2d(256)),
            ('leaky5', nn.LeakyReLU(0.1, inplace=True)),
            ('pool5', nn.MaxPool2d(2, 2)),

            # conv6
            ('conv6', nn.Conv2d(256, 512, 3, 1, 1, bias=False)),
            ('bn6', nn.BatchNorm2d(512)),
            ('leaky6', nn.LeakyReLU(0.1, inplace=True)),
            ('pool6', MaxPoolStride1()),

            # conv7
            ('conv7', nn.Conv2d(512, 1024, 3, 1, 1, bias=False)),
            ('bn7', nn.BatchNorm2d(1024)),
            ('leaky7', nn.LeakyReLU(0.1, inplace=True)),

            # conv8
            ('conv8', nn.Conv2d(1024, 512, 3, 1, 1, bias=False)),
            ('bn8', nn.BatchNorm2d(512)),
            ('leaky8', nn.LeakyReLU(0.1, inplace=True)),

            # output
            ('output', nn.Conv2d(512, num_output, 1, 1, 0)),
        ]))

    def forward(self, x):
        # out1 = self.cnn1(x)
        # print(out1)
        # out2 = self.bn1(out1)
        # print(out2)
        # out3 = self.mp1(out2)
        # x = self.cnn2(out3)
        x = self.cnn(x)
        return x

    def print_network(self):
        print(self)

    def load_weights(self, path):
        #buf = np.fromfile('tiny-yolo-voc.weights', dtype = np.float32)
        buf = np.fromfile(path, dtype=np.float32)
        start = 4

        # start = load_conv_bn(buf, start, self.cnn1[0], self.bn1[0])
        # start = load_conv_bn(buf, start, self.cnn2[0], self.cnn2[1])
        # start = load_conv_bn(buf, start, self.cnn2[4], self.cnn2[5])
        # start = load_conv_bn(buf, start, self.cnn2[8], self.cnn2[9])
        # start = load_conv_bn(buf, start, self.cnn2[12], self.cnn2[13])
        # start = load_conv_bn(buf, start, self.cnn2[16], self.cnn2[17])
        # start = load_conv_bn(buf, start, self.cnn2[20], self.cnn2[21])
        #
        # start = load_conv_bn(buf, start, self.cnn2[23], self.cnn2[24])
        # start = load_conv(buf, start, self.cnn2[26])

        start = load_conv_bn(buf, start, self.cnn[0], self.cnn[1])
        start = load_conv_bn(buf, start, self.cnn[4], self.cnn[5])
        start = load_conv_bn(buf, start, self.cnn[8], self.cnn[9])
        start = load_conv_bn(buf, start, self.cnn[12], self.cnn[13])
        start = load_conv_bn(buf, start, self.cnn[16], self.cnn[17])
        start = load_conv_bn(buf, start, self.cnn[20], self.cnn[21])

        start = load_conv_bn(buf, start, self.cnn[24], self.cnn[25])
        start = load_conv_bn(buf, start, self.cnn[27], self.cnn[28])
        start = load_conv(buf, start, self.cnn[30])

if __name__ == '__main__':
    from PIL import Image
    from utils import *
    # m = Darknet('cfg/tiny-yolo.cfg')
    # m.load_weights('tiny-yolo.weights')
    m = TinyYoloNet()
    m.type(torch.cuda.FloatTensor)
    m.train()
    m.load_weights('tiny-yolo.weights')
    print(m)
    
    use_cuda = 1
    if use_cuda:
        m.type(torch.cuda.FloatTensor)

    img = Image.open('data/horses.jpg').convert('RGB')
    sized = img.resize((240,320))
    boxes = do_detect(m, sized, 0.6, 0.5, use_cuda)

    class_names = load_class_names('data/coco.names')
    plot_boxes(img, boxes, 'predict1.jpg', class_names)  

