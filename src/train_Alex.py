import numpy as np
from PIL import Image
import glob

import chainer
from chainer import Variable, Chain, optimizers, serializers, datasets
import chainer.links as L
import chainer.functions as F

from chainer.datasets import tuple_dataset
from chainer import training, iterators
from chainer.training import extensions


# --データセットの作成
def image2TrainAndTest(pathsAndLabels, size=128, channels=3):
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "*")
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)

    imageData = []
    labelData = []
    for pathAndLabel in allData:
        img = Image.open(pathAndLabel[0])
        print(pathAndLabel)
        r, g, b = img.split()
        rImgData = np.asarray(np.float32(r) / 255.0)
        gImgData = np.asarray(np.float32(g) / 255.0)
        bImgData = np.asarray(np.float32(b) / 255.0)
        imgData = np.asarray([rImgData, gImgData, bImgData])
        imageData.append(imgData)
        labelData.append(np.int32(pathAndLabel[1]))

        threshold = np.int32(len(imageData) / 8 * 7)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test = tuple_dataset.TupleDataset(imageData[threshold:], labelData[threshold:])

    return train, test

pathsAndLabels = []
pathsAndLabels.append(np.asarray(["/Users/mori/Desktop/MS_Classification/0/", 0]))
pathsAndLabels.append(np.asarray(["/Users/mori/Desktop/MS_Classification/1/", 1]))
# データセットの取得
train_data, test_data = image2TrainAndTest(pathsAndLabels)


# -- Chainの記述 --
class Alex(chainer.Chain):

    def __init__(self):
        super(Alex, self).__init__(
            # L.Convolution2D(チャンネル数, フィルター数, フィルタのサイズ)
            conv1=L.Convolution2D(3, 96, 8, stride=4),  # 画像サイズ(3, 128, 128) → (96, 31, 31)
            conv2=L.Convolution2D(96, 256, 5, pad=2),  # 画像サイズ(96, 15, 15) → (256, 15, 15)
            conv3=L.Convolution2D(256, 384, 3, pad=1),  # 画像サイズ(256, 7, 7) → (384, 7, 7)
            conv4=L.Convolution2D(384, 384, 3, pad=1),  # 画像サイズ(384, 7, 7) → (384, 7, 7)
            conv5=L.Convolution2D(384, 256, 3, pad=1),  # 画像サイズ(384, 7, 7) → (256, 7, 7)

            fc6=L.Linear(9216, 4096),  # 入力は256 x 6 x 6 = 9216
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1024),
            fc9=L.Linear(1024, 2),
        )


    def clear(self):
        self.loss = None
        self.accuracy = None


    def __call__(self, x, t):
        self.clear()
        v = self.predict(x)
        self.loss = F.softmax_cross_entropy(v, t)
        self.accuracy = F.accuracy(v, t)
        return self.loss


    def predict(self, x):
        # F.max_pooling_2d(入力画像, 領域のサイズ)
        h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(self.conv1(x))), 3, stride=2)  # 画像サイズ(96, 31, 31) → (96, 15, 15)
        h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(self.conv2(h))), 3, stride=2)  # 画像サイズ(256, 15, 15) → (256, 7, 7)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=1)  # 画像サイズ(256, 7, 7) → (256, 6, 6)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = F.dropout(F.relu(self.fc8(h)))
        h = self.fc9(h)

        return h

# -- モデルとoptimizerの設定 --
model = Alex()
optimizer = optimizers.Adam()
optimizer.setup(model)

# -- 学習 --
iterator = iterators.SerialIterator(train_data, 100)
updater = training.StandardUpdater(iterator, optimizer)
trainer = training.Trainer(updater, (20, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.run()

# -- モデルの保存 --
serializers.save_npz("ms_classification.npz", model)

# -- テスト--
correct = 0
for i in range(len(test_data)):
    x = Variable(np.array([test_data[i][0]], dtype=np.float32))
    t = test_data[i][1]
    y = model.predict(x)
    maxIndex = np.argmax(y.data)
    if (maxIndex == t):
        correct += 1

# -- 正解率 --
print("Correct:", correct,  "Total:", len(test_data), "Acuuracy:", correct / len(test_data) * 100, "%")