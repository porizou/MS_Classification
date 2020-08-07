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
class MS(Chain):
    def __init__(self):
        super(MS, self).__init__(
            # L.Convolution2D(チャンネル数, フィルター数, フィルタのサイズ)
            cnn1=L.Convolution2D(3, 15, 5), # 画像サイズ(3, 128, 128) → (15, 124, 124)
            cnn2=L.Convolution2D(15, 40, 5), # 画像サイズ(15, 62, 62) → (40, 58, 58)
            l1=L.Linear(33640, 400), # 入力は40 x 29 x 29 = 33640
            l2=L.Linear(400, 2),
        )

    def __call__(self, x, t):
        return F.softmax_cross_entropy(self.predict(x), t)

    def predict(self, x):
        # F.max_pooling_2d(入力画像, 領域のサイズ)
        h1 = F.max_pooling_2d(F.relu(self.cnn1(x)), 2)  # 画像サイズ(15, 124, 124) → (15, 62, 62)
        h2 = F.max_pooling_2d(F.relu(self.cnn2(h1)), 2)  # 画像サイズ(40, 58, 58) → (40, 29, 29)
        h3 = F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)

# -- モデルとoptimizerの設定 --
model = MS()
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