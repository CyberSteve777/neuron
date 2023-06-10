from keras.datasets import mnist
import os
from PIL import Image


(train_X, train_y), (test_X, test_y) = mnist.load_data()
paths = []
num = []
os.chdir(os.pardir)
os.chdir("./digits/testing")
# print(len(test_y))
for i in range(10):
    if not os.path.exists(str(i)):
        os.mkdir(str(i))
    paths.append(os.path.abspath(os.curdir) + os.sep + str(i))
    num.append(1)
for i in range(len(test_X)):
    digit = test_X[i]
    pic = Image.fromarray(digit)
    lb = test_y[i]
    name = str(num[lb])
    pic.save(os.path.join(paths[lb], f"{name}.png"))
    num[lb] += 1

