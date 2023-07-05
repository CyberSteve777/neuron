from keras.datasets import mnist
import os
from PIL import Image

(train_x, train_y), (test_x, test_y) = mnist.load_data()

os.chdir(os.pardir)
paths = {
    os.path.abspath("./digits/training"): train_x,
    os.path.abspath("./digits/testing"): test_x
}
for p, ar in paths.items():
    os.chdir(p)
    pths = []
    num = []
    print(os.path.abspath(os.curdir))
    for i in range(10):
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        pths.append(os.path.abspath(os.curdir) + os.sep + str(i))
        num.append(1)
    for i in range(10000):
        digit = test_x[i]
        pic = Image.fromarray(digit)
        lb = test_y[i]
        name = str(num[lb])
        pic.save(os.path.join(pths[lb], f"{name}.png"))
        num[lb] += 1
