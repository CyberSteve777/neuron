import os
from PIL import Image


def process_images(directory, label, out):
    files = os.listdir(directory)
    for file in files:
        if file.endswith('.png'):
            t.write(str(label) + '\n')
            path = os.path.join(directory, file)
            with Image.open(path) as image_file:
                for i in range(image_file.height):
                    for j in range(image_file.width):
                        normalized_value = float(image_file.getpixel((j, i))) / 255
                        out.write(f'{normalized_value:.3f} ')
                    out.write('\n')


t = open(os.path.join(os.pardir, 'lib_MNIST_edit.txt'), "w")
rdir = os.path.abspath("../digits/training")
cnt = 0
for roots, dirs, fls in os.walk(rdir):
    cnt += len(fls)
t.write(f"{cnt}\n")
for ind in range(10):
    dr = os.path.abspath(os.path.join(rdir, str(ind)))
    process_images(dr, ind, t)
t.close()
print('done!')
