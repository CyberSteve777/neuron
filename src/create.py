import os
from PIL import Image


def get_files_recursive(direct, ext):
    fl = []
    for root, dirs, files in os.walk(direct):
        for f in files:
            if f.endswith(ext):
                fl.append(os.path.join(root, f))
    return fl


directory = os.path.abspath('../input_digits')
extension = '.png'
os.chdir(os.pardir)
file_list = get_files_recursive(directory, extension)
with open('test.txt', 'a') as file:
    file.write(f'{len(file_list)}\n')

    for count in range(len(file_list)):
        file.write('1\n')
        file_path = file_list[count]
        with Image.open(file_path) as image_file:
            # Process the image file as needed
            # Here, we'll just print the pixel values
            # Replace this code with your actual processing logic
            for i in range(image_file.height):
                for j in range(image_file.width):
                    normalized_value = float(image_file.getpixel((j, i))) / 255
                    file.write(f'{normalized_value:.3f} ')
                file.write('\n')

print('done!')
