import os
import io

small_data_dir = 'D:\\all-dataset\\dogs-vs-cats-small'
# 划分数据集:train|validation|test
train_dir = os.path.join(small_data_dir, 'train')
validation_dir = os.path.join(small_data_dir, 'validation')
test_dir = os.path.join(small_data_dir, 'test')

# cats dogs train
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
# cats dogs validation
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# cats dogs test
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

from PIL import Image

lines = []
catslist = os.listdir(train_cats_dir)
for list in catslist:
    path = ''
    try:
        path = os.path.join(train_cats_dir, list)
        im = Image.open(path)
    except Exception:
        print(path + ' load err')
        continue
    print(path)
    line = path + ' 0\n'
    lines.append(line)

dogslist = os.listdir(train_dogs_dir)
for list in dogslist:
    path = ''
    try:
        path = os.path.join(train_dogs_dir, list)
        im = Image.open(path)
    except Exception:
        print(path + ' load err')
        continue
    print(path)
    line = path + ' 1\n'
    lines.append(line)
filepath = './catvsdog-train.txt'
file = open(filepath, 'w+')
file.writelines(lines)
