import os
import shutil

src_dir_path = 'D:\\dataset\\image-classify\\train\\n1'
dest_dir_path = 'D:\\dataset\\image-classify\\train\\n'
if not os.path.exists(dest_dir_path):
    os.mkdir(dest_dir_path)

import os

i = 108894 #n
#i = 10064282 #p
for fpathe, dirs, fs in os.walk(src_dir_path):
    for f in fs:
        i += 1
        childPath = os.path.join(fpathe, f)
        if os.path.isdir(childPath):
            print(childPath)
        shutil.copy(childPath, os.path.join(dest_dir_path, str(i) + '.jpg'))

print(i)