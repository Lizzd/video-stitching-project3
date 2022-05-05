import os
import pyperclip

s = ''
cnt = 200
for f in os.listdir('frames/'):
    if cnt % 3 == 0:
        s += '"d:/data/video-stitching-project3/frames/' + f + '" '
    cnt -= 1
    if not cnt:
        break
# s1 = '''.\example_cpp_stitching_detailed ''' + s + '''--timelapse as_is'''
s1 = '''python stitching_detailed.py ''' + s + '''--timelapse as_is'''
print(s1)
pyperclip.copy(s1)