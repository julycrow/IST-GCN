import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    #if platform == "win32":
    # Change these variables to point to the correct folder (Release/x64 etc.)
    #sys.path.append('D:/PycharmProject/openpose/build/python/openpose/Release')
    #os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + 'D:/PycharmProject/openpose/build/x64/Release' + dir_path + 'D:/PycharmProject/openpose/build/bin'
    sys.path.append(dir_path + '/../../openpose/build/python/openpose/Release');
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../openpose/build/x64/Release;' +  dir_path + '/../../openpose/build/bin;'
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e