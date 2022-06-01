import os
cwd = os.path.abspath(os.path.join(__file__, os.pardir))
import sys
sys.path.append(cwd)


model_adress_ = cwd + "/yolov5s-face.pt"