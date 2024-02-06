    
import cv2

if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("Number of CUDA enabled devices: ", cv2.cuda.getCudaEnabledDeviceCount())
        print("Current device: ", cv2.cuda.getDevice())
else:
        print("No CUDA enabled devices")