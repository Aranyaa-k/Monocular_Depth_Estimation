import cv2
import torch
import time
import numpy as np

#model_type = "DPT_Large"   #MiDas v3 - Large (Highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"  #MiDas v3 - Hybrid (Medium accuracy, medium inference speed)
#model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

print(torch.cuda.is_available())

midas_transforms = torch.hub.load("intel-isl/MiDaS","transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    start = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(prediction.unsqueeze(1), size=img.shape[:2],mode="bicubic",align_corners=False,).squeeze()

        depth_map = prediction.cpu().numpy()

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        end = time.time()
        totalTime = end - start

        fps = 1/ totalTime

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        cv2.putText(img, f'Depth Map Test 5.2', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map)

    key = cv2.waitKey(1)    
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
