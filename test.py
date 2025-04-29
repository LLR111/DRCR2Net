import torch
import cv2
from network import modul
from metrics import *

path = ""
out = ""
files = os.listdir(path)
net = modul.NET().load_state_dict(torch.load("./pretrained_model/model_dict.pth"))
if not os.path.exists(out):
    os.makedirs(out)
for file in files:
    file_path = os.path.join(path, file)
    img_320 = np.array(Image.open(file_path)) /255.
    img_320_t = np.transpose(img_320, (2, 0, 1))
    img_320_tensor = torch.tensor(img_320_t, dtype=torch.float32, device='cuda')
    img_320_tensor = img_320_tensor.unsqueeze(0)
    y = net(img_320_tensor)
    y = y.detach().cpu().numpy()
    y = np.squeeze(y)
    y = crf_refine(y,gt)
    out_path = os.path.join(out,file)
    cv2.imwrite(out_path,y*255)
    print(file)
