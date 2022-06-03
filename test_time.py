import os
import time
import torch
from CS_net_model import *
from model import *
from modelother import *
from models.deform_unet import *

# 是否使用cuda
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model_dir = './model_save/DVC-unet/bestmodel1.pth'
inputs = torch.randn(1, 1, 304, 304).to(device)
first_net = segnet().to(device)
#first_net.load_state_dict(torch.load(model_dir))
# second_net = fusion(channels=32, pn_size=3).to(device)
start_t = time.time()
_, _, _, outputs = first_net(inputs)
end_t = time.time()
print("cost time: %f s" % (end_t - start_t))