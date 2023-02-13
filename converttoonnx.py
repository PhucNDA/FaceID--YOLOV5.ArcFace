import torch 
import torch.onnx
from torch.autograd import Variable

from iresnet import iresnet100

model = iresnet100(False)
model.load_state_dict(torch.load("backbone.pth"))
dummy_input = Variable(torch.randn(1, 3, 112, 112))
torch.onnx.export(model, dummy_input, "backbone.onnx")
