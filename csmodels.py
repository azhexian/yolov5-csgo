from pyexpat import model
import torch
from models.common import DetectMultiBackend
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half =device !='cpu'
weights='./best.pt'
imgsz=640

def loal_model():
    model = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    model.model.half() if half else model.model.float()
    return model
