import numpy as np
import cv2
import time 
import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
import torch.nn as nn
import torch.nn.functional as F
import os
import torchvision.models as models
import scipy 
from matplotlib.pyplot import imread
from PIL import Image

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=820,
    display_height=616,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epocdummy_input = torch.randn(32, 3, 256, 256, device='cuda')

#device = torch.device('cuda')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

classes = ['metal', 'paper', 'glass', 'trash', 'plastic', 'cardboard']


def predict_image(img, model):
    # Convert to a batch of 1
    #Does it have to be a batch of 1 (circle back here!!!)
    #Also check that the dimensions and tensors are the same.
    xb = to_device(img, device="cuda")
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


model = ResNet()
model.load_state_dict(torch.load('/home/ican/Downloads/newest_file', map_location="cuda:0"))
model.to(device)
model.eval()



prev_frame_time = 0
next_frame_time = 0



cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
ret = True



clip = []
while(ret):
    # Capture frame-by-frame
    #window_handle = cv2.namedWindow('iCAN_Detection', cv2.WINDOW_AUTOSIZE)
    ret, frame = cap.read()
    if not ret and frame is None: 
        continue

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #image = imread(gray, cv2.IMREAD_COLOR)
    image = Image.fromarray(frame)
    #The following code is to calculate frame_rate
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time

    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 

    #Now unto copied code where I understand but little
    

    tmp_ = center_crop(cv2.resize(frame, (256, 256)))
    tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
    clip.append(tmp)
    if len(clip) == 32:
        inputs = np.array(clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, (0, 1,2,0)) #trasnpose using the expected number of inputs
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad(): 
            outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 1)
            clip.pop(0)

      


    font = cv2.FONT_HERSHEY_SIMPLEX 
  
    # puting the FPS count on the frame 
    cv2.putText(gray, fps, (7, 70), font, 3, (255, 255, 0))
   # cv2.imshow(frame, gray) #requirere another distribution 
    #mat this other one was gray. 
    #cv2.waitKey(30)

    # Display the resulting frame
    cv2.imshow('image', gray)
   #print(predict_image(frame, model))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break






# When everything done, release the capture
cap.release()