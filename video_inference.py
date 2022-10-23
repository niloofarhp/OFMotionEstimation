import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
from easydict import EasyDict
from torchvision import transforms
from transforms import sep_transforms

from utils.flow_utils import flow_to_image, resize_flow
from utils.torch_utils import restore_model
from models.pwclite import PWCLite


class TestHelper():
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            sep_transforms.Zoom(*self.cfg.test_shape),
            sep_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    def init_model(self):
        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='checkpoints/KITTI12/pwclite_ar.tar')
    parser.add_argument('-s', '--test_shape', default=[384, 640], type=int, nargs=2)
    parser.add_argument('-i', '--img_list', nargs='+',
                        default=['examples/img1.png', 'examples/img2.png'])
    args = parser.parse_args()

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': 2,#len(args.img_list),
            'reduce_dense': True
        },
        'pretrained_model': "./checkpoints/Sintel/pwclite_ar.tar",#"/home/niloofarhp/Documents/Projects/ARFlow/outputs/checkpoints/221022/151459/Sintel_model_best.pth.tar", # args.model,
        'test_shape': args.test_shape,
    }


    #test inference on video
    ts = TestHelper(cfg)
    cap = cv.VideoCapture("shibuya.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)
    count=0
    img_list = []
    outPutFrames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if(not ret):
            break
        count+=1
        img_list.append(frame)
        if(count == 2):
            imgs = [img.astype(np.float32) for img in img_list]
            h, w = imgs[0].shape[:2]
            flow_12 = ts.run(imgs)['flows_fw'][0]
            flow_12 = resize_flow(flow_12, (h, w))
            np_flow_12 = flow_12[0].detach().cpu().numpy().transpose([1, 2, 0])
            vis_flow = flow_to_image(np_flow_12)
            #fig = plt.figure()
            #plt.imshow(vis_flow)
            #plt.show()
            outPutFrames.append(vis_flow)
            #cv.imshow("dense optical flow", vis_flow)
            count = 1
            img_list[0] = img_list[1]
            img_list.pop()
    fourcc = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv.VideoWriter("outputVideo_beforeTrain.mp4",fourcc, fps, (vis_flow.shape[1],vis_flow.shape[0]))  
    for i in range(len(outPutFrames)):
       out.write(outPutFrames[i])
    out.release()      