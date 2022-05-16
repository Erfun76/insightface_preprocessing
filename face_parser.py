import torch
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import facer
from PIL import Image
import cv2


def read_hwc(cv2_im) -> torch.Tensor:
    img = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    np_image = np.array(image.convert('RGB'))
    return torch.from_numpy(np_image)


class FaceParser():
    def __init__(self):
        self.face_parser = facer.face_parser('farl/lapa/448', device=device)

    def parse(self, image, face_detection):
        image = read_hwc(image).unsqueeze(0).permute(0, 3, 1, 2).to(device=device)
        with torch.inference_mode():
            faces = self.face_parser(image, face_detection)
        seg_logits = faces[0]['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        label = np.array(seg_probs.argmax(dim=1).float().cpu()[0])
        return self.__face_state(label)

    def __face_state(self, label, eye_thresh=3, lip_thresh=5):
        left_eye_label = 5
        right_eye_label = 4
        upper_lip_label = 7
        lower_lip_label = 9
        state = {"mouth_coverage":False, "eyes_coverage":False}

        if (np.sum(label==left_eye_label)<eye_thresh) & (np.sum(label==right_eye_label)<eye_thresh):
            state["eyes_coverage"] = True
        
        if (np.sum(label==upper_lip_label)<lip_thresh) & (np.sum(label==lower_lip_label)<lip_thresh):
            state["mouth_coverage"] = True

        return state