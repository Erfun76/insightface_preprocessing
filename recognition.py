import cv2
from skimage import transform as trans
import numpy as np
import torch
from numpy.linalg import norm as l2norm
from model.backbones import get_model


arcface_src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)


class FaceFeature():
    '''
    A module to extract features from face
    '''
    def __init__(self, face_detector, recognition_model_name = "r100",recognition_weight="./model/weights/best_model.pt" ,mode="arcface"):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        print(self.device)
        self.face_detector = face_detector
        self.net = get_model(recognition_model_name, fp16=False)
        self.net.load_state_dict(torch.load(recognition_weight, map_location=self.device))
        self.net.eval()
        self.net.to(self.device)
        self.mode = mode

    @torch.no_grad()
    def get(self, image):
        norm_features = []
        detections = self.face_detector.get(image)
        if detections:
            for detection in detections:
                d = np.array(detection['kps'])
                face_crop = self.norm_crop(image, d)
                img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).unsqueeze(0).float()
                img.div_(255).sub_(0.5).div_(0.5)
                img = img.to(self.device)
                feature = self.net(img).cpu().numpy()
                norm_features.append(feature[0]/l2norm(feature[0]))
            return norm_features, detections
        else: 
            return None

    def norm_crop(self, img, landmark, image_size=112):
        M, pose_index = self.__estimate_norm(landmark, image_size)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    # lmk is prediction; src is template
    def __estimate_norm(self, lmk, image_size=112):
        assert lmk.shape == (5, 2)

        tform = trans.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if self.mode  == 'arcface':
            if image_size == 112:
                src = arcface_src
            else:
                src = float(image_size) / 112 * arcface_src
        else:
            src = arcface_src # src_map[image_size]
        src = np.array([src])
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
            #         print(error)
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index
