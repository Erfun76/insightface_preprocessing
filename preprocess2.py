import sys
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from skimage import transform as trans
import os
from tqdm import tqdm
from auto_capture import CameraMetric
from recognition import FaceFeature
from yolo5.detector import Yolo5Detector
from sklearn.cluster import DBSCAN
from collections import Counter
from face_parser import FaceParser
import torch


src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(src, axis=0)

def validate_image(similarity, face_parsing, brightness, sharpness):
    if (similarity < similarity_thresh) | (brightness > brightness_max_thresh) | \
        (brightness < brightness_min_thresh) | (sharpness < sharpness_min_thresh) | \
        (face_parsing['mouth_coverage']) | (face_parsing['eyes_coverage']):
        return False
    else: 
        return True
        

if __name__ == '__main__':
    #app = FaceAnalysis(name = 'buffalo_s', allowed_modules=['detection', 'recognition'])
    #app.prepare(ctx_id=0, det_size=(640,640))
    face_detector = Yolo5Detector()
    face_feature_extractor = FaceFeature(face_detector)

    imageMetric = CameraMetric()
    parser = FaceParser()

    # HyperParmeter
    dataset_path = 'dataset'
    dataset_output = 'dataset_112x112_cleaned_3'
    dataset_temp = 'dataset_112x112_temp_3'
    brightness_max_thresh = 200
    brightness_min_thresh = 60
    sharpness_min_thresh = 90
    similarity_thresh = 0.5
    mode = 0 # mode 0 just distance ,mode 1 clustering
    dist_threshold = 0.6 # threshold for clustering mode 1

    pbar = tqdm(total=len(os.listdir(dataset_path)))
    for i, f1 in enumerate(os.listdir(dataset_path)):
        dir = os.path.join(dataset_path, f1)
        output_dir = os.path.join(dataset_output, f1)
        temp_dir = os.path.join(dataset_temp, f1)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        list_feats = []
        for f2 in os.listdir(dir):
            im_path = os.path.join(dir, f2)
            im = cv2.imread(im_path)
            feats, _ = face_feature_extractor.get(im)

            if feats:
                feat = feats[0]
                feat = np.array(feat, dtype=np.float32)
                list_feats.append(feat)
        if mode == 0:
            anchor_feat = np.mean(list_feats, axis=0)
        else:
            clustering = DBSCAN(eps = dist_threshold, min_samples = 2).fit(list_feats)
            labels = clustering.labels_
            max_label = Counter(labels).most_common()[0][0]
            list_feats = np.array(list_feats)
            anchor_feat = np.mean(list_feats[labels==max_label], axis=0)

        for f2 in os.listdir(dir):
            im_path = os.path.join(dir, f2)
            im = cv2.imread(im_path)
            feats, faces = face_feature_extractor.get(im)
            
            similarity_list = []
            if feats:
                for feat in feats:
                    similarity_list.append(np.matmul(anchor_feat, feat.T))
                face_idx = np.argmax(similarity_list)
                face_sim = np.max(similarity_list)

                face1 = face_feature_extractor.norm_crop(im, np.array(faces[face_idx]['kps']))
                # dictionary of face detection
                parser_dict = [{"points":torch.tensor([faces[face_idx]['kps']]).to(device='cuda'),\
                     "rects": torch.tensor(faces[face_idx]['bbox']).to(device='cuda'),\
                          "score":torch.tensor([float(faces[face_idx]['det_score'])]).to(device='cuda')}]

                face_parsing = parser.parse(im, parser_dict)
                brightness = imageMetric.calc_brightness(face1)
                sharpness = imageMetric.calc_sharpness(face1)

                if validate_image(face_sim, face_parsing, brightness, sharpness):
                    output_file_dir = os.path.join(output_dir, f2)
                    cv2.imwrite(output_file_dir,face1)
                else:
                    temp_file_dir = os.path.join(temp_dir, f2)
                    cv2.imwrite(temp_file_dir,im)
            torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()
