from insightface.app import FaceAnalysis
import numpy as np
import cv2
from skimage import transform as trans
import os
from tqdm import tqdm
from auto_capture import CameraMetric
from sklearn.cluster import DBSCAN
from collections import Counter
from face_parser import FaceParser
import torch


# face alignment keypoints mapping placement in 112*112 crop
src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(src, axis=0)

# crop image after alignment
def norm_crop(img, landmark, image_size=112, mode='arcface'):
        M, pose_index = estimate_norm(landmark, image_size, mode)
        # crop face with given size
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)

    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]

    # loop over different mapping(in our case it is always one mapping)
    for i in np.arange(src.shape[0]):
        # mapping estimation
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

# this function is used to check thresholds
def validate_image(similarity, face_parsing, brightness, sharpness):
    if (similarity < similarity_thresh) | (brightness > brightness_max_thresh) | \
        (brightness < brightness_min_thresh) | (sharpness < sharpness_min_thresh) | \
        (face_parsing['mouth_coverage']) | (face_parsing['eyes_coverage']):
        return False
    else: 
        return True
        

# initialize face detector and face matcher
app = FaceAnalysis(name = 'buffalo_s', allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640,640))
# initialize camera metric (it is same library used in auto capture)
imageMetric = CameraMetric()
# this library is used to detect obstacle in front of face. (https://github.com/FacePerceiver/facer.git)
parser = FaceParser()

# HyperParmeter
dataset_path = 'dataset' # input dataset path
dataset_output = 'dataset_112x112_cleaned_2' # path to output valid faces
dataset_temp = 'dataset_112x112_temp_2' # path to invalid faces
# refer to autocapture module
brightness_max_thresh = 200 
brightness_min_thresh = 60
sharpness_min_thresh = 90
similarity_thresh = 0.2 # least similarity to validate a face belongs to same person class
mode = 0 # mode 0 just distance ,mode 1 clustering
dist_threshold = 0.6 # threshold for clustering mode 1

pbar = tqdm(total=len(os.listdir(dataset_path)))
# loop over classes(different persons)
for i, f1 in enumerate(os.listdir(dataset_path)):
    dir = os.path.join(dataset_path, f1)
    # make output path
    output_dir = os.path.join(dataset_output, f1)
    temp_dir = os.path.join(dataset_temp, f1)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    list_feats = []
    # loop over images in a class images
    for f2 in os.listdir(dir):
        im_path = os.path.join(dir, f2)
        im = cv2.imread(im_path)
        # detect face and extract feature
        faces = app.get(im)

        if faces: # check if there is any face detected
            feat = faces[0].normed_embedding # take first detection as this image detection
            feat = np.array(feat, dtype=np.float32)
            list_feats.append(feat) # add all feat of a class to a list to average over them an find class center
    if mode == 0:
      anchor_feat = np.mean(list_feats, axis=0) # feature that represents this class
    # find feature using clustering 
    else:
      clustering = DBSCAN(eps = dist_threshold, min_samples = 2).fit(list_feats)
      labels = clustering.labels_
      max_label = Counter(labels).most_common()[0][0]
      list_feats = np.array(list_feats)
      anchor_feat = np.mean(list_feats[labels==max_label], axis=0)

    # loop over images in a class images again to find nearest detection to feature that represents this class
    for f2 in os.listdir(dir):
        im_path = os.path.join(dir, f2)
        im = cv2.imread(im_path)
        faces = app.get(im)
        similarity_list = []
        if faces:
            # loop over faces in an image
            for face in faces:
                feat = face.normed_embedding
                similarity_list.append(np.matmul(anchor_feat, feat.T)) # calculate similarity with anchor feature 
            face_idx = np.argmax(similarity_list) # choose must similar
            face_sim = np.max(similarity_list)

            face1 = norm_crop(im, faces[face_idx]['kps'])
            # dictionary of face detection
            parser_dict = [{"points":torch.tensor([faces[face_idx]['kps']]).to(device='cuda'),\
                 "rects": torch.tensor(faces[face_idx]['bbox']).to(device='cuda'),\
                      "score":torch.tensor([faces[face_idx]['det_score']]).to(device='cuda')}]
            
            # parse faces
            face_parsing = parser.parse(im, parser_dict)

            brightness = imageMetric.calc_brightness(face1)
            sharpness = imageMetric.calc_sharpness(face1)

            if validate_image(face_sim, face_parsing, brightness, sharpness):
                output_file_dir = os.path.join(output_dir, f2)
                cv2.imwrite(output_file_dir,face1)
            else:
                temp_file_dir = os.path.join(temp_dir, f2)
                cv2.imwrite(temp_file_dir,face1)
        torch.cuda.empty_cache()
    pbar.update(1)
pbar.close()