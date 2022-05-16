from insightface.app import FaceAnalysis
import numpy as np
import cv2
from skimage import transform as trans
import os

src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(src, axis=0)

def norm_crop(img, landmark, image_size=112, mode='arcface'):
        M, pose_index = estimate_norm(landmark, image_size, mode)
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

app = FaceAnalysis(name = 'buffalo_s', allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640,640))

dataset_path = 'dataset'
dataset_output = 'dataset_112x112'
for i, f1 in enumerate(os.listdir(dataset_path)):
    dir = os.path.join(dataset_path, f1)
    output_dir = os.path.join(dataset_output, f1)
    os.mkdir(output_dir)
    for f2 in os.listdir(dir):
        im_path = os.path.join(dir, f2)
        im = cv2.imread(im_path)
        faces = app.get(im)
        if faces:
            face1 = norm_crop(im, faces[0]['kps'])
            output_file_dir = os.path.join(output_dir, f2)
            cv2.imwrite(output_file_dir,face1)
