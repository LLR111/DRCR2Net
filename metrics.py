import os
from PIL import Image
import numpy as np
from sklearn.metrics import recall_score, precision_score
import pydensecrf.densecrf as dcrf

def cal_precision_recall_mae(mask_path,gt_path):
    files = os.listdir(gt_path)
    masks,gts = [],[]
    for file in files:
        mask_file = os.path.join(mask_path,file)
        gt_file = os.path.join(gt_path,file)
        mask = np.array(Image.open(mask_file))/255.
        gt = np.array(Image.open(gt_file))/255.
        masks.extend(mask.ravel())
        gts.extend(gt.ravel())
        print(file)

    RECALL = recall_score(gts, masks)
    PERC = precision_score(gts, masks)
    fmeasure = (1 + 0.3) * PERC * RECALL / (0.3 * PERC + RECALL)
    masks = np.array(masks)
    gts = np.array(gts)
    mae = np.mean(np.abs((gts - masks)))

    return fmeasure,mae

def crf_refine(img, annos):
    print(img.shape)
    print(annos.shape)
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape
    EPSILON = 1e-8
    M = 2
    tau = 1.05
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)
    anno_norm = annos / 255.
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))
    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]
    res = (res * 255).astype(np.uint8)
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')

