import sys
sys.path.append('..')
import os
import json
import numpy as np
import time
import torch
import torch.nn.functional as F
from config.cfg import parse
from metric.eval_metric import calc_mAPJ


def non_maximum_suppression(heatmap):
    max_heatmap = F.max_pool2d(heatmap, 3, stride=1, padding=1)
    mask = heatmap == max_heatmap
    weight = torch.ones_like(mask) * 0.6
    weight[mask] = 1.0
    heatmap = weight * heatmap
    return heatmap


def calc_junction(jloc, joff, thresh=1e-2, top_K=1000):
    jloc = torch.from_numpy(jloc)
    joff = torch.from_numpy(joff)

    jloc = non_maximum_suppression(jloc)

    H, W = jloc.shape[-2:]
    score = jloc.flatten()
    joff = joff.reshape(2, -1).t()

    num = min(int((score >= thresh).sum().item()), top_K)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // W, indices % W
    junc = torch.stack([x, y], dim=1) + joff[indices] + 0.5

    junc[:, 0] = junc[:, 0].clamp(min=0, max=W - 1e-4)
    junc[:, 1] = junc[:, 1].clamp(min=0, max=H - 1e-4)

    junc = junc.numpy()
    score = score.numpy()

    return junc, score


def eval_mAPJ(gt_file, pred_file):
    with open(gt_file, 'r') as f:
        gt_annotations = json.load(f)
    with open(pred_file, 'r') as f:
        pred_annotations = json.load(f)

    junc_gts, junc_preds, junc_scores, im_ids = [], [], [], []
    for i, (gt_ann, pred_ann) in enumerate(zip(gt_annotations, pred_annotations)):
        sx, sy = 128.0 / gt_ann['width'], 128.0 / gt_ann['height']

        junc_gt = np.asarray(gt_ann['junc'])
        junc_gt[:, 0] = junc_gt[:, 0] * sx
        junc_gt[:, 1] = junc_gt[:, 1] * sy
        junc_gts.append(junc_gt)

        jloc_pred = np.asarray(pred_ann['jloc_pred'])
        joff_pred = np.asarray(pred_ann['joff_pred'])
        junc_pred, junc_score = calc_junction(jloc_pred, joff_pred)
        if len(junc_pred):
            junc_preds.append(junc_pred)
            junc_scores.append(junc_score)
            im_ids.append(np.array([i] * junc_pred.shape[0], dtype=np.int32))

    junc_preds = np.concatenate(junc_preds)
    junc_scores = np.concatenate(junc_scores)
    im_ids = np.concatenate(im_ids)
    indices = np.argsort(-junc_scores)
    junc_preds = junc_preds[indices]
    im_ids = im_ids[indices]

    mAPJ, P, R = calc_mAPJ(junc_gts, junc_preds, im_ids, [0.5, 1.0, 2.0])
    return mAPJ, P, R


if __name__ == '__main__':
    # Parameter
    os.chdir('..')
    cfg = parse()

    # Path
    gt_file = os.path.join(cfg.dataset_path, 'test.json')
    pred_file = os.path.join(cfg.output_path, 'result.json')

    start = time.time()
    mAPJ, P, R = eval_mAPJ(gt_file, pred_file)
    print(f'mAPJ: {mAPJ:.1f} | P: {P:.1f} | R: {R:.1f}')
    end = time.time()
    print('Time: %f s' % (end - start))
