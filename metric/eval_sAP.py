import sys
sys.path.append('..')
import os
import json
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from config.cfg import parse
from metric.eval_metric import calc_msAP, calc_sAP, plot_pr_curve


def eval_sAP(gt_file, pred_file, cfg=None):
    with open(gt_file, 'r') as f:
        gt_annotations = json.load(f)
    with open(pred_file, 'r') as f:
        pred_annotations = json.load(f)

    line_gts, line_preds, line_scores, im_ids = [], [], [], []
    for i, (gt_ann, pred_ann) in enumerate(zip(gt_annotations, pred_annotations)):
        sx, sy = 128.0 / gt_ann['width'], 128.0 / gt_ann['height']

        line_gt = np.asarray(gt_ann['lines']).reshape(-1, 2, 2)
        line_gt[:, :, 0] = line_gt[:, :, 0] * sx
        line_gt[:, :, 1] = line_gt[:, :, 1] * sy
        line_gts.append(line_gt)

        line_pred = np.asarray(pred_ann['line_pred'])
        line_score = np.asarray(pred_ann['line_score'])
        if len(line_pred):
            line_pred[:, :, 0] = line_pred[:, :, 0] * sx
            line_pred[:, :, 1] = line_pred[:, :, 1] * sy
            line_preds.append(line_pred)
            line_scores.append(line_score)
            im_ids.append(np.array([i] * line_pred.shape[0], dtype=np.int32))

    line_preds = np.concatenate(line_preds)
    line_scores = np.concatenate(line_scores)
    im_ids = np.concatenate(im_ids)
    indices = np.argsort(-line_scores)
    line_preds = line_preds[indices]
    im_ids = im_ids[indices]

    msAP, P, R, sAP = calc_msAP(line_gts, line_preds, im_ids, [5.0, 10.0, 15.0])

    if cfg is not None:
        figure_path = cfg.figure_path
        name = os.path.splitext(cfg.model_name)[0]
        sAP10, _, _, rcs, prs = calc_sAP(line_gts, line_preds, im_ids, 10.0)
        figure = plot_pr_curve(rcs, prs, title='sAP${^{10}}$', legend=[f'{name}={sAP10:.1f}'],)
        figure.savefig(os.path.join(figure_path, f'sAP10-{name}.pdf'), format='pdf', bbox_inches='tight')
        sio.savemat(os.path.join(figure_path, f'sAP10-{name}.mat'), {'rcs': rcs, 'prs': prs, 'AP': sAP10})
        plt.show()

    return msAP, P, R, sAP


if __name__ == "__main__":
    # Parameter
    os.chdir('..')
    cfg = parse()
    os.makedirs(cfg.figure_path, exist_ok=True)

    # Path
    gt_file = os.path.join(cfg.dataset_path, 'test.json')
    test_file = os.path.join(cfg.output_path, 'result.json')
    start = time.time()
    msAP, P, R, sAP = eval_sAP(gt_file, test_file, cfg)
    print(f'msAP: {msAP:.1f} | P: {P:.1f} | R: {R:.1f} | sAP5: {sAP[0]:.1f} | sAP10: {sAP[1]:.1f} | sAP15: {sAP[2]:.1f}')
    end = time.time()
    print('Time: %f s' % (end - start))
