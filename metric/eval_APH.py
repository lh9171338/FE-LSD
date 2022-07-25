import sys
sys.path.append('..')
import os
import json
import shutil
import subprocess
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import interpolate
from config.cfg import parse
from metric.eval_metric import plot_pr_curve


output_size = 128
thresh = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]


def eval_APH(gt_file, pred_file, image_path, cfg):
    run_all = True
    figure_path = cfg.figure_path
    name = os.path.splitext(cfg.model_name)[0]
    output_file = os.path.join(figure_path, f'temp-{name}.mat')

    if run_all:
        temp_path = os.path.join(figure_path, 'temp')
        os.makedirs(temp_path, exist_ok=True)
        print(f'intermediate matlab results will be saved at: {temp_path}')

        with open(gt_file) as f:
            gt_annotations = json.load(f)
        with open(pred_file) as f:
            pred_annotations = json.load(f)

        for gt_ann in gt_annotations:
            filename = gt_ann['filename']
            mat_name = os.path.splitext(filename)[0] + '.mat'
            lines = np.asarray(gt_ann['lines']).reshape((-1, 2, 2))
            sx, sy = output_size / gt_ann['width'], output_size / gt_ann['height']
            lines[:, :, 0] = lines[:, :, 0] * sx
            lines[:, :, 1] = lines[:, :, 1] * sy
            os.makedirs(os.path.join(temp_path, 'gt'), exist_ok=True)
            sio.savemat(os.path.join(temp_path, 'gt', mat_name), {'lines': lines.reshape(-1, 4)})

        for t in thresh:
            for pred_ann in pred_annotations:
                filename = pred_ann['filename']
                mat_name = os.path.splitext(filename)[0] + '.mat'
                lines = np.asarray(pred_ann['line_pred'])
                scores = np.asarray(pred_ann['line_score'])
                sx, sy = output_size / pred_ann['width'], output_size / pred_ann['height']
                if len(lines):
                    lines[:, :, 0] = lines[:, :, 0] * sx
                    lines[:, :, 1] = lines[:, :, 1] * sy
                idx = np.where(scores > t)[0]
                os.makedirs(os.path.join(temp_path, str(t)), exist_ok=True)
                sio.savemat(os.path.join(temp_path, str(t), mat_name), {'lines': lines[idx].reshape(-1, 4)})

        cmd = 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; '
        cmd += "eval_release('{:s}', '{:s}', '{:s}', {:d}); quit;\"".format(
            image_path, temp_path, output_file, output_size
        )
        print('Running:\n{}'.format(cmd))
        os.environ['MATLABPATH'] = 'metric/matlab/'
        subprocess.call(cmd, shell=True)
        shutil.rmtree(temp_path)

    mat = sio.loadmat(output_file)
    tps = mat['sumtp'][:, 0]
    fps = mat['sumfp'][:, 0]
    N = mat['sumgt'][:, 0]

    rcs = (tps / N)
    prs = (tps / (tps + fps))
    mask = np.logical_not(np.isnan(prs))
    rcs = rcs[mask]
    prs = prs[mask]
    indices = np.argsort(rcs)
    rcs = np.sort(rcs[indices])
    prs = np.sort(prs[indices])[::-1]

    recall = np.concatenate(([0.0], rcs, [1.0]))
    precision = np.concatenate(([0.0], prs, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    APH = np.sum((recall[i + 1] - recall[i]) * precision[i + 1]) * 100
    FH = (2 * np.array(prs) * np.array(rcs) / (np.array(prs) + np.array(rcs))).max() * 100

    f = interpolate.interp1d(rcs, prs, kind='linear', bounds_error=False, fill_value='extrapolate')
    x = np.arange(0, 1, 0.01) * rcs[-1]
    y = f(x)

    figure = plot_pr_curve(x, y, title='AP${^{H}}$', legend=[f'{name}={FH:.1f}'])
    figure.savefig(os.path.join(figure_path, f'APH-{name}.pdf'), format='pdf', bbox_inches='tight')
    sio.savemat(os.path.join(figure_path, f'APH-{name}.mat'), {'rcs': x, 'prs': y, 'AP': APH})
    plt.show()

    return APH, FH


if __name__ == "__main__":
    # Parameter
    os.chdir('..')
    cfg = parse()

    gt_file = os.path.join(cfg.dataset_path, 'test.json')
    test_file = os.path.join(cfg.output_path, 'result.json')
    image_path = os.path.join(cfg.dataset_path, 'images')

    start = time.time()
    APH, FH = eval_APH(gt_file, test_file, image_path, cfg)
    print(f'APH: {APH:.1f} | FH: {FH:.1f}')
    end = time.time()
    print('Time: %f s' % (end - start))
