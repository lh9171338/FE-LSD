import os
import json
import cv2
import time
import torch
import tqdm
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch.cuda.amp import autocast as autocast
from network.build import build_model
from network.dataset import Dataset
from config.cfg import parse
from metric.eval_mAPJ import eval_mAPJ
from metric.eval_sAP import eval_sAP


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        return data
    if isinstance(data, list):
        return [to_device(d, device) for d in data]


def convert_model(model, state_dict):
    new_state_dict = model.state_dict()
    for key, value in state_dict.items():
        try:
            C = len(value)
        except:
            continue
        if 'shallow_res1' in key:
            new_key = key.replace('shallow_res1', 'shallow_res')
            new_state_dict[new_key][:C] = value
        elif 'shallow_res2' in key:
            new_key = key.replace('shallow_res2', 'shallow_res')
            new_state_dict[new_key][C:] = value
        elif 'encoders1' in key:
            new_key = key.replace('encoders1', 'encoders')
            new_state_dict[new_key][:C] = value
        elif 'encoders2' in key:
            new_key = key.replace('encoders2', 'encoders')
            new_state_dict[new_key][C:] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def save_lines(image, lines, filename, plot=False):
    height, width = image.shape[:2]

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, width - 0.5])
    plt.ylim([height - 0.5, -0.5])
    plt.imshow(image[:, :, ::-1])
    for pts in lines:
        pts = pts - 0.5
        plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.5)
        plt.scatter(pts[:, 0], pts[:, 1], color="#33FFFF", s=1.2, edgecolors="none", zorder=5)

    plt.savefig(filename, dpi=height, bbox_inches=0)
    if plot:
        plt.show()
    plt.close()


def test(model, loader, cfg, device):
    # Test
    model.eval()

    results = []
    start = time.time()

    for images, annotations in tqdm.tqdm(loader, desc='test: '):
        images, annotations = images.to(device), to_device(annotations, device)

        with autocast():
            outputs = model(images, annotations)

        for output in outputs:
            # Save image
            if cfg.save_image:
                if len(output['line_pred']):
                    line_pred = output['line_pred'].detach().cpu().numpy()
                    line_score = output['line_score'].detach().cpu().numpy()
                    filename = output['filename']

                    if cfg.with_clear:
                        src_file = os.path.join(cfg.dataset_path, 'images-clear', filename)
                        dst_file = os.path.join(cfg.output_path, 'images-clear', filename)
                    else:
                        src_file = os.path.join(cfg.dataset_path, 'images-blur', filename)
                        dst_file = os.path.join(cfg.output_path, 'images-blur', filename)
                    image = cv2.imread(src_file)
                    mask = line_score > cfg.score_thresh
                    line_pred = line_pred[mask]
                    save_lines(image, line_pred, dst_file)

            if cfg.evaluate:
                for k in output.keys():
                    if isinstance(output[k], torch.Tensor):
                        output[k] = output[k].tolist()
                results.append(output)

    end = time.time()
    if cfg.evaluate:
        with open(os.path.join(cfg.output_path, 'result.json'), 'w') as f:
            json.dump(results, f)

        print(f'FPS: {len(loader) / (end - start):.1f}')

        gt_file = os.path.join(cfg.dataset_path, 'test.json')
        pred_file = os.path.join(cfg.output_path, 'result.json')
        mAPJ, P, R = eval_mAPJ(gt_file, pred_file)
        msAP, P, R, sAP = eval_sAP(gt_file, pred_file, cfg)
        print(f'metric: {sAP[0]:.1f} | {sAP[1]:.1f} | {sAP[2]:.1f} | {msAP:.1f} | {mAPJ:.1f}')


if __name__ == '__main__':
    # Parameter
    cfg = parse()

    os.makedirs(cfg.output_path, exist_ok=True)
    os.makedirs(cfg.figure_path, exist_ok=True)
    if cfg.save_image:
        if cfg.with_clear:
            os.makedirs(os.path.join(cfg.output_path, 'images-clear'), exist_ok=True)
        else:
            os.makedirs(os.path.join(cfg.output_path, 'images-blur'), exist_ok=True)

    # Use GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:0' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Load model
    model = build_model(cfg).to(device)
    model_filename = os.path.join(cfg.model_path, cfg.model_name)
    checkpoint = torch.load(model_filename, map_location=device)
    if 'model' in checkpoint.keys():
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    try:
        model.load_state_dict(state_dict, strict=True)
    except:
        state_dict = convert_model(model, state_dict)
        model.load_state_dict(state_dict, strict=True)

    # Load dataset
    dataset = Dataset(cfg, split='test')
    loader = Data.DataLoader(dataset=dataset, batch_size=cfg.test_batch_size, num_workers=cfg.num_workers,
                             shuffle=False, collate_fn=dataset.collate, pin_memory=True)

    # Test network
    test(model, loader, cfg, device)
