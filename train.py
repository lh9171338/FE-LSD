import os
import numpy as np
import copy
import random
import json
import shutil
import tqdm
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from network.build import build_model
from network.dataset import Dataset
from config.cfg import parse
from metric.eval_mAPJ import eval_mAPJ
from metric.eval_sAP import eval_sAP
import warnings
warnings.filterwarnings('ignore')


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


def train(model, loader, cfg, device):
    # Option
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, amsgrad=True)
    if cfg.last_epoch != -1:
        print('Load pretrained model...')
        checkpoint_file = os.path.join(cfg.model_path, cfg.model_name)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, last_epoch=cfg.last_epoch)

    # Summary
    writer = SummaryWriter(cfg.log_path)

    # Train
    step = (cfg.last_epoch + 1) * len(loader['train'].dataset) // cfg.train_batch_size + 1
    step_ = (cfg.last_epoch + 1) * len(loader['train'].dataset) + cfg.train_batch_size
    best_sAP = [0 for _ in range(5)]
    best_state_dict = None
    for epoch in range(cfg.last_epoch + 1, cfg.num_epochs):
        # Train
        model.train()

        for images, annotations in tqdm.tqdm(loader['train'], desc='train: '):
            images, annotations = images.to(device), to_device(annotations, device)

            loss_dict, labels, scores = model(images, annotations)
            loss = sum([cfg.loss_weights[k] * loss_dict[k] for k in cfg.loss_weights.keys()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Visualize
            step_ = step * cfg.train_batch_size
            if step % cfg.print_freq == 0:
                lr = scheduler.get_last_lr()[0]

                score = scores.detach().cpu().numpy() > 0.5
                label = labels.detach().cpu().numpy() > 0.5
                tn, fp, fn, tp = confusion_matrix(label, score).ravel()

                msg = f'epoch: {epoch}/{cfg.num_epochs} | lr: {lr:e} | loss: {loss.item():6f} |'
                for key, value in loss_dict.items():
                    msg += f' {key}: {value.item():6f} |'
                print(msg)
                print(f'tp: {tp} tn: {tn} fp: {fp} fn: {fn}')

                writer.add_scalar('lr', lr, step_)
                writer.add_scalar('loss', loss, step_)
                for key, value in loss_dict.items():
                    writer.add_scalar(key, value, step_)

            step += 1
        scheduler.step()

        if epoch % cfg.save_freq == 0:
            # Save model
            save_path = os.path.join(cfg.model_path, f'{os.path.splitext(cfg.model_name)[0]}-{epoch:03d}')
            os.makedirs(save_path, exist_ok=True)
            checkpoint_file = os.path.join(cfg.model_path, cfg.model_name)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, checkpoint_file)

            # Val
            model.eval()

            results = []
            for images, annotations in tqdm.tqdm(loader['val'], desc='val: '):
                images, annotations = images.to(device), to_device(annotations, device)

                outputs = model(images, annotations)
                for output in outputs:
                    for k in output.keys():
                        if isinstance(output[k], torch.Tensor):
                            output[k] = output[k].tolist()
                    results.append(output)

            with open(os.path.join(save_path, 'result.json'), 'w') as f:
                json.dump(results, f)

            gt_file = os.path.join(cfg.dataset_path, 'test.json')
            pred_file = os.path.join(save_path, 'result.json')
            mAPJ, PJ, RJ = eval_mAPJ(gt_file, pred_file)
            print(f'mAPJ: {mAPJ:.1f} | {PJ:.1f} | {RJ:.1f}')
            msAP, P, R, sAP = eval_sAP(gt_file, pred_file)
            print(f'msAP: {msAP:.1f} | {P:.1f} | {R:.1f} | {sAP[0]:.1f} | {sAP[1]:.1f} | {sAP[2]:.1f}')
            writer.add_scalar('mAPJ', mAPJ, step_)
            writer.add_scalar('msAP', msAP, step_)
            shutil.rmtree(save_path)

            if msAP > best_sAP[3]:
                best_sAP = [mAPJ, PJ, RJ, msAP, P, R, *sAP]
                best_state_dict = copy.deepcopy(model.state_dict())
            msg = f'best msAP: {best_sAP[0]:.1f} | {best_sAP[1]:.1f} | {best_sAP[2]:.1f} | ' \
                  f'{best_sAP[3]:.1f} | {best_sAP[4]:.1f} | {best_sAP[5]:.1f} | ' \
                  f'{best_sAP[6]:.1f} | {best_sAP[7]:.1f} | {best_sAP[8]:.1f}'
            print(msg)

    writer.close()

    # Save best model
    model_filename = os.path.join(cfg.model_path, cfg.model_name)
    torch.save(best_state_dict, model_filename)


if __name__ == '__main__':
    # Parameter
    cfg = parse()
    os.makedirs(cfg.model_path, exist_ok=True)

    # Use GPU or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    use_gpu = cfg.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:0' if use_gpu else 'cpu')
    print('use_gpu: ', use_gpu)

    # Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    if use_gpu:
        torch.cuda.manual_seed_all(cfg.seed)

    # Load model
    model = build_model(cfg).to(device)
    if cfg.pretrained_model_name != '':
        pretrained_model_filename = os.path.join(cfg.model_path, cfg.pretrained_model_name)
        print(f'Loading pretrained model: {pretrained_model_filename}')
        checkpoint = torch.load(pretrained_model_filename, map_location=device)
        if 'model' in checkpoint.keys():
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)

    # Load dataset
    train_dataset = Dataset(cfg, split='train')
    val_dataset = Dataset(cfg, split='test')
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=cfg.train_batch_size,
                                   num_workers=cfg.num_workers, shuffle=True, collate_fn=train_dataset.collate)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=cfg.test_batch_size,
                                 num_workers=cfg.num_workers, shuffle=False, collate_fn=train_dataset.collate)
    loader = {'train': train_loader, 'val': val_loader}

    # Train network
    train(model, loader, cfg, device)
