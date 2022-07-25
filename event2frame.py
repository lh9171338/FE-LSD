import os
import cv2
import glob
import numpy as np
import argparse
from lh_tool.Iterator import SingleProcess, MultiProcess


def Event2Image(image_file, event_file, dst_file, cfg):
    assert cfg.dim % 2 == 0, 'Wrong dim'
    num_bins = cfg.dim // 2
    scale = cfg.scale
    plot = cfg.plot

    image = cv2.imread(image_file)
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
    event = np.zeros((height, width, 2, num_bins), dtype=np.float32)

    with np.load(event_file) as events:
        ts = events['t'].astype(np.float32)
        ts = (num_bins - 1) * ts / ts.max()
        xs = events['x']
        ys = events['y']
        ps = events['p'].astype(np.int8)
    mask = np.logical_and(np.logical_and(xs >= 0, xs < width), np.logical_and(ys >= 0, ys < height))
    ts = ts[mask]
    xs = xs[mask]
    ys = ys[mask]
    ps = ps[mask]

    ts0 = np.clip(np.floor(ts), a_min=0, a_max=num_bins-1)
    ts1 = np.clip(ts0 + 1, a_min=0, a_max=num_bins-1)
    ts0i, ts1i = np.int32(ts0), np.int32(ts1)

    np.add.at(event, (ys, xs, ps, ts0i), ts1 - ts)
    np.add.at(event, (ys, xs, ps, ts1i), ts - ts0)
    neg = event[:, :, 0, -1]
    pos = event[:, :, 1, -1]
    event_ = np.stack((neg, np.zeros_like(neg), pos), axis=-1)
    event_ = (((event_ / event_.max()) ** 0.5) * 255).astype(np.uint8)

    # save result
    dst_file_ = dst_file.replace('.npz', '.png')
    cv2.imwrite(dst_file_, event_)

    # plot
    if plot:
        cv2.namedWindow('event', 0)
        cv2.imshow('event', event_)
        cv2.waitKey()


def Event2ECSAE(image_file, event_file, dst_file, cfg):
    assert cfg.dim in [2, 4], 'Wrong dim'
    polarity = cfg.dim == 4
    scale = cfg.scale
    plot = cfg.plot

    image = cv2.imread(image_file)
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)

    if polarity:
        event_count = np.zeros((height, width, 2), dtype=np.float32)
        event_time = np.zeros((height, width, 2), dtype=np.float32)

        with np.load(event_file) as events:
            ts = events['t'].astype(np.float32)
            xs = events['x']
            ys = events['y']
            ps = events['p'].astype(np.int8)
        mask = np.logical_and(np.logical_and(xs >= 0, xs < width), np.logical_and(ys >= 0, ys < height))
        ts = ts[mask]
        xs = xs[mask]
        ys = ys[mask]
        ps = ps[mask]

        np.add.at(event_count, (ys, xs, ps), 1)
        event_time[ys, xs, ps] = ts
        event_count /= event_count.max()
        event_time /= event_time.max()
        event = np.concatenate((event_count, event_time), axis=-1)

        # save result
        np.savez(dst_file, event=event)

        # plot
        if plot:
            event_count = np.concatenate((event[:, :, 1:2], np.zeros((height, width, 1)), event[:, :, 0:1]), axis=-1)
            event_time = np.concatenate((event[:, :, 3:4], np.zeros((height, width, 1)), event[:, :, 2:3]), axis=-1)
            cv2.namedWindow('event count', 0)
            cv2.namedWindow('event time', 0)
            cv2.imshow('event count', event_count)
            cv2.imshow('event time', event_time)
            cv2.waitKey()
    else:
        event_count = np.zeros((height, width), dtype=np.float32)
        event_time = np.zeros((height, width), dtype=np.float32)

        with np.load(event_file) as events:
            ts = events['t'].astype(np.float32)
            xs = events['x']
            ys = events['y']
        mask = np.logical_and(np.logical_and(xs >= 0, xs < width), np.logical_and(ys >= 0, ys < height))
        ts = ts[mask]
        xs = xs[mask]
        ys = ys[mask]

        np.add.at(event_count, (ys, xs), 1)
        event_time[ys, xs] = ts
        event_count /= event_count.max()
        event_time /= event_time.max()
        event = np.stack((event_count, event_time), axis=-1)

        # save result
        np.savez(dst_file, event=event)

        # plot
        if plot:
            cv2.namedWindow('event count', 0)
            cv2.namedWindow('event time', 0)
            cv2.imshow('event count', event_count)
            cv2.imshow('event time', event_time)
            cv2.waitKey()


def Event2VoxelGrid(image_file, event_file, dst_file, cfg):
    num_bins = cfg.dim
    scale = cfg.scale
    plot = cfg.plot

    image = cv2.imread(image_file)
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
    event = np.zeros((height, width, num_bins), dtype=np.float32)

    with np.load(event_file) as events:
        ts = events['t'].astype(np.float32)
        ts = (num_bins - 1) * ts / ts.max()
        xs = events['x']
        ys = events['y']
        ps = events['p'].astype(np.int8)
        ps[ps == 0] = -1
    mask = np.logical_and(np.logical_and(xs >= 0, xs < width), np.logical_and(ys >= 0, ys < height))
    ts = ts[mask]
    xs = xs[mask]
    ys = ys[mask]
    ps = ps[mask]

    ts0 = np.clip(np.floor(ts), a_min=0, a_max=num_bins-1)
    ts1 = np.clip(ts0 + 1, a_min=0, a_max=num_bins-1)
    ts0i, ts1i = np.int32(ts0), np.int32(ts1)

    np.add.at(event, (ys, xs, ts0i), ps * (ts1 - ts))
    np.add.at(event, (ys, xs, ts1i), ps * (ts - ts0))
    event /= event.max()

    # normalize
    mask = np.nonzero(event)
    mean, std = event[mask].mean(), event[mask].std()
    event[mask] = (event[mask] - mean) / std

    # save result
    np.savez(dst_file, event=event)

    # plot
    if plot:
        for i in range(num_bins):
            mask = event[:, :, i] > 0
            pos_image = event[:, :, i]
            neg_image = np.abs(event[:, :, i])
            pos_image[~mask] = 0
            neg_image[mask] = 0
            image = np.concatenate((neg_image[..., None], np.zeros((height, width, 1)), pos_image[..., None]), axis=-1)
            cv2.namedWindow(f'voxel grid {i + 1}', 0)
            cv2.imshow(f'voxel grid {i + 1}', image)
        cv2.waitKey()


def Event2EST(image_file, event_file, dst_file, cfg):
    assert cfg.dim % 2 == 0, 'Wrong dim'
    num_bins = cfg.dim // 2
    scale = cfg.scale
    plot = cfg.plot

    image = cv2.imread(image_file)
    height, width = int(image.shape[0] * scale), int(image.shape[1] * scale)
    event = np.zeros((height, width, 2, num_bins), dtype=np.float32)

    with np.load(event_file) as events:
        ts = events['t'].astype(np.float32)
        ts = (num_bins - 1) * ts / ts.max()
        xs = events['x']
        ys = events['y']
        ps = events['p'].astype(np.int8)
    mask = np.logical_and(np.logical_and(xs >= 0, xs < width), np.logical_and(ys >= 0, ys < height))
    ts = ts[mask]
    xs = xs[mask]
    ys = ys[mask]
    ps = ps[mask]

    ts0 = np.clip(np.floor(ts), a_min=0, a_max=num_bins-1)
    ts1 = np.clip(ts0 + 1, a_min=0, a_max=num_bins-1)
    ts0i, ts1i = np.int32(ts0), np.int32(ts1)

    np.add.at(event, (ys, xs, ps, ts0i), ts1 - ts)
    np.add.at(event, (ys, xs, ps, ts1i), ts - ts0)
    event = event.reshape((height, width, -1))
    event /= event.max()

    # save result
    np.savez(dst_file, event=event)

    # plot
    if plot:
        cv2.namedWindow(f'image', 0)
        cv2.imshow(f'image', image)
        for i in range(num_bins):
            pos_image = (event[:, :, i + num_bins] * 255).astype(np.uint8)
            neg_image = (event[:, :, i] * 255).astype(np.uint8)
            pos_image = cv2.equalizeHist(pos_image)
            neg_image = cv2.equalizeHist(neg_image)
            image = np.concatenate((neg_image[..., None], np.zeros((height, width, 1)), pos_image[..., None]), axis=-1)
            cv2.namedWindow(f'est {i + 1}', 0)
            cv2.imshow(f'est {i + 1}', image)
        cv2.waitKey()


def run(image_file_list, event_file_list, dst_file_list, cfg):

    if cfg.representation == 'Image':
        process = Event2Image
    elif cfg.representation == 'EC-SAE':
        process = Event2ECSAE
    elif cfg.representation == 'VG':
        process = Event2VoxelGrid
    else:
        process = Event2EST
    SingleProcess(process).run(image_file_list, event_file_list, dst_file_list, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--dataset_path', type=str, default='dataset', help='dataset path')
    parser.add_argument('-d', '--dataset_name', type=str, default='FE-Wireframe', help='dataset name')
    parser.add_argument('-r', '--representation', type=str, default='EST', choices=['Image', 'EC-SAE', 'VG', 'EST'], help='representation')
    parser.add_argument('-b', '--dim', type=int, default=10, help='dim')
    parser.add_argument('-s', '--scale', type=float, default=1, help='scale')
    parser.add_argument('--plot', action='store_true', help='plot')
    cfg = parser.parse_args()
    print(cfg)

    # Path
    dataset_path = os.path.join(cfg.dataset_path, cfg.dataset_name)
    json_file = os.path.join(dataset_path, 'train.json')

    image_path = os.path.join(dataset_path, 'images-blur')
    event_raw_path = os.path.join(dataset_path, 'events_raw')
    event_path = os.path.join(dataset_path, f'events-{cfg.representation}-{cfg.dim}')
    os.makedirs(event_path, exist_ok=True)

    # File list
    image_file_list = sorted(glob.glob(os.path.join(image_path, '*.png')))
    event_file_list = sorted(glob.glob(os.path.join(event_raw_path, '*.npz')))
    dst_file_list = [os.path.join(event_path, os.path.basename(event_file)) for event_file in event_file_list]

    run(image_file_list, event_file_list, dst_file_list, cfg)
