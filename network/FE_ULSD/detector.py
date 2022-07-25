import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import comb
from .stacked_hg import build_hg
from .multi_task_head import MultiTaskHead
from .hafm import HAFMencoder


def weighted_mse_loss(logits, target, mask=None):
    loss = F.mse_loss(logits, target, reduction='none')
    if mask is not None:
        w = mask.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean()


def weighted_l1_loss(logits, target, mask=None):
    loss = F.l1_loss(logits, target, reduction='none')
    if mask is not None:
        w = mask.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean()


def weighted_smooth_l1_loss(logits, target, mask=None):
    loss = F.smooth_l1_loss(logits, target, reduction='none')
    if mask is not None:
        loss = loss * mask
    return loss.mean()


def calc_junction(jloc, joff, topK, thresh=0.0):
    H, W = jloc.shape[-2:]
    score = jloc.flatten()
    joff = joff.reshape(2, -1).t()

    num = min(int((score > thresh).sum().item()), topK)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // H, indices % W
    junc = torch.stack((x, y), dim=1) + joff[indices] + 0.5

    return junc, score


def calc_line(cloc, coff, eoff, order, topK, thresh=0.0):
    n_pts = order + 1
    H, W = cloc.shape[-2:]
    score = cloc.flatten()
    coff = coff.reshape(2, -1).t()
    eoff = eoff.reshape(2, n_pts // 2 * 2, -1).permute(2, 1, 0)

    num = min(int((score > thresh).sum().item()), topK)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // H, indices % W

    center = torch.stack((x, y), dim=1) + coff[indices] + 0.5
    line = center[:, None] + eoff[indices]
    if n_pts % 2 == 1:
        line = torch.cat((line[:, :n_pts // 2], center[:, None], line[:, n_pts // 2:]), dim=1)

    return line, score


def non_maximum_suppression(heatmap, kernel_size=3):
    max_heatmap = F.max_pool2d(heatmap, kernel_size, stride=1, padding=kernel_size // 2)
    weights = (heatmap == max_heatmap).float()
    heatmap = weights * heatmap
    return heatmap


class WireframeDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_hg(cfg)
        self.head = MultiTaskHead(inplanes=cfg.num_feats, head_size=cfg.head_size)
        self.hafm_encoder = HAFMencoder(cfg)

        self.order = 1
        self.num_feats = cfg.num_feats
        self.n_dyn_junc = cfg.n_dyn_junc
        self.n_dyn_line = cfg.n_dyn_line
        self.junc_thresh = cfg.junc_thresh
        self.line_thresh = cfg.line_thresh
        self.n_dyn_posl = cfg.n_dyn_posl
        self.n_dyn_negl = cfg.n_dyn_negl
        self.n_pts = cfg.n_pts
        self.dim_loi = cfg.dim_loi
        self.dim_fc = cfg.dim_fc
        self.nms_size = cfg.nms_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_feats, self.dim_loi, 3, padding=1),
            nn.BatchNorm2d(self.dim_loi),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim_loi, self.dim_loi, 3, padding=1)
        )
        self.pool1d = nn.MaxPool1d(4, stride=4)
        self.fc = nn.Sequential(
            nn.Linear(self.dim_loi * (self.n_pts // 4), self.dim_fc),
            nn.BatchNorm1d(self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.BatchNorm1d(self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1),
            nn.Sigmoid()
        )

        lambda1 = torch.stack((torch.linspace(1, 0, self.order + 1), torch.linspace(0, 1, self.order + 1)), dim=1)
        self.register_buffer('lambda1', lambda1)

        p = comb(self.order, np.arange(self.order + 1))
        k = np.arange(0, self.order + 1)
        t0 = np.linspace(0, 1, self.order + 1)[:, None]
        coeff0 = p * (t0 ** k) * ((1 - t0) ** (self.order - k))
        t = np.linspace(0, 1, self.n_pts)[:, None]
        coeff = p * (t ** k) * ((1 - t) ** (self.order - k))
        lambda2 = np.matmul(coeff, np.linalg.inv(coeff0))
        lambda2 = torch.from_numpy(lambda2).float()
        self.register_buffer('lambda2', lambda2)

        self.loss = nn.BCELoss(reduction='none')

    def pooling(self, feature, loi_pred):
        C, H, W = feature.shape

        pts = (self.lambda2[None, :, :, None] * loi_pred[:, None]).sum(dim=2) - 0.5
        pts = pts.reshape(-1, 2)
        px, py = pts[:, 0].contiguous(), pts[:, 1].contiguous()
        px0 = px.floor().clamp(min=0, max=W - 1)
        py0 = py.floor().clamp(min=0, max=H - 1)
        px1 = (px0 + 1).clamp(min=0, max=W - 1)
        py1 = (py0 + 1).clamp(min=0, max=H - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        loi_feature = (feature[:, py0l, px0l] * (py1 - py) * (px1 - px) +
                       feature[:, py1l, px0l] * (py - py0) * (px1 - px) +
                       feature[:, py0l, px1l] * (py1 - py) * (px - px0) +
                       feature[:, py1l, px1l] * (py - py0) * (px - px0)).reshape(C, -1, self.n_pts).permute(1, 0, 2)

        loi_feature = self.pool1d(loi_feature)
        loi_feature = loi_feature.flatten(1)
        loi_score = self.fc(loi_feature).squeeze(dim=-1)
        return loi_score

    def forward(self, images, annotations):
        if self.training:
            return self.forward_train(images, annotations)
        else:
            return self.forward_test(images, annotations)

    @torch.no_grad()
    def forward_test(self, images, annotations):
        features = self.backbone(images)
        outputs = self.head(features)
        loi_features = self.conv(features)

        jloc_preds = outputs[:, 1:3].softmax(1)[:, 1:]
        joff_preds = outputs[:, 3:5]
        cloc_preds = outputs[:, 5:7].softmax(1)[:, 1:]
        coff_preds = outputs[:, 7:9]
        eoff_preds = outputs[:, 9:].tanh() * 128.0

        results = []
        B, C, H, W = outputs.shape
        for i in range(B):
            jloc_pred = jloc_preds[i]
            joff_pred = joff_preds[i]
            cloc_pred = cloc_preds[i]
            coff_pred = coff_preds[i]
            eoff_pred = eoff_preds[i]
            loi_feature = loi_features[i]
            annotation = annotations[i]

            # Generate junctions and lines
            junc_pred, junc_score = calc_junction(non_maximum_suppression(jloc_pred, self.nms_size), joff_pred,
                                                  topK=self.n_dyn_junc, thresh=self.junc_thresh)
            loi_pred, loi_score = calc_line(cloc_pred, coff_pred, eoff_pred, self.order,
                                            topK=self.n_dyn_line, thresh=self.line_thresh)

            try:
                # Match junctions and lines
                dist_junc_to_end1, idx_junc_to_end1 = ((loi_pred[:, None, 0] - junc_pred[None]) ** 2).sum(dim=-1).min(
                    dim=-1)
                dist_junc_to_end2, idx_junc_to_end2 = ((loi_pred[:, None, -1] - junc_pred[None]) ** 2).sum(dim=-1).min(
                    dim=-1)

                idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
                idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
                iskeep = idx_junc_to_end_min != idx_junc_to_end_max

                loi_pred = loi_pred[iskeep]
                loi_score = loi_score[iskeep]
                idx_junc = torch.stack((idx_junc_to_end_min[iskeep], idx_junc_to_end_max[iskeep]), dim=1)
                dist_junc = dist_junc_to_end1[iskeep] + dist_junc_to_end2[iskeep]
                cost = dist_junc - torch.log(loi_score)

                if self.order == 1:
                    idx_junc, unique_indices = np.unique(idx_junc.detach().cpu().numpy(), return_index=True, axis=0)
                    loi_pred = torch.stack((junc_pred[idx_junc[:, 0]], junc_pred[idx_junc[:, 1]]), dim=1)
                    mask = loi_pred[:, 0, 1] > loi_pred[:, 1, 1]
                    loi_pred[mask] = loi_pred[mask][:, [1, 0]]

                else:
                    indeces = cost.argsort()
                    loi_pred = loi_pred[indeces]
                    idx_junc = idx_junc[indeces]

                    idx_junc, unique_indices = np.unique(idx_junc.detach().cpu().numpy(), return_index=True, axis=0)
                    loi_pred = loi_pred[unique_indices]

                    end_pred = torch.stack((junc_pred[idx_junc[:, 0]], junc_pred[idx_junc[:, 1]]), dim=1)
                    mask = end_pred[:, 0, 1] > end_pred[:, 1, 1]
                    end_pred[mask] = end_pred[mask][:, [1, 0]]
                    delta_end_pred = end_pred - loi_pred[:, [0, -1]]
                    loi_pred += (self.lambda1[None, :, :, None] * delta_end_pred[:, None]).sum(dim=2)

                line_score = self.pooling(loi_feature, loi_pred)

                sx = annotation['width'] / W
                sy = annotation['height'] / H

                loi_pred[:, :, 0] *= sx
                loi_pred[:, :, 1] *= sy
            except:
                print(annotation['filename'])
                loi_pred = []
                line_score = []

            result = {
                'line_pred': loi_pred,
                'line_score': line_score,
                'jloc_pred': jloc_pred,
                'joff_pred': joff_pred,
                'filename': annotation['filename'],
                'width': annotation['width'],
                'height': annotation['height']
            }
            results.append(result)

        return results

    def forward_train(self, images, annotations):
        batch_size = images.shape[0]
        features = self.backbone(images)
        targets, metas = self.hafm_encoder(annotations)
        outputs = self.head(features)
        loi_features = self.conv(features.detach())

        loss_dict = {
            'loss_lmap': 0.0,
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_cloc': 0.0,
            'loss_coff': 0.0,
            'loss_eoff': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
        }

        n_eoff = targets['eoff'].shape[1] // 2
        loss_dict['loss_lmap'] = F.binary_cross_entropy(outputs[:, 0:1].sigmoid(), targets['lmap'])
        loss_dict['loss_jloc'] = F.binary_cross_entropy(outputs[:, 1:3].softmax(dim=1)[:, 1:], targets['jloc'])
        loss_dict['loss_joff'] = weighted_mse_loss(outputs[:, 3:5], targets['joff'], targets['jloc'])
        loss_dict['loss_cloc'] = F.binary_cross_entropy(outputs[:, 5:7].softmax(dim=1)[:, 1:], targets['cloc'])
        loss_dict['loss_coff'] = weighted_mse_loss(outputs[:, 7:9], targets['coff'], targets['cloc'])
        loss_dict['loss_eoff'] = n_eoff * weighted_smooth_l1_loss(outputs[:, 9:].tanh() * 128.0, targets['eoff'], targets['cloc'])

        jloc_preds = outputs[:, 1:3].softmax(1)[:, 1:]
        joff_preds = outputs[:, 3:5]
        cloc_preds = outputs[:, 5:7].softmax(1)[:, 1:]
        coff_preds = outputs[:, 7:9]
        eoff_preds = outputs[:, 9:].tanh() * 128.0

        for i in range(batch_size):
            jloc_pred = jloc_preds[i]
            joff_pred = joff_preds[i]
            cloc_pred = cloc_preds[i]
            coff_pred = coff_preds[i]
            eoff_pred = eoff_preds[i]
            loi_feature = loi_features[i]
            meta = metas[i]
            lpre = meta['lpre']
            lpre_label = meta['lpre_label']
            line = meta['line']

            with torch.no_grad():
                # Generate junctions and lines
                junc_pred, junc_score = calc_junction(non_maximum_suppression(jloc_pred, self.nms_size), joff_pred,
                                                      topK=self.n_dyn_junc)
                loi_pred, loi_score = calc_line(cloc_pred, coff_pred, eoff_pred, self.order, topK=self.n_dyn_line)

                # Match junctions and lines
                dist_junc_to_end1, idx_junc_to_end1 = ((loi_pred[:, None, 0] - junc_pred[None]) ** 2).sum(dim=-1).min(
                    dim=-1)
                dist_junc_to_end2, idx_junc_to_end2 = ((loi_pred[:, None, -1] - junc_pred[None]) ** 2).sum(dim=-1).min(
                    dim=-1)

                idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
                idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
                iskeep = idx_junc_to_end_min != idx_junc_to_end_max

                loi_pred = loi_pred[iskeep]
                loi_score = loi_score[iskeep]
                idx_junc = torch.stack((idx_junc_to_end_min[iskeep], idx_junc_to_end_max[iskeep]), dim=1)
                dist_junc = dist_junc_to_end1[iskeep] + dist_junc_to_end2[iskeep]
                cost = dist_junc - torch.log(loi_score)

                if self.order == 1:
                    idx_junc, unique_indices = np.unique(idx_junc.detach().cpu().numpy(), return_index=True, axis=0)
                    loi_score = loi_score[unique_indices]

                    loi_pred = torch.stack((junc_pred[idx_junc[:, 0]], junc_pred[idx_junc[:, 1]]), dim=1)
                    mask = loi_pred[:, 0, 1] > loi_pred[:, 1, 1]
                    loi_pred[mask] = loi_pred[mask][:, [1, 0]]

                else:
                    indeces = cost.argsort()
                    loi_pred = loi_pred[indeces]
                    loi_score = loi_score[indeces]
                    idx_junc = idx_junc[indeces]

                    idx_junc, unique_indices = np.unique(idx_junc.detach().cpu().numpy(), return_index=True, axis=0)
                    loi_pred = loi_pred[unique_indices]
                    loi_score = loi_score[unique_indices]

                    end_pred = torch.stack((junc_pred[idx_junc[:, 0]], junc_pred[idx_junc[:, 1]]), dim=1)
                    mask = end_pred[:, 0, 1] > end_pred[:, 1, 1]
                    end_pred[mask] = end_pred[mask][:, [1, 0]]
                    delta_end_pred = end_pred - loi_pred[:, [0, -1]]
                    loi_pred += (self.lambda1[None, :, :, None] * delta_end_pred[:, None]).sum(dim=2)

                loi_pred_mirror = loi_pred[:, range(loi_pred.shape[1])[::-1]]
                dists1 = ((loi_pred[:, None] - line) ** 2).sum(dim=-1).mean(dim=-1)
                dists2 = ((loi_pred_mirror[:, None] - line) ** 2).sum(dim=-1).mean(dim=-1)
                # costs = torch.min(dists1, dists2) - torch.log(loi_score)[:, None]
                # _, indices = costs.min(dim=0)
                costs = torch.min(dists1, dists2)
                costs, indices = costs.min(dim=0)
                mask = costs < 1.5 * 1.5
                indices = indices[mask]
                label = torch.zeros_like(loi_score)
                label[indices] = 1.0
                pos_id = (label == 1).nonzero(as_tuple=False).flatten()
                neg_id = (label == 0).nonzero(as_tuple=False).flatten()

                if len(pos_id) > self.n_dyn_posl:
                    idx = torch.randperm(pos_id.shape[0], device=pos_id.device)[:self.n_dyn_posl]
                    pos_id = pos_id[idx]

                if len(neg_id) > self.n_dyn_negl:
                    idx = torch.randperm(neg_id.shape[0], device=neg_id.device)[:self.n_dyn_negl]
                    neg_id = neg_id[idx]

                keep_id = torch.cat((pos_id, neg_id))
                loi_pred = loi_pred[keep_id]
                loi_label = torch.cat([torch.ones(len(pos_id), dtype=loi_pred.dtype, device=loi_pred.device),
                                       torch.zeros(len(neg_id), dtype=loi_pred.dtype, device=loi_pred.device)])
                loi_pred = torch.cat((loi_pred, lpre))
                loi_label = torch.cat((loi_label, lpre_label))

            loi_score = self.pooling(loi_feature, loi_pred)
            loss = self.loss(loi_score, loi_label)
            loss_positive = loss[loi_label == 1].mean()
            loss_negative = loss[loi_label == 0].mean()

            loss_dict['loss_pos'] += loss_positive / batch_size
            loss_dict['loss_neg'] += loss_negative / batch_size

        return loss_dict, loi_label, loi_score
