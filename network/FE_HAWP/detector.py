import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .stacked_hg import build_hg
from .hafm import HAFMencoder


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)
    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()


def sigmoid_l1_loss(logits, targets, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        w = mask.mean(3, keepdim=True).mean(2, keepdim=True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean()


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


def get_junctions(jloc, joff, topk=300, th=0.0):
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index // 128).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % 128).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores > th]


class WireframeDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_dyn_junc = cfg.n_dyn_junc
        self.junc_thresh = cfg.junc_thresh
        self.n_dyn_posl = cfg.n_dyn_posl
        self.n_dyn_negl = cfg.n_dyn_negl
        self.n_pts0 = cfg.n_pts0
        self.n_pts1 = cfg.n_pts1
        self.dim_feat = cfg.num_feats
        self.dim_loi = cfg.dim_loi
        self.dim_fc = cfg.dim_fc
        self.scale = cfg.dis_th
        self.use_residual = cfg.use_residual

        self.hafm_encoder = HAFMencoder(cfg)
        self.backbone = build_hg(cfg)

        # self.
        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :])
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

        self.conv = nn.Conv2d(self.dim_feat, self.dim_loi, 1)
        self.pool1d = nn.MaxPool1d(self.n_pts0 // self.n_pts1, stride=self.n_pts0 // self.n_pts1)
        self.fc = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1)
        )

    def pooling(self, loi_feature, lines):
        C, H, W = loi_feature.shape
        start_points, end_points = lines[:, :2], lines[:, 2:]

        sampled_points = start_points[:, :, None] * self.tspan + end_points[:, :, None] * (1 - self.tspan) - 0.5
        sampled_points = sampled_points.transpose(1, 2).reshape(-1, 2)
        px, py = sampled_points[:, 0], sampled_points[:, 1]
        px0 = px.floor().clamp(min=0, max=W - 1)
        py0 = py.floor().clamp(min=0, max=H - 1)
        px1 = (px0 + 1).clamp(min=0, max=W - 1)
        py1 = (py0 + 1).clamp(min=0, max=H - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((loi_feature[:, py0l, px0l] * (py1 - py) * (px1 - px) +
               loi_feature[:, py1l, px0l] * (py - py0) * (px1 - px) +
               loi_feature[:, py0l, px1l] * (py1 - py) * (px - px0) +
               loi_feature[:, py1l, px1l] * (py - py0) * (px - px0)).view(self.dim_loi, -1, self.n_pts0)
              ).transpose(0, 1).contiguous()

        xp = self.pool1d(xp)
        xp = xp.view(-1, self.n_pts1 * self.dim_loi)
        logits = self.fc(xp).flatten()
        return logits

    def forward(self, images, annotations):
        if self.training:
            return self.forward_train(images, annotations)
        else:
            return self.forward_test(images, annotations)

    @torch.no_grad()
    def forward_test(self, images, annotations):
        outputs, features = self.backbone(images)
        loi_features = self.conv(features)

        output = outputs[0]
        md_preds = output[:, :3].sigmoid()
        dis_preds = output[:, 3:4].sigmoid()
        res_preds = output[:, 4:5].sigmoid()
        jloc_preds = output[:, 5:7].softmax(1)[:, 1:]
        joff_preds = output[:, 7:9].sigmoid() - 0.5
        B, C, H, W = output.shape

        results = []
        for i in range(B):
            md_pred = md_preds[i]
            dis_pred = dis_preds[i]
            res_pred = res_preds[i]
            jloc_pred = jloc_preds[i]
            joff_pred = joff_preds[i]
            loi_feature = loi_features[i]
            annotation = annotations[i]

            if self.use_residual:
                line_preds = self.proposal_lines(md_pred, dis_pred, res_pred, self.scale).view(-1, 4)
            else:
                line_preds = self.proposal_lines(md_pred, dis_pred, None, self.scale).view(-1, 4)

            jloc_pred_nms = non_maximum_suppression(jloc_pred)
            topK = min(self.n_dyn_junc, int((jloc_pred_nms > self.junc_thresh).float().sum().item()))
            junc_preds = get_junctions(jloc_pred_nms, joff_pred, topk=topK)

            try:
                dis_junc_to_end1, idx_junc_to_end1 = torch.sum((line_preds[:, :2] - junc_preds[:, None]) ** 2, dim=-1).min(0)
                dis_junc_to_end2, idx_junc_to_end2 = torch.sum((line_preds[:, 2:] - junc_preds[:, None]) ** 2, dim=-1).min(0)
                idx_junc_to_end_min = torch.min(idx_junc_to_end1,idx_junc_to_end2)
                idx_junc_to_end_max = torch.max(idx_junc_to_end1,idx_junc_to_end2)
                iskeep = idx_junc_to_end_min < idx_junc_to_end_max
                idx_lines_for_junctions = torch.stack([idx_junc_to_end_min[iskeep], idx_junc_to_end_max[iskeep]], dim=1).unique(dim=0)
                lines = torch.cat([junc_preds[idx_lines_for_junctions[:, 0]], junc_preds[idx_lines_for_junctions[:, 1]]], dim=1)

                scores = self.pooling(loi_feature, lines).sigmoid()
                iskeep = scores > 0.05
                lines = lines[iskeep]
                scores = scores[iskeep]

                sx = annotation['width'] / W
                sy = annotation['height'] / H
                lines = lines.view(-1, 2, 2)

                lines[:, :, 0] *= sx
                lines[:, :, 1] *= sy
            except:
                print(annotation['filename'])
                lines = []
                scores = []

            result = {
                'line_pred': lines,
                'line_score': scores,
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
        device = images.device

        targets, metas = self.hafm_encoder(annotations)
        outputs, features = self.backbone(images)

        loss_dict = {
            'loss_md': 0.0,
            'loss_dis': 0.0,
            'loss_res': 0.0,
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_pos': 0.0,
            'loss_neg': 0.0,
        }
        mask = targets['mask']
        for output in outputs:
            loss_map = torch.mean(F.l1_loss(output[:, :3].sigmoid(), targets['md'], reduction='none'), dim=1, keepdim=True)
            loss_dict['loss_md'] += torch.mean(loss_map * mask) / torch.mean(mask)
            loss_map = F.l1_loss(output[:, 3:4].sigmoid(), targets['dis'], reduction='none')
            loss_dict['loss_dis'] += torch.mean(loss_map * mask) / torch.mean(mask)
            loss_residual_map = F.l1_loss(output[:, 4:5].sigmoid(), loss_map, reduction='none')
            loss_dict['loss_res'] += torch.mean(loss_residual_map * mask) / torch.mean(mask)
            loss_dict['loss_jloc'] += cross_entropy_loss_for_junction(output[:, 5:7], targets['jloc'])
            loss_dict['loss_joff'] += sigmoid_l1_loss(output[:, 7:9], targets['joff'], -0.5, targets['jloc'])

        loi_features = self.conv(features)
        output = outputs[0]
        md_preds = output[:, :3].sigmoid()
        dis_preds = output[:, 3:4].sigmoid()
        res_preds = output[:, 4:5].sigmoid()
        jloc_preds = output[:, 5:7].softmax(1)[:, 1:]
        joff_preds = output[:, 7:9].sigmoid() - 0.5

        for i in range(batch_size):
            md_pred = md_preds[i]
            dis_pred = dis_preds[i]
            res_pred = res_preds[i]
            jloc_pred = jloc_preds[i]
            joff_pred = joff_preds[i]
            loi_feature = loi_features[i]
            meta = metas[i]

            if self.use_residual:
                line_preds = self.proposal_lines(md_pred, dis_pred, res_pred, self.scale).view(-1, 4)
            else:
                line_preds = self.proposal_lines(md_pred, dis_pred, None, self.scale).view(-1, 4)
            junction_gt = meta['junc']
            N = junction_gt.shape[0]

            junc_preds = get_junctions(non_maximum_suppression(jloc_pred), joff_pred, topk=min(N * 2 + 2, self.n_dyn_junc))
            dis_junc_to_end1, idx_junc_to_end1 = torch.sum((line_preds[:, :2] - junc_preds[:, None]) ** 2, dim=-1).min(0)
            dis_junc_to_end2, idx_junc_to_end2 = torch.sum((line_preds[:, 2:] - junc_preds[:, None]) ** 2, dim=-1).min(0)

            idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
            idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)
            iskeep = idx_junc_to_end_min < idx_junc_to_end_max
            idx_lines_for_junctions = torch.cat([idx_junc_to_end_min[iskeep, None], idx_junc_to_end_max[iskeep, None]], dim=1).unique(dim=0)
            idx_lines_for_junctions_mirror = torch.cat([idx_lines_for_junctions[:, 1:], idx_lines_for_junctions[:, :1]], dim=1)
            idx_lines_for_junctions = torch.cat([idx_lines_for_junctions, idx_lines_for_junctions_mirror])
            lines = torch.cat([junc_preds[idx_lines_for_junctions[:, 0]], junc_preds[idx_lines_for_junctions[:, 1]]], dim=1)

            cost_, match_ = torch.sum((junc_preds - junction_gt[:, None]) ** 2, dim=-1).min(0)
            match_[cost_ > 1.5 * 1.5] = N
            Lpos = meta['Lpos']
            Lneg = meta['Lneg']
            labels = Lpos[match_[idx_lines_for_junctions[:, 0]], match_[idx_lines_for_junctions[:, 1]]]

            iskeep = torch.zeros_like(labels, dtype=torch.bool)
            cdx = labels.nonzero(as_tuple=False).flatten()
            if len(cdx) > self.n_dyn_posl:
                perm = torch.randperm(len(cdx), device=device)[:self.n_dyn_posl]
                cdx = cdx[perm]
            iskeep[cdx] = 1

            cdx = (labels == 0).nonzero(as_tuple=False).flatten()
            if len(cdx) > self.n_dyn_negl:
                perm = torch.randperm(len(cdx), device=device)[:self.n_dyn_negl]
                cdx = cdx[perm]
            iskeep[cdx] = 1

            lines = lines[iskeep]
            labels = labels[iskeep]

            lines = torch.cat([lines, meta['lpre']])
            labels = torch.cat([labels.float(), meta['lpre_label']])

            scores = self.pooling(loi_feature, lines)
            loss_ = self.loss(scores, labels)

            loss_positive = loss_[labels == 1].mean()
            loss_negative = loss_[labels == 0].mean()

            loss_dict['loss_pos'] += loss_positive / batch_size
            loss_dict['loss_neg'] += loss_negative / batch_size

        return loss_dict, labels, scores

    def proposal_lines(self, md_maps, dis_maps, residual_maps, scale):
        """

        :param md_maps: 3 x H x W, the range should be (0,1) for every element
        :param dis_maps: 1 x H x W
        :return:
        """
        device = md_maps.device
        sign_pad = torch.tensor([-1, 0, 1], device=device, dtype=torch.float32).reshape(3, 1, 1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1, 1, 1))
        else:
            dis_maps_new = dis_maps.repeat((3, 1, 1)) + sign_pad * residual_maps.repeat((3, 1, 1))
        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()

        y0, x0 = torch.meshgrid(_y, _x)
        md_ = (md_maps[0] - 0.5) * np.pi * 2
        st_ = md_maps[1] * np.pi / 2
        ed_ = -md_maps[2] * np.pi / 2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)
        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st / cs_st
        y_ed = ss_ed / cs_ed

        x_st_rotated = (cs_md - ss_md * y_st)[None] * dis_maps_new * scale
        y_st_rotated = (ss_md + cs_md * y_st)[None] * dis_maps_new * scale
        x_ed_rotated = (cs_md - ss_md * y_ed)[None] * dis_maps_new * scale
        y_ed_rotated = (ss_md + cs_md * y_ed)[None] * dis_maps_new * scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0, max=width-1)
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0, max=height-1)
        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0, max=width-1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0, max=height-1)

        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final)).permute((1, 2, 3, 0))

        return lines