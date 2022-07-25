import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

from . import _C


class HAFMencoder(object):
    def __init__(self, cfg):
        self.dis_th = cfg.dis_th
        self.ang_th = cfg.ang_th
        self.n_stc_posl = cfg.n_stc_posl
        self.n_stc_negl = cfg.n_stc_negl

    def __call__(self, annotations):
        targets = []
        metas = []
        for ann in annotations:
            target, meta = self._process_per_image(ann)

            targets.append(target)
            metas.append(meta)

        return default_collate(targets), metas

    def adjacent_matrix(self, n, edges, device):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.bool, device=device)
        if len(edges) > 0:
            mat[edges[:, 0], edges[:, 1]] = 1
            mat[edges[:, 1], edges[:, 0]] = 1
        return mat

    def _process_per_image(self, ann):
        junctions = ann['junc']
        height, width = 128, 128
        device = junctions.device

        jmap = torch.zeros((1, height, width), device=device)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)
        xint, yint = junctions[:, 0].long(), junctions[:, 1].long()
        jmap[0, yint, xint] = 1
        joff[0, yint, xint] = junctions[:, 0] - xint.float() - 0.5
        joff[1, yint, xint] = junctions[:, 1] - yint.float() - 0.5

        edges_positive = ann['edges_positive']
        edges_negative = ann['edges_negative']
        pos_mat = self.adjacent_matrix(junctions.size(0), edges_positive, device)
        neg_mat = self.adjacent_matrix(junctions.size(0), edges_negative, device)
        lines = torch.cat([junctions[edges_positive[:, 0]], junctions[edges_positive[:, 1]]], dim=1)
        if len(edges_negative):
            lines_neg = torch.cat([junctions[edges_negative[:2000, 0]], junctions[edges_negative[:2000, 1]]], dim=1)
        else:
            lines_neg = lines[:0]
        lmap, _, _ = _C.encodels(lines, height, width, height, width, len(lines))

        idx = torch.randperm(len(lines), device=device)[:self.n_stc_posl]
        lpos = lines[idx]
        idx = torch.randperm(len(lines_neg), device=device)[:self.n_stc_negl]
        lneg = lines_neg[idx]

        lpre = torch.cat([lpos, lneg], dim=0)
        swap_ = torch.rand(len(lpre), device=device) > 0.5
        lpre[swap_] = lpre[swap_][:, [2, 3, 0, 1]]
        lpre_label = torch.cat([torch.ones(len(lpos), device=device), torch.zeros(len(lneg), device=device)])

        meta = {
            'junc': junctions,
            'Lpos': pos_mat,
            'Lneg': neg_mat,
            'lpre': lpre,
            'lpre_label': lpre_label,
            'lines': lines,
        }

        dismap = torch.sqrt(lmap[0] ** 2 + lmap[1] ** 2)[None]

        def _normalize(inp):
            mag = torch.sqrt(inp[0] * inp[0] + inp[1] * inp[1])
            return inp / (mag + 1e-6)

        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])

        md_ = md_map.reshape(2, -1).t()
        st_ = st_map.reshape(2, -1).t()
        ed_ = ed_map.reshape(2, -1).t()
        Rt = torch.cat(
            [torch.cat([md_[:, None, None, 0], md_[:, None, None, 1]], dim=2),
             torch.cat([-md_[:, None, None, 1], md_[:, None, None, 0]], dim=2)], dim=1)

        st_ = torch.matmul(Rt, st_[..., None]).squeeze(-1).t()
        ed_ = torch.matmul(Rt, ed_[..., None]).squeeze(-1).t()
        swap_mask = (st_[1] < 0) * (ed_[1] > 0)
        pos_map = st_.clone()
        neg_map = ed_.clone()
        temp = pos_map[:, swap_mask]
        pos_map[:, swap_mask] = neg_map[:, swap_mask]
        neg_map[:, swap_mask] = temp

        pos_map[0] = pos_map[0].clamp(min=1e-9)
        pos_map[1] = pos_map[1].clamp(min=1e-9)
        neg_map[0] = neg_map[0].clamp(min=1e-9)
        neg_map[1] = neg_map[1].clamp(max=-1e-9)
        pos_map = pos_map.view(-1, height, width)
        neg_map = neg_map.view(-1, height, width)

        md_angle = torch.atan2(md_map[1], md_map[0]) / (2 * np.pi) + 0.5
        pos_angle = torch.atan2(pos_map[1], pos_map[0]) / (0.5 * np.pi)
        neg_angle = -torch.atan2(neg_map[1], neg_map[0]) / (0.5 * np.pi)

        mask = ((pos_map[1:] > self.ang_th) * (neg_map[1:] < -self.ang_th) * (dismap <= self.dis_th)).float()
        hafm_ang = torch.stack([md_angle, pos_angle, neg_angle], dim=0)
        hafm_dis = dismap.clamp(max=self.dis_th) / self.dis_th

        target = {
            'jloc': jmap,
            'joff': joff,
            'md': hafm_ang,
            'dis': hafm_dis,
            'mask': mask
        }

        return target, meta