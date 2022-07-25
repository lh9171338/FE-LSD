import torch
import numpy as np
import cv2
from torch.utils.data.dataloader import default_collate


def insert_line(image, lines, color, thickness=1):
    lines = np.round(lines).astype(np.int32)
    for pt1, pt2 in lines:
        cv2.line(image, pt1, pt2, color=color, thickness=thickness)

    return image


class HAFMencoder(object):
    def __init__(self, cfg):
        self.n_stc_posl = cfg.n_stc_posl
        self.n_stc_negl = cfg.n_stc_negl
        self.n_pts = 2

    def __call__(self, annotations):
        targets = []
        metas = []
        for ann in annotations:
            target, meta = self._process_per_image(ann)

            targets.append(target)
            metas.append(meta)

        return default_collate(targets), metas

    def _process_per_image(self, ann):
        junctions = ann['junc']
        height, width = 128, 128
        device = junctions.device

        edges_positive = ann['edges_positive']
        edges_negative = ann['edges_negative']
        lines = torch.stack((junctions[edges_positive[:, 0]], junctions[edges_positive[:, 1]]), dim=1)
        if len(edges_negative):
            lines_neg = torch.stack([junctions[edges_negative[:2000, 0]], junctions[edges_negative[:2000, 1]]], dim=1)
        else:
            lines_neg = lines[:0]
        idx = torch.randperm(len(lines), device=device)[:self.n_stc_posl]
        lpos = lines[idx]
        idx = torch.randperm(len(lines_neg), device=device)[:self.n_stc_negl]
        lneg = lines_neg[idx]

        jloc = torch.zeros((1, height, width), device=device)
        joff = torch.zeros((2, height, width), device=device, dtype=torch.float32)
        cloc = torch.zeros((1, height, width), device=device)
        coff = torch.zeros((2, height, width), device=device, dtype=torch.float32)
        eoff = torch.zeros((2, (self.n_pts // 2) * 2, height, width), device=device)
        xint, yint = junctions[:, 0].long(), junctions[:, 1].long()
        jloc[0, yint, xint] = 1
        joff[0, yint, xint] = junctions[:, 0] - xint.float() - 0.5
        joff[1, yint, xint] = junctions[:, 1] - yint.float() - 0.5

        mask = lines[:, 0, 1] > lines[:, 1, 1]
        lines[mask] = lines[mask][:, [1, 0]]
        centers = lines.mean(dim=1)
        xint, yint = centers[:, 0].long(), centers[:, 1].long()
        cloc[0, yint, xint] = 1
        coff[0, yint, xint] = centers[:, 0] - xint.float() - 0.5
        coff[1, yint, xint] = centers[:, 1] - yint.float() - 0.5
        eoff[0, :, yint, xint] = (lines[:, :, 0] - centers[:, None, 0]).t()
        eoff[1, :, yint, xint] = (lines[:, :, 1] - centers[:, None, 1]).t()
        eoff = eoff.reshape(-1, height, width)

        lmap = np.zeros((1, height, width), dtype=np.float32)
        lmap[0] = insert_line(lmap[0], lines.detach().cpu().numpy(), color=255) / 255.0
        lmap = torch.from_numpy(lmap).float().to(device)

        lpre = torch.cat((lpos, lneg))
        swap_ = torch.rand(len(lpre), device=device) > 0.5
        lpre[swap_] = lpre[swap_][:, [1, 0]]
        lpre_label = torch.cat((torch.ones(len(lpos), device=device), torch.zeros(len(lneg), device=device)))

        target = {'lmap': lmap, 'jloc': jloc, 'joff': joff, 'cloc': cloc, 'coff': coff, 'eoff': eoff}
        meta = {'junc': junctions, 'line': lines, 'lpre': lpre, 'lpre_label': lpre_label}

        return target, meta
