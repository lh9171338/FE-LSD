from network.FE_HAWP.detector import WireframeDetector as FE_HAWP_Detector
from network.FE_ULSD.detector import WireframeDetector as FE_ULSD_Detector


def build_model(cfg):
    names = ['FE-HAWP', 'FE-ULSD']
    arch = cfg.arch
    assert arch in names, 'Unrecognized arch name'

    if arch == 'FE-HAWP':
        model = FE_HAWP_Detector(cfg)
    elif arch == 'FE-ULSD':
        model = FE_ULSD_Detector(cfg)
    else:
        model = None

    return model
