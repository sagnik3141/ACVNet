from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .kitti_dataset_multi import KITTIDatasetSPC
from .kitti_dataset_spc import KITTIDatasetSPCSingle

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "kitti_spc_multi": KITTIDatasetSPC,
    "kitti_spc": KITTIDatasetSPCSingle
}
