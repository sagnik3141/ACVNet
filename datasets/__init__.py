from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .kitti_dataset_multi import KITTIDatasetSPC

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "kitti_spc_multi": KITTIDatasetSPC
}
