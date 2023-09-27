#from .coco import CocoCropDataset
#from .transforms import DistillCrop
#from .roi_heads import ContRoIHead
from .faster_rcnn_rvkd import FasterRCNN_reviewKD
from .faster_rcnn_rvkd_scale import FasterRCNN_reviewKD_SCALE
from .faster_rcnn_dkd import FasterRCNN_DKDreviewkd
from .faster_rcnn_dkd_scale import FasterRCNN_DKDreviewkd_SCALE
from .faster_rcnn_mgd import FasterRCNN_MGD
from .faster_rcnn_mgd_scale import FasterRCNN_MGD_SCALE
from .faster_rcnn_pkd import FasterRCNN_PKD
from .faster_rcnn_pkd_scale import FasterRCNN_PKD_SCALE



__all__ = [ 'FasterRCNN_KD', 'FasterRCNN_reviewKD','FasterRCNN_reviewKD_SCALE', 'FasterRCNN_DKDreviewkd', 'FasterRCNN_DKDreviewkd_SCALE', 'FasterRCNN_MGD'
           ,'FasterRCNN_MGD_SCALE', 'FasterRCNN_PKD', 'FasterRCNN_PKD_SCALE']