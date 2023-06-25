from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from copy import deepcopy

@DATASETS.register_module()
class CocoCropDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 pre_pipeline=None,
                 crop_pipeline=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 seg_suffix='.png',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk')):

        super(CocoCropDataset, self).__init__(ann_file,
                 pipeline,
                 classes,
                 data_root,
                 img_prefix,
                 seg_prefix,
                 seg_suffix,
                 proposal_file,
                 test_mode,
                 filter_empty_gt,
                 file_client_args)
    
        self.pretrain_pipeline = Compose(pre_pipeline)
        self.crop_pipeline = Compose(crop_pipeline)
        

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
            
        self.pre_pipeline(results)
        results = self.pretrain_pipeline(results)
        
        results_ori, results_crop = deepcopy(results), deepcopy(results)
        del results
        
        results_ori = self.pipeline(results_ori)
        results_crop = self.crop_pipeline(results_crop)
        return results_ori, results_crop
     
        
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        
        while True:
            data_ori, data_crop = self.prepare_train_img(idx)
            if data_ori is None:
                idx = self._rand_another(idx)
                continue
            
            # Duplicate data
            return data_ori, data_crop
        
        
        
def imp():
    from torchvision.utils import save_image
    crop = data_ori['img'].data[:, 355:785, 139:571]
    save_image(crop.unsqueeze(0), 'img_ori.png')
    save_image(data_crop['img'].data.unsqueeze(0), 'img_crop.png')