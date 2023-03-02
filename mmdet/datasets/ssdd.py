from .xml_style import XMLDataset


class SSDDDataset(XMLDataset):

    # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')

    CLASSES = ('ship', )

    # CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9',)

    def __init__(self, **kwargs):
        super(SSDDDataset, self).__init__(**kwargs)
    #     if 'VOC2007' in self.img_prefix:
        self.year = 2012
    #     elif 'VOC2012' in self.img_prefix:
    #         self.year = 2012
    #     else:
    #         raise ValueError('Cannot infer dataset year from img_prefix')
