from makiflow.base import MakiTensor
from makiflow.layers import ConvLayer
from makiflow.layers import ReshapeLayer


class DCParams:
    TYPE = 'DetectorClassifier'
    NAME = 'NAME'
    CLASS_NUMBER = 'CLASS_NUMBER'
    DBOXES = 'DBOXES'

    REG_X_NAME = 'REG_X_NAME'
    RKW = 'RKW'
    RKH = 'RKH'
    RIN_F = 'RIN_F'
    USE_REG_BIAS = 'USE_REG_BIAS'
    REG_INIT_TYPE = 'REG_INIT_TYPE'

    CLASS_X_NAME = 'CLASS_X_NAME'
    CKW = 'CKW'
    CKH = 'CKH'
    CIN_F = 'CIN_F'
    USE_CLASS_BIAS = 'USE_CLASS_BIAS'
    CLASS_INIT_TYPE = 'CLASS_INIT_TYPE'


class DetectorClassifier:
    """
    This class represents a part of SSD algorithm. It consists of several parts:
    conv layers -> detector -> confidences + localization regression.
    """

    def __init__(
            self,
            reg_fms: MakiTensor, rkw, rkh, rin_f,
            class_fms: MakiTensor, ckw, ckh, cin_f,
            num_classes, dboxes: list, name,
            use_reg_bias=True, use_class_bias=True,
            reg_init_type='he', class_init_type='he'
    ):
        """
        Parameters
        ----------
        reg_fms : MakiTensor
            Source of features for the bbox regressor.
        rkw : int
            Width of the regressor's kernel.
        rkh : int
            Height of the regressor's kernel.
        rin_f : int
            Number of the input feature maps for the regressor.
        class_fms : MakiTensor
            Source of features for the classificator.
        ckw : int
            Width of the classificator's kernel.
        ckh : int
            Height of the classificator's kernel.
        cin_f : int
            Number of the input feature maps for the classificator.
        num_classes : int
            Number of classes to be classified + 'no object' class. 'no object' class is used for
            background.
        dboxes : list
            List of tuples of the default boxes' characteristics (width and height). Each characteristic
            must be represented in relative coordinates. Boxes' coordinates are always in the center of
            the cell of the feature map. Example:
            [(1, 1), (0.5, 1.44), (1.44, 0.5)] - (1,1) - center box matches one cell of the feature map.
            We recommend to use the following configuration:
            [
                (1, 1), (1, 2), (2, 1),
                (1.26, 1.26), (1.26, 2.52), (2.52, 1.26),
                (1.59, 1.59), (1.59, 3.17), (3.17, 1.59)
            ]
            This configuration corresponds to bboxes in RetinaNet.
        name : str or int
            Will be used as conjugation for the names of the classificator and regressor.
        """
        self.reg_x = reg_fms
        self.rkw = rkw
        self.rkh = rkh
        self.rin_f = rin_f
        self.class_x = class_fms
        self.ckw = ckw
        self.ckh = ckh
        self.cin_f = cin_f
        self.class_number = num_classes
        self._dboxes = dboxes
        self.name = str(name)
        self.use_class_bias = use_class_bias
        self.use_reg_bias = use_reg_bias
        self.reg_init_type = reg_init_type
        self.class_init_type = class_init_type

        classifier_out_f = num_classes * len(dboxes)
        bb_regressor_out_f = 4 * len(dboxes)
        self.classifier = ConvLayer(
            ckw, ckh, cin_f, classifier_out_f, use_bias=use_class_bias,
            activation=None, padding='SAME', kernel_initializer=class_init_type,
            name='SSDClassifier_' + str(name)
        )
        self.bb_regressor = ConvLayer(
            rkw, rkh, rin_f, bb_regressor_out_f, use_bias=use_reg_bias,
            activation=None, padding='SAME', kernel_initializer=reg_init_type,
            name='SSDBBDetector_' + str(name)
        )
        self._make_detections()

    def _make_detections(self):
        """
        Creates list with "flattened" predicted confidences and regressed localization offsets for each dbox.
        Example: [confidences, offsets]
        """
        n_dboxes = len(self._dboxes)

        # FLATTEN PREDICTIONS OF THE CLASSIFIER
        confidences = self.classifier(self.class_x)
        # [BATCH SIZE, WIDTH, HEIGHT, DEPTH]
        conf_shape = confidences.get_shape()
        # width of the tensor
        conf_w = conf_shape[1]
        # height of the tensor
        conf_h = conf_shape[2]
        conf_reshape = ReshapeLayer([-1, conf_w * conf_h * n_dboxes, self.class_number], 'ConfReshape_' + self.name)
        self._confidences = conf_reshape(confidences)

        # FLATTEN PREDICTIONS OF THE REGRESSOR
        offsets = self.bb_regressor(self.reg_x)
        # [BATCH SIZE, WIDTH, HEIGHT, DEPTH]
        off_shape = offsets.get_shape()
        # width of the tensor
        off_w = off_shape[1]
        # height of the tensor
        off_h = off_shape[2]
        # 4 is for four offsets: [x1, y1, x2, y2]
        off_reshape = ReshapeLayer([-1, off_w * off_h * n_dboxes, 4], name='OffReshape_' + self.name)
        self._offsets = off_reshape(offsets)

    def get_dboxes(self):
        return self._dboxes

    def get_conf_offsets(self):
        return [self._confidences, self._offsets]

    def get_feature_map_shape(self):
        """
        It is used for creating default boxes since their size depends
        on the feature maps' widths and heights.
        """
        return self.reg_x.get_shape()

    def to_dict(self):
        return {
            'type': DCParams.TYPE,
            'params': {
                DCParams.REG_X_NAME: self.reg_x.get_name(),
                DCParams.RKW: self.rkw,
                DCParams.RKH: self.rkh,
                DCParams.RIN_F: self.rin_f,
                DCParams.USE_REG_BIAS: self.use_reg_bias,
                DCParams.REG_INIT_TYPE: self.reg_init_type,

                DCParams.CLASS_X_NAME: self.class_x.get_name(),
                DCParams.CKW: self.ckw,
                DCParams.CKH: self.ckh,
                DCParams.CIN_F: self.cin_f,
                DCParams.USE_CLASS_BIAS: self.use_class_bias,
                DCParams.CLASS_INIT_TYPE: self.class_init_type,

                DCParams.CLASS_NUMBER: self.class_number,
                DCParams.DBOXES: self._dboxes,
                DCParams.NAME: self.name
            }
        }
