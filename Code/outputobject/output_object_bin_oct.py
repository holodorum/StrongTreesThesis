from outputobject.output_object import OutputObject
from utilities.utils_bin_oct import UtilsBinOCT


class OutputObjectbinOCT(OutputObject):
    def __init__(self, approach_name, input_file, depth, time_limit, _lambda, input_sample, calibration, mode, warmstart):
        super().__init__(approach_name, input_file, depth,
                         time_limit, _lambda, input_sample, calibration, mode, warmstart)
        self.utils = UtilsBinOCT()

    def model_parameters(self, grb_model):
        '''
        Gives the model get beta, b and p
        '''
        self.b_value = grb_model.model.getAttr("X", grb_model.f)
        self.beta_value = grb_model.model.getAttr("X", grb_model.p)
        self.p_value = 1  # We always split in binary tree
