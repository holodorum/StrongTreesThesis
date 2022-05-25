from outputobject.output_object import OutputObject
from utilities.utils_oct import UtilsOCT


class OutputObjectOCT(OutputObject):
    def __init__(self, approach_name, input_file, depth, time_limit, _lambda, input_sample, calibration, mode, warmstart):
        super().__init__(approach_name, input_file, depth,
                         time_limit, _lambda, input_sample, calibration, mode, warmstart)
        self.utils = UtilsOCT()
