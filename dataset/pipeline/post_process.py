import numpy as np
import cv2


class PostProcesser:
    def __init__(self, threshold, binarization=True):
        self._threshold = threshold
        self.do_binarization = binarization

    def __call__(self, results, logit):
        logit = self._to_numpy(logit)
        logit = self._depadding(results, logit)
        logit = self._to_origin_shape(results, logit)
        if self.do_binarization:
            logit = self.binarization(logit)
        return logit

    def _to_numpy(self, logit):
        logit = logit.sigmoid()
        logit = logit.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        return logit

    def _depadding(self, results, logit):
        pad_t, pad_b, pad_l, pad_r = results['pad_t'], results['pad_b'], results['pad_l'], results['pad_r']
        height, width = logit.shape[:2]
        logit = logit[pad_t: height - pad_b, pad_l: width - pad_r, :]
        return logit

    def _to_origin_shape(self, results, logit):
        logit = cv2.resize(logit, dsize=results['original_shape'], interpolation=cv2.INTER_LINEAR)
        return logit

    def binarization(self, logit, dtype=np.float):
        logit = (self._threshold < logit).astype(dtype)
        return logit
