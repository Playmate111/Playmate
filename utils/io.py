import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj
