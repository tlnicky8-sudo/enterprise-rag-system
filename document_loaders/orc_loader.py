"""
获得一个ORC实例，同时根据环境参数自动选择最佳计算资源
    1. 决定使用哪个 OCR 引擎
        优先尝试 rapidocr_paddle（底层基于 PaddlePaddle，支持 GPU 和 CPU）。
        如果没装 PaddlePaddle，则回退到 rapidocr_onnxruntime（基于 ONNX Runtime，主要用于 CPU）。
    2. 在能用 PaddlePaddle 时，决定是否开启 GPU 加速
        use_cuda=True → 模型全部使用 GPU。
        use_cuda=False → 模型都使用 CPU（这在 GPU 不可用或没装 CUDA 时使用）。
"""



# edu_document_loaders/edu_ocr.py 源码
from typing import TYPE_CHECKING
'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。它在 CPU 上进行了优化，资源占用较低.
'''

def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        '''
        det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
        '''
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
    except ImportError:
        #
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
