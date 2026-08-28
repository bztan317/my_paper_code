from ...core import register
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator


@register()
class CocoEvaluator(FasterCocoEvaluator):
    pass
