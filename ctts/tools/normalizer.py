from typing import Callable
from functools import partial
from typing import Callable

def normalizer_en_nemo_text() -> Callable[[str], str]:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    return partial(
        Normalizer(input_case="cased", lang="en").normalize,
        verbose=False,
        punct_post_process=True,
    )

def normalizer_zh_tn() -> Callable[[str], str]:
    from tn.chinese.normalizer import Normalizer

    return Normalizer().normalize
