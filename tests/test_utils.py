from ..src.utils import preprocess


def test_preprocess():
    data = [1, 2, None, 4]
    cleaned = preprocess(data)
    assert None not in cleaned
