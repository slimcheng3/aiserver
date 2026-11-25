"""训练可视化占位。推理不调用这些函数。"""


def create_grid_image(*_args, **_kwargs):
    raise RuntimeError("推理不使用训练可视化")


def test_time_visulize(*_args, **_kwargs):
    raise RuntimeError("推理不使用训练可视化")
