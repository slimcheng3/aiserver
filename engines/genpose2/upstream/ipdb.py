"""占位，避免上游训练代码在推理时依赖调试器。"""


def set_trace(*_args, **_kwargs) -> None:
    raise RuntimeError("推理路径里不该走到调试断点")
