"""在 vLLM 镜像里把 Hugging Face 模型目录量化成 fp8、awq 或 gptq。

官方镜像的入口是服务进程，所以这个文件由量化作业单独执行。
fp8 动态方案只根据权重定尺子。awq 和 gptq 还要读校准文本。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = os.environ["WEIGHTS_PATH"]
DST = os.environ["QUANT_OUTPUT"]
QUANT = os.environ.get("QUANT", "fp8")
CALIBRATION = os.environ.get("CALIBRATION_PATH", "")


# 镜像自带 torch 2.8 和 transformers 4.57。0.15 会把 torch 升到 2.14，torchvision 随之失效。
COMPATIBLE_PACKAGES = (
    "llmcompressor==0.9.0.6",
    "compressed-tensors==0.13.0",
    "datasets==4.4.1",
)


def ensure_compressor() -> None:
    try:
        import llmcompressor  # noqa: F401
    except ImportError:
        print("镜像里没有 llmcompressor，正在安装与当前 torch 匹配的版本", flush=True)
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--upgrade-strategy",
                "only-if-needed",
                *COMPATIBLE_PACKAGES,
            ],
        )


def make_config_loadable(path: Path) -> None:
    """vLLM 0.11 的量化参数不认识 scale_dtype 和 zp_dtype。这两个空字段删掉才能启动。"""
    config_path = path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    groups = (payload.get("quantization_config") or {}).get("config_groups") or {}
    for group in groups.values():
        if not isinstance(group, dict):
            continue
        for key in ("weights", "input_activations", "output_activations"):
            section = group.get(key)
            if isinstance(section, dict):
                section.pop("scale_dtype", None)
                section.pop("zp_dtype", None)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recipe_for(quant: str):
    if quant == "fp8":
        from llmcompressor.modifiers.quantization import QuantizationModifier

        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    if quant == "awq":
        from llmcompressor.modifiers.awq import AWQModifier

        return AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])
    if quant == "gptq":
        from llmcompressor.modifiers.quantization import GPTQModifier

        return GPTQModifier(ignore=["lm_head"], scheme="W4A16", targets=["Linear"], offload_hessians=True)
    raise SystemExit(f"当前只做 fp8、awq、gptq，收到 {quant}")


def main() -> None:
    src = Path(SRC)
    dst = Path(DST)
    if not (src / "config.json").is_file():
        raise SystemExit(f"{src} 不是 Hugging Face 模型目录，缺少 config.json")
    dst.mkdir(parents=True, exist_ok=True)
    ensure_compressor()

    import torch
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载 {src}", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(src)
    recipe = recipe_for(QUANT)
    if QUANT == "fp8":
        print("按 fp8 量化线性层，lm_head 保持原精度", flush=True)
        oneshot(model=model, recipe=recipe)
    else:
        folder = Path(CALIBRATION)
        files = sorted(folder.glob("*.json"))
        if not files:
            raise SystemExit(f"{folder} 里没有校准文本")
        count = sum(len(json.loads(path.read_text(encoding="utf-8"))) for path in files)
        print(f"按 {QUANT} 量化线性层，使用 {count} 条校准文本，lm_head 保持原精度", flush=True)
        oneshot(
            model=model,
            dataset="json",
            dataset_path=str(folder),
            splits="train",
            recipe=recipe,
            num_calibration_samples=count,
            max_seq_length=512,
            shuffle_calibration_samples=False,
        )
    print(f"写入 {dst}", flush=True)
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    make_config_loadable(dst)
    print("量化完成", flush=True)


if __name__ == "__main__":
    main()
