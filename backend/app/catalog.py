from __future__ import annotations

import shlex

import yaml

from app.config import catalog_path
from app.schemas import ModelDef


def load_models() -> list[ModelDef]:
    path = catalog_path()
    if not path.exists():
        raise FileNotFoundError(f"找不到模型目录: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("models") or []
    models: list[ModelDef] = []
    for item in items:
        engine = dict(item.get("engine") or {})
        command = engine.get("command")
        if isinstance(command, str):
            engine["command"] = shlex.split(command)
        env = item.get("env") or {}
        normalized = {
            **item,
            "engine": engine,
            "env": {str(key): "" if value is None else str(value) for key, value in env.items()},
        }
        model = ModelDef.model_validate(normalized)
        roles = [artifact.role for artifact in model.artifacts]
        if not roles:
            raise ValueError(f"{model.id} 没有权重角色")
        if len(roles) != len(set(roles)):
            raise ValueError(f"{model.id} 的权重角色重复")
        if not model.engine.models:
            raise ValueError(f"{model.id} 的运行时没有声明支持的模型")
        if len(model.engine.models) != len(set(model.engine.models)):
            raise ValueError(f"{model.id} 的运行时把同一个模型声明了两次")
        if not model.engine.weightQuants:
            raise ValueError(f"{model.id} 的运行时没有声明支持的权重类型")
        if len(model.engine.weightQuants) != len(set(model.engine.weightQuants)):
            raise ValueError(f"{model.id} 的运行时把同一个权重类型声明了两次")
        models.append(model)
    ids = [model.id for model in models]
    if len(ids) != len(set(ids)):
        raise ValueError("模型 id 重复")
    known = set(ids)
    for model in models:
        if model.id not in model.engine.models:
            raise ValueError(f"{model.id} 的运行时没有把自己列入支持的模型")
        unknown = [item for item in model.engine.models if item not in known]
        if unknown:
            raise ValueError(f"{model.id} 的运行时声明了未知模型：{', '.join(unknown)}")
    return models


def all_models() -> list[ModelDef]:
    """目录里的模型，加上自定义运行时声明、但目录里还没有的模型。"""
    catalog = load_models()
    from app.runtimes import list_runtimes

    known = {model.id for model in catalog}
    extras: list[ModelDef] = []
    for runtime in list_runtimes():
        if runtime["kind"] != "user":
            continue
        for model_id in runtime["models"]:
            if model_id in known:
                continue
            known.add(model_id)
            extras.append(_custom_model(model_id, runtime))
    return [*catalog, *extras]


def _custom_model(model_id: str, runtime: dict) -> ModelDef:
    from app.schemas import ArtifactSpec, EngineSpec, InterfaceSpec, ModelDef, ResourceSpec

    return ModelDef(
        id=model_id,
        displayName=model_id,
        vendor="自定义",
        task="自定义模型",
        tone="ink",
        summary="由自定义运行时声明。装配时绑定一份权重。",
        interfaces=InterfaceSpec(inputs=["request"], outputs=["response"]),
        engine=EngineSpec(
            image=runtime["image"],
            codeRef=runtime["codeRef"],
            command=list(runtime["command"]),
            port=runtime["port"],
            healthPath=runtime["healthPath"],
            protocol=runtime["protocol"],
            models=list(runtime["models"]),
            weightQuants=list(runtime["weightQuants"]),
        ),
        resources=ResourceSpec(),
        artifacts=[
            ArtifactSpec(
                role="checkpoint",
                label="权重",
                required=True,
                prefer="any",
                hint="这份模型的权重文件或目录",
            )
        ],
    )


def get_model(model_id: str) -> ModelDef | None:
    for model in all_models():
        if model.id == model_id:
            return model
    return None
