from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Protocol = Literal["http", "websocket", "grpc"]
WeightSource = Literal["local", "upload", "s3"]
WeightKind = Literal["file", "directory", "remote"]
WeightOrigin = Literal["library", "registered", "upload", "s3"]
PreferKind = Literal["file", "directory", "any"]
QuantType = Literal["fp16", "bf16", "int8", "awq", "gptq", "fp8"]


class ArtifactSpec(BaseModel):
    role: str
    label: str
    required: bool = True
    prefer: PreferKind = "any"
    hint: str = ""
    quantized: bool = True


class EngineSpec(BaseModel):
    image: str
    codeRef: str = ""
    command: list[str]
    port: int
    weightsPath: str = "/weights/checkpoint"
    healthPath: str = "/health"
    protocol: Protocol = "http"
    models: list[str] = Field(default_factory=list)
    weightQuants: list[QuantType] = Field(default_factory=list)
    hardware: str = "nvidia"

    @field_validator("weightsPath")
    @classmethod
    def weights_path_ok(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("/") or value == "/" or ".." in value.split("/"):
            raise ValueError("容器内权重路径需要是绝对路径，例如 /weights/checkpoint")
        if len(value) > 200:
            raise ValueError("容器内权重路径过长")
        return value

    @field_validator("hardware")
    @classmethod
    def known_hardware(cls, value: str) -> str:
        if value not in {"nvidia", "ascend", "auto"}:
            raise ValueError("hardware 只能是 nvidia、ascend 或 auto")
        return value


class ResourceSpec(BaseModel):
    accelerators: int = 1
    acceleratorMemoryGi: int = 16
    cpu: int = 4
    memoryGi: int = 32

    @model_validator(mode="before")
    @classmethod
    def accept_gpu_names(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "accelerators" not in data and "gpu" in data:
            data["accelerators"] = data["gpu"]
        if "acceleratorMemoryGi" not in data and "gpuMemoryGi" in data:
            data["acceleratorMemoryGi"] = data["gpuMemoryGi"]
        return data

    @field_validator("accelerators", "acceleratorMemoryGi", "cpu", "memoryGi")
    @classmethod
    def non_negative(cls, value: int) -> int:
        if value < 0 or value > 4096:
            raise ValueError("资源数量需要在 0 到 4096 之间")
        return value


class InterfaceSpec(BaseModel):
    inputs: list[str]
    outputs: list[str]


class ModelDef(BaseModel):
    id: str
    displayName: str
    vendor: str
    task: str
    tone: str = "ink"
    summary: str
    interfaces: InterfaceSpec
    engine: EngineSpec
    resources: ResourceSpec
    env: dict[str, str] = Field(default_factory=dict)
    artifacts: list[ArtifactSpec]


class WeightAsset(BaseModel):
    id: str
    source: WeightSource
    origin: WeightOrigin
    kind: WeightKind
    name: str
    uri: str
    removable: bool
    missing: bool = False
    sizeBytes: int | None = None
    fileCount: int | None = None
    sha256: str | None = None
    relativePath: str | None = None
    localPath: str | None = None
    endpoint: str | None = None
    region: str | None = None
    bucket: str | None = None
    key: str | None = None
    note: str | None = None
    createdAt: str | None = None


class S3Register(BaseModel):
    uri: str
    endpoint: str = ""
    region: str = ""
    note: str = ""


class S3Verify(BaseModel):
    uri: str
    endpoint: str = ""
    region: str = ""


class S3VerifyResult(BaseModel):
    ok: bool
    sizeBytes: int | None = None
    detail: str | None = None


class LocalPathRegister(BaseModel):
    path: str
    note: str = ""


class WeightBinding(BaseModel):
    role: str
    assetId: str
    quant: QuantType | None = None


class EngineOverride(BaseModel):
    image: str
    codeRef: str = ""
    command: list[str]
    port: int
    healthPath: str = ""
    protocol: Protocol
    resources: ResourceSpec
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("image")
    @classmethod
    def image_ok(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 300:
            raise ValueError("请填写引擎镜像")
        return value

    @field_validator("command")
    @classmethod
    def command_ok(cls, value: list[str]) -> list[str]:
        if not value or any(not part.strip() for part in value):
            raise ValueError("启动命令不能为空")
        return [part.strip() for part in value]

    @field_validator("port")
    @classmethod
    def port_ok(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("端口需要在 1 到 65535")
        return value


class PackageCreate(BaseModel):
    modelId: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    weights: list[WeightBinding]
    engine: EngineOverride

    @field_validator("name")
    @classmethod
    def name_ok(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 80:
            raise ValueError("名称需要 1 到 80 个字符")
        if any(char in value for char in '\\/:*?"<>|'):
            raise ValueError("名称不能包含路径符号")
        return value

    @field_validator("version")
    @classmethod
    def version_ok(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 40:
            raise ValueError("版本需要 1 到 40 个字符")
        return value

    @field_validator("description")
    @classmethod
    def description_ok(cls, value: str) -> str:
        value = value.strip()
        if len(value) > 2000:
            raise ValueError("说明不能超过 2000 字")
        return value


class PackageView(BaseModel):
    absoluteRoot: str
    manifest: dict


class BuildRequest(BaseModel):
    image: str
    force: bool = False


class BaseCreate(BaseModel):
    name: str
    image: str
    note: str = ""


class DeployCreate(BaseModel):
    packageId: str
    hostPort: int | None = None

    @field_validator("hostPort")
    @classmethod
    def host_port_ok(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if not 1 <= value <= 65535:
            raise ValueError("宿主机端口要在 1 到 65535")
        return value


class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0

    @field_validator("prompt")
    @classmethod
    def prompt_ok(cls, value: str) -> str:
        value = value.strip()
        if not value or len(value) > 8000:
            raise ValueError("请填写 1 到 8000 字的提示")
        return value

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_ok(cls, value: int) -> int:
        if not 1 <= value <= 2048:
            raise ValueError("max_tokens 需要在 1 到 2048")
        return value

    @field_validator("temperature")
    @classmethod
    def temperature_ok(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError("temperature 需要在 0 到 2")
        return value


class MetaInfo(BaseModel):
    dataDir: str
    localLibrary: str
    catalogPath: str
    packageDir: str
