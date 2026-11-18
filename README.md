# 模型打包台

把 SAM 3、GenPose2、π0.5、Qwen 收成一份模型包。推理部署不在这里做，部署平台以后只读 `manifest.json`。

模型包不复制权重。本地文件和上传文件记本机路径，S3 只记地址，避免把几十 GB 再拷一份。

## 运行

```powershell
python -m pip install -r backend\requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8787 --reload
```

另一个终端：

```powershell
cd frontend
npm install
npm run dev
```

打开 http://127.0.0.1:5173 。接口文档在 http://127.0.0.1:8787/docs 。

## 要改的地方

引擎镜像、启动命令、服务代码位置在 `backend/catalog/models.yaml`。保存后刷新页面即可。生成包时会把当时的值抄进 manifest，之后改目录不影响已生成的包。

权重可以来自三处：

- 丢进 `data/library/local`（文件，或一个模型目录）
- 在页面上登记本机已有路径
- 在页面上登记 `s3://bucket/key`（可用 MinIO endpoint）
- 从浏览器上传，文件落在 `data/library/uploads`

生成的包在 `data/packages/<id>/manifest.json`。

## 模型包

```json
{
  "apiVersion": "aiserver.modelpackage/v1",
  "kind": "ModelPackage",
  "metadata": {"id": "pkg_...", "name": "车间分割", "version": "1.0.0"},
  "model": {"id": "sam3"},
  "engine": {"image": "local/sam3-engine:latest", "command": ["python", "serve.py"], "port": 8080},
  "weights": [{"role": "checkpoint", "source": "s3", "uri": "s3://models/sam3/sam3.pt"}]
}
```

本地路径只在这台打包机上有效。换机器部署时用 S3 地址。
