# 模型打包台

把 SAM 3、GenPose2、π0.5、Qwen 收成一份模型包。推理部署不在这里做，部署平台以后只读 `manifest.json`。

模型包不复制权重。本地文件和上传文件记本机路径，S3 只记地址，避免把几十 GB 再拷一份。

## 运行

在仓库根目录执行下面的命令。它会先检查环境，再安装还没有的依赖，然后在后台启动。服务已经在运行时直接返回，不会重新安装，也不会再启动一份。服务没在运行时，虚拟环境在就不再创建；依赖清单没变就跳过安装，变了才重新安装。上次执行中断的话，再执行只会补上没完成的那一步。

```bash
scripts/console.sh start
```

查看状态和停止：

```bash
scripts/console.sh status
scripts/console.sh stop
```

页面是 http://127.0.0.1:5173 ，接口是 http://127.0.0.1:8787 ，接口文档在 http://127.0.0.1:8787/docs 。日志在 `data/run/api.log` 和 `data/run/web.log`。当前用户没有 docker 权限时，启动脚本会用 `sg docker` 再执行一次。

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
