#!/usr/bin/env bash
# 启动或停止交付台。启动前先做环境检查。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN="$ROOT/data/run"
API_PID="$RUN/api.pid"
WEB_PID="$RUN/web.pid"
API_LOG="$RUN/api.log"
WEB_LOG="$RUN/web.log"
API_URL="http://127.0.0.1:8787/api/health"
WEB_URL="http://127.0.0.1:5173/"

usage() {
  echo "用法: scripts/console.sh start|stop|status"
}

prepare_path() {
  if command -v node >/dev/null 2>&1; then
    return
  fi
  local bin
  for bin in "$ROOT"/.tools/node-*/bin; do
    if [[ -x "$bin/node" ]]; then
      export PATH="$bin:$PATH"
      return
    fi
  done
}

ensure_docker_group() {
  if docker info --format '{{.ServerVersion}}' >/dev/null 2>&1; then
    return
  fi
  if [[ -n "${AISERVER_IN_DOCKER_GROUP:-}" ]]; then
    return
  fi
  local err
  err=$(docker info --format '{{.ServerVersion}}' 2>&1 || true)
  if [[ "$err" == *[Pp]ermission* ]] && command -v sg >/dev/null 2>&1 && getent group docker >/dev/null 2>&1; then
    export AISERVER_IN_DOCKER_GROUP=1
    exec sg docker -c "$(printf '%q start' "$(readlink -f "$0")")"
  fi
}

cmdline() {
  [[ -r "/proc/$1/cmdline" ]] || return 0
  tr '\0' ' ' <"/proc/$1/cmdline" || true
}

is_api() {
  local cmd
  cmd=$(cmdline "$1")
  [[ "$cmd" == "$ROOT/.venv/bin/python -m uvicorn app.main:app"* ]]
}

is_web() {
  local cmd cwd
  cmd=$(cmdline "$1")
  cwd=$(readlink -f "/proc/$1/cwd" 2>/dev/null || true)
  if [[ "$cmd" == *"$ROOT/frontend"* && "$cmd" == *vite* ]]; then
    return 0
  fi
  [[ "$cwd" == "$ROOT/frontend" && "$cmd" == npm* && "$cmd" == *dev* ]]
}

repo_pids() {
  local kind=$1 pid
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    if [[ "$kind" == api ]] && is_api "$pid"; then
      echo "$pid"
    elif [[ "$kind" == web ]] && is_web "$pid"; then
      echo "$pid"
    fi
  done
}

listening() {
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(0.3); raise SystemExit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1])))==0 else 1)' "$1"
}

descendants() {
  local pid=$1 child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null || true); do
    descendants "$child"
    echo "$child"
  done
}

stop_pid() {
  local pid=$1 child
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  [[ -d "/proc/$pid" ]] || return 0
  local targets=()
  while read -r child; do
    [[ -n "$child" ]] && targets+=("$child")
  done < <(descendants "$pid")
  targets+=("$pid")
  kill -TERM "${targets[@]}" 2>/dev/null || true
  local _i
  for _i in 1 2 3 4 5 6 7 8 9 10; do
    [[ -d "/proc/$pid" ]] || return 0
    sleep 0.3
  done
  kill -KILL "${targets[@]}" 2>/dev/null || true
}

stop_recorded() {
  local file=$1 kind=$2 pid
  [[ -f "$file" ]] || return 0
  pid=$(<"$file")
  rm -f "$file"
  if [[ "$kind" == api ]] && is_api "$pid"; then
    stop_pid "$pid"
  elif [[ "$kind" == web ]] && is_web "$pid"; then
    stop_pid "$pid"
  fi
}

stop_matching() {
  local kind=$1 pid
  local found=()
  while read -r pid; do
    [[ -n "$pid" ]] && found+=("$pid")
  done < <(repo_pids "$kind")
  for pid in "${found[@]+"${found[@]}"}"; do
    stop_pid "$pid"
  done
}

cmd_stop() {
  stop_recorded "$API_PID" api
  stop_recorded "$WEB_PID" web
  stop_matching api
  stop_matching web
  local _i
  for _i in 1 2 3 4 5 6 7 8 9 10; do
    if ! listening 8787 && ! listening 5173; then
      break
    fi
    sleep 0.3
  done
  if listening 8787 || listening 5173; then
    if ! repo_pids api | grep -q . && ! repo_pids web | grep -q .; then
      echo "交付台进程已停。8787 或 5173 仍被其他程序占用。"
      return 0
    fi
  fi
  echo "交付台已停止。"
}

cmd_status() {
  local api web
  api=$(repo_pids api | tr '\n' ' ' || true)
  web=$(repo_pids web | tr '\n' ' ' || true)
  if [[ -z "${api// /}" && -z "${web// /}" ]]; then
    echo "交付台没有在运行。"
    return 0
  fi
  [[ -n "${api// /}" ]] && echo "接口进程 ${api}"
  [[ -n "${web// /}" ]] && echo "页面进程 ${web}"
  listening 8787 && echo "接口 http://127.0.0.1:8787"
  listening 5173 && echo "页面 http://127.0.0.1:5173"
}

file_hash() {
  sha256sum "$1" | awk '{print $1}'
}

ensure_deps() {
  local py="$ROOT/.venv/bin/python"
  local req="$ROOT/backend/requirements.txt"
  local req_stamp="$ROOT/.venv/.aiserver-requirements.sha256"
  if [[ ! -x "$py" ]]; then
    if [[ -d "$ROOT/.venv" ]]; then
      echo "虚拟环境不完整。删掉 .venv 后再执行 scripts/console.sh start"
      return 1
    fi
    echo "创建虚拟环境"
    python3 -m venv "$ROOT/.venv"
  fi
  local req_hash
  req_hash=$(file_hash "$req")
  if [[ -f "$req_stamp" && "$(<"$req_stamp")" == "$req_hash" ]]; then
    echo "Python 依赖已安装，跳过"
  else
    echo "安装 Python 依赖"
    "$py" -m pip install -r "$req"
    printf '%s\n' "$req_hash" >"$req_stamp"
  fi

  local lock="$ROOT/frontend/package-lock.json"
  [[ -f "$lock" ]] || lock="$ROOT/frontend/package.json"
  local npm_stamp="$ROOT/frontend/node_modules/.aiserver-npm.sha256"
  local npm_hash
  npm_hash=$(file_hash "$lock")
  if [[ -d "$ROOT/frontend/node_modules" && -f "$npm_stamp" && "$(<"$npm_stamp")" == "$npm_hash" ]]; then
    echo "前端依赖已安装，跳过"
  else
    echo "安装前端依赖"
    npm install --prefix "$ROOT/frontend"
    printf '%s\n' "$npm_hash" >"$npm_stamp"
  fi
}

wait_url() {
  local url=$1
  python3 -c 'import sys,time,urllib.request
url=sys.argv[1]
for _ in range(40):
    try:
        with urllib.request.urlopen(url, timeout=1) as resp:
            if resp.status < 500:
                raise SystemExit(0)
    except Exception:
        time.sleep(0.5)
raise SystemExit(1)' "$url"
}

cmd_start() {
  prepare_path
  local api web
  api=$(repo_pids api || true)
  web=$(repo_pids web || true)
  if [[ -n "$api" && -n "$web" ]] && listening 8787 && listening 5173; then
    echo "交付台已经在运行：http://127.0.0.1:5173"
    return 0
  fi
  if [[ -n "$api" || -n "$web" ]] || listening 8787 || listening 5173; then
    echo "接口或页面只起来了一部分，或端口被占用。先执行 scripts/console.sh stop"
    return 1
  fi
  ensure_docker_group
  python3 "$ROOT/scripts/check_env.py"
  ensure_deps
  python3 "$ROOT/scripts/check_env.py" --ready
  mkdir -p "$RUN"
  cd "$ROOT"
  setsid "$ROOT/.venv/bin/python" -m uvicorn app.main:app \
    --app-dir "$ROOT/backend" \
    --host 0.0.0.0 \
    --port 8787 \
    --reload \
    --reload-dir "$ROOT/backend" \
    >>"$API_LOG" 2>&1 </dev/null &
  echo $! >"$API_PID"
  cd "$ROOT/frontend"
  setsid npm run dev -- --host 0.0.0.0 --port 5173 --strictPort \
    >>"$WEB_LOG" 2>&1 </dev/null &
  echo $! >"$WEB_PID"
  cd "$ROOT"
  if ! wait_url "$API_URL" || ! wait_url "$WEB_URL"; then
    echo "启动失败，正在停掉刚拉起的进程。日志在 data/run/api.log 和 data/run/web.log"
    cmd_stop
    return 1
  fi
  echo "交付台已启动"
  echo "页面  http://127.0.0.1:5173"
  echo "接口  http://127.0.0.1:8787"
}

case "${1:-}" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  *) usage; exit 2 ;;
esac
