#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
code_dir="$root_dir/code"
openpi_root="${OPENPI_ROOT:-$root_dir/third_party/openpi}"
openpi_venv="${OPENPI_VENV:-$root_dir/.venvs/openpi_uav}"
openpi_base_python="${OPENPI_BASE_PYTHON:-/home/ysq/.conda/envs/kaggle311/bin/python}"
runtime_env="${RUNTIME_CONDA_ENV:-ysq_qwen}"
python_index="${OPENPI_PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
openpi_repository="${OPENPI_REPOSITORY:-https://github.com/Physical-Intelligence/openpi.git}"
openpi_commit="${OPENPI_COMMIT:-215abfb217dbac7d5f1273282331b9b1866c0479}"
openpi_patch="$code_dir/patches/openpi_uav.patch"
direct_network=(env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy)

if [[ ! -d "$openpi_root/.git" ]]; then
  mkdir -p "$(dirname "$openpi_root")"
  GIT_LFS_SKIP_SMUDGE=1 "${direct_network[@]}" git clone \
    --filter=blob:none --no-checkout "$openpi_repository" "$openpi_root"
  "${direct_network[@]}" git -C "$openpi_root" checkout "$openpi_commit"
fi

if [[ ! -f "$openpi_patch" ]]; then
  echo "[ERROR] Missing OpenPI UAV patch: $openpi_patch" >&2
  exit 1
fi
if git -C "$openpi_root" apply --reverse --check "$openpi_patch" >/dev/null 2>&1; then
  echo "[openpi-setup] UAV patch already applied"
elif git -C "$openpi_root" apply --check "$openpi_patch"; then
  git -C "$openpi_root" apply "$openpi_patch"
  echo "[openpi-setup] applied UAV patch to $openpi_root"
else
  echo "[ERROR] OpenPI source is incompatible with $openpi_patch" >&2
  echo "[ERROR] Expected base commit: $openpi_commit" >&2
  exit 1
fi

if [[ ! -x "$openpi_venv/bin/python" ]]; then
  mkdir -p "$(dirname "$openpi_venv")"
  "$openpi_base_python" -m venv "$openpi_venv"
fi

openpi_python="${OPENPI_PYTHON:-$openpi_venv/bin/python}"
if ! "$openpi_python" -m pip show uv >/dev/null 2>&1; then
  "${direct_network[@]}" "$openpi_python" -m pip install --index-url "$python_index" uv
fi

"${direct_network[@]}" git -C "$openpi_root" submodule update --init --recursive
(
  cd "$openpi_root"
  "${direct_network[@]}" "$openpi_python" -m uv export \
    --frozen --no-dev --no-emit-project \
    | "${direct_network[@]}" env GIT_LFS_SKIP_SMUDGE=1 UV_DEFAULT_INDEX="$python_index" \
      "$openpi_python" -m uv pip install --python "$openpi_python" -r -
  "${direct_network[@]}" env UV_DEFAULT_INDEX="$python_index" \
    "$openpi_python" -m uv pip install --python "$openpi_python" --no-deps -e "$openpi_root"
)

runtime_python="$(conda run -n "$runtime_env" python -c 'import sys; print(sys.executable)')"
"${direct_network[@]}" "$runtime_python" -m pip install \
  --index-url "$python_index" \
  --no-deps \
  -e "$openpi_root/packages/openpi-client"
"${direct_network[@]}" "$runtime_python" -m pip install \
  --index-url "$python_index" \
  --no-deps \
  "dm-tree>=0.1.8" "websockets>=11.0"

echo "[openpi-setup] training environment: $openpi_python"
echo "[openpi-setup] runtime client environment: $runtime_python"
