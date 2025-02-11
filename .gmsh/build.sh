#!/usr/bin/env bash
if [[ "$OSTYPE" == "linux"* ]]; then
  echo '================================================================================'
  echo '                  RUNNING BUILD_MANYLINUX ON LINUX HOST...                      '
  echo '================================================================================'
  arch=$(uname -m)
  mkdir -p build_manylinux
  cp build_manylinux.py build_manylinux/.
  docker run -u $(id -u ${USER}):$(id -g ${USER}) -e PYTHONUNBUFFERED=1 --rm -v $(pwd)/patches:/patches -v $(pwd)/build_manylinux:/io quay.io/pypa/manylinux2014_${arch} /io/build_manylinux.py
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo '================================================================================'
  echo '                    RUNNING BUILD_MACOS ON MACOS HOST...                        '
  echo '================================================================================'
  python3 build_macos.py
else
  echo '================================================================================'
  echo '                UNSPPORTED DOCKER HOST DETECTED. ABORTING...                    '
  echo '================================================================================'
fi
