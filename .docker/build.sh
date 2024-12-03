#!bash
if [[ "$OSTYPE" == "linux"* ]]; then
  echo '================================================================================'
  echo '                  RUNNING BUILD_MANYLINUX ON LINUX HOST...                      '
  echo '================================================================================'
  docker run -u $(id -u ${USER}):$(id -g ${USER}) -e PYTHONUNBUFFERED=1 --rm -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /io/build_manylinux.py
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
