#!/usr/bin/bash
# docker run -u $(id -u ${USER}):$(id -g ${USER}) -e PYTHONUNBUFFERED=1 --rm -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /io/build_wheel.py
docker run -e PYTHONUNBUFFERED=1 --rm -v $(pwd):/io quay.io/pypa/manylinux2010_x86_64 /io/build_wheel.py
