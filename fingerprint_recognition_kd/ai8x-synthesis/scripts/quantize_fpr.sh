#!/bin/sh
python quantize.py proj/qat_best.pth.tar proj/qat_best-q.pth.tar --device MAX78000 -v "$@"
