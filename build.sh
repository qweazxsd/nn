#!/bin/bash

set -xe

export PKG_CONFIG_PATH=$HOME/Software/raylib/lib/pkgconfig/

LIBS="-lm `pkg-config --libs raylib` -ldl"
CFLAGS="-O3 -Wall -Wextra -I./External/ `pkg-config --cflags raylib`"

#clang  $CFLAGS -o xor xor.c  $LIBS
#clang  $CFLAGS -o test test.c  $LIBS
#clang  $CFLAGS -DDMALLOC -DDMALLOC_FUNC_CHECK -o mnist mnist.c  $LIBS -ldmalloc
clang  $CFLAGS -o mnist mnist.c  $LIBS
