#!/usr/bin/env bash
# Build a minimal FFmpeg with libass (subtitle burn-in) into ~/.local/opt/ffmpeg-karaoke.
#
# Why: Homebrew's `ffmpeg` has no libass, and `ffmpeg-full` on Intel macOS has no bottles for
# llvm/rust/etc., so it compiles for hours. This builds only what karaoke needs (~15 min, 4 cores).
#
#   brew install libass x264 nasm pkgconf   # libass pulls freetype, fribidi, harfbuzz, libunibreak
#   scripts/build_ffmpeg.sh
#
# karaoke/config.py finds the result automatically (or set KARAOKE_FFMPEG / KARAOKE_FFPROBE).
set -euo pipefail

VERSION="${FFMPEG_VERSION:-8.1.1}"
PREFIX="${PREFIX:-$HOME/.local/opt/ffmpeg-karaoke}"
BUILD="${BUILD_DIR:-$HOME/.cache/karaoke-build}"
JOBS="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

mkdir -p "$BUILD"
cd "$BUILD"
if [ ! -d "ffmpeg-$VERSION" ]; then
  curl -fL --retry 3 -o "ffmpeg-$VERSION.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$VERSION.tar.xz"
  tar -xf "ffmpeg-$VERSION.tar.xz"
fi
cd "ffmpeg-$VERSION"

./configure \
  --prefix="$PREFIX" \
  --enable-gpl \
  --enable-libx264 \
  --enable-libass \
  --enable-libfreetype \
  --enable-libharfbuzz \
  --enable-libfribidi \
  --enable-libfontconfig \
  --disable-doc \
  --disable-ffplay \
  --disable-debug

make -j"$JOBS"
make install

"$PREFIX/bin/ffmpeg" -hide_banner -filters | grep -E " (ass|subtitles|xfade|zoompan|perspective|gblur) "
echo "ffmpeg with libass installed in $PREFIX/bin"
