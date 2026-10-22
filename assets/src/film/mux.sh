#!/usr/bin/env bash
# Soundtrack + picture -> work/mkshv-film.mp4 at -14 LUFS, true peak <= -1.5 dBTP (two-pass loudnorm).
set -euo pipefail
cd "$(dirname "$0")/work"
DUR=27.5
PRE="afade=t=out:st=26.8:d=0.7,alimiter=limit=0.25:attack=1:release=80:level=disabled"
M=$(ffmpeg -hide_banner -i soundtrack_raw.wav -t $DUR -af "$PRE,loudnorm=I=-14:TP=-1.5:LRA=18:print_format=json" -f null - 2>&1 | sed -n '/^{/,/^}/p')
g(){ echo "$M" | grep "\"$1\"" | grep -oE '[-0-9.]+'; }
ffmpeg -y -loglevel error -i picture.mp4 -i soundtrack_raw.wav -t $DUR -map 0:v -map 1:a -c:v copy \
  -af "$PRE,loudnorm=I=-14:TP=-1.5:LRA=18:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true" \
  -ar 48000 -ac 2 -c:a aac -b:a 192k -movflags +faststart mkshv-film.mp4
ffmpeg -hide_banner -i mkshv-film.mp4 -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|LRA|Peak):" | tail -3
