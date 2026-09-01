#!/bin/bash
# File CARLA/calibrated/ into  <model>/<sunny|night>/  subfolders.
#
# Files are MOVED, not copied, so nothing is duplicated. Re-runnable: the delivery scripts write
# new clips flat into calibrated/, and running this again sweeps them into place.
#
# Two naming fixes applied on the way in:
#   * "cov" was a JOB tag, not a model. Those clips are filed under the model that actually
#     rendered them (town05dense sunny = v50, town07 night = v51).
#   * the old v44 clips carry no weather in their name; they are all daytime, so they go to sunny.
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
set -u
C=$CARLA2REAL_OUT/calibrated
cd "$C" || exit 1

for f in *.mp4; do
  [ -e "$f" ] || continue

  case "$f" in
    *_night_*) W=night ;;
    *_sunny_*) W=sunny ;;
    *)         W=sunny ;;   # v44-era names have no weather field; that set is all daytime
  esac

  # model tag sits between _vp55_ and _FINAL
  M=$(printf '%s' "$f" | sed -n 's/.*_vp55_\(.*\)_FINAL.*/\1/p')
  [ -n "$M" ] || M=unknown

  # "cov" named the coverage JOB, not a model -- resolve it to the real one
  if [ "$M" = cov ]; then
    case "$W" in night) M=v51 ;; sunny) M=v50 ;; esac
  fi

  mkdir -p "$M/$W"
  mv -f "$f" "$M/$W/$f"
done

echo "calibrated/ now:"
for d in */; do
  d=${d%/}
  [ -d "$d" ] || continue
  printf '  %-12s %s\n' "$d" "$(for w in sunny night; do
      n=$(ls "$d/$w"/*.mp4 2>/dev/null | wc -l); [ "$n" -gt 0 ] && printf '%s:%d  ' "$w" "$n"; done)"
done
