# Shared: block until no other process is using the GPU.
#
# `pgrep -f <pattern>` matches this script's own wrapper shell, because the harness embeds the
# whole command text in `bash -c '...'`. Filtering on "^$$" only removes the current shell, not
# the wrapper or the launcher, so the naive form never exits. Guard on the full process ancestry
# plus the shell-snapshot signature instead. This exact bug wedged retry_coverage.sh for two
# hours on 2026-08-23.
gpu_busy() {
  local anc="" p=$$ pid a
  while [ -n "$p" ] && [ "$p" != "1" ]; do anc="$anc $p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); done
  for pid in $(pgrep -f 'main_IRT|test\.py --name|train\.py --name|CarlaUE4|dz_label|gen_depth_moge' 2>/dev/null); do
    case " $anc " in *" $pid "*) continue ;; esac
    a=$(ps -o args= -p "$pid" 2>/dev/null)
    # pgrep can return a pid that exits before ps reads it; empty args means gone, not busy
    [ -z "$a" ] && continue
    case "$a" in *shell-snapshots*|*gpu_wait*|*retry_coverage*|*eval_v52*) continue ;; esac
    return 0
  done
  return 1
}
wait_for_gpu() { local n=0; while gpu_busy && [ $n -lt 720 ]; do sleep 60; n=$((n+1)); done; }
