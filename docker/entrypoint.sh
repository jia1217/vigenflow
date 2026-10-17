#!/bin/sh
# Entry point of the VigenFlow image (docs/docker.md):
#   serve [vgf-serve options]        start the server on port 11283 (the default)
#   prepare [prepare_base options]   download the model weights into the weights volume
#   any other command                run it, for example bash
set -e

APP=/opt/vigenflow/app
WEIGHTS=/opt/vigenflow/all_model_weights

case "${1:-}" in
    "") set -- serve ;;
    -*) set -- serve "$@" ;;  # only options: they are for vgf-serve
esac

wants_help() {
    for arg; do
        case "$arg" in -h|--help) return 0 ;; esac
    done
    return 1
}

case "$1" in
serve)
    shift
    if ! wants_help "$@"; then
        if [ ! -e /dev/accel/accel0 ]; then
            echo "No NPU in this container: /dev/accel/accel0 is missing." >&2
            echo "Start it with --device /dev/accel --ulimit memlock=-1:-1, on a host with the amdxdna driver." >&2
            exit 1
        fi
        if ! mountpoint -q "$WEIGHTS"; then
            echo "Note: nothing is mounted at $WEIGHTS, so no model weights are found unless they are in the image." >&2
            echo "      Mount the weights volume with -v vigenflow-weights:$WEIGHTS" >&2
        fi
    fi
    # Line-buffered, so docker logs shows each request's timings as soon as vgf-serve prints them: its
    # stdout is a pipe here, not a terminal.
    exec stdbuf -oL -eL "$APP/server_running/vgf-serve" "$@"
    ;;
prepare)
    shift
    if ! wants_help "$@" && ! mountpoint -q "$WEIGHTS"; then
        echo "Nothing is mounted at $WEIGHTS, so the downloaded weights would be lost with the container." >&2
        echo "Mount a volume or a host folder there, for example -v vigenflow-weights:$WEIGHTS" >&2
        exit 1
    fi
    exec stdbuf -oL -eL "$APP/prepare/prepare_base.exe" "$@"
    ;;
*)
    exec "$@"
    ;;
esac
