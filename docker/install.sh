#!/usr/bin/env bash
# VigenFlow quick install with Docker or Podman (docs/docker.md):
#
#   curl -fsSL https://raw.githubusercontent.com/jia1217/vigenflow/main/docker/install.sh | bash
#
# Checks the NPU driver (and offers to install it on Ubuntu), installs Docker on Ubuntu when there is no
# Docker or Podman, asks which models to download, then starts VigenFlow and OpenWebUI. Run it again to
# update VigenFlow, update the weights or add models; downloaded weights are kept.
#
# Unattended runs:
#   VIGENFLOW_MODELS=flux2_klein_4B,z_image_turbo   models to download, or "all" (default: ask; 1 = FLUX.2)
#   VIGENFLOW_ACCEPT_KREA_LICENSE=yes               accept Krea 2 Turbo's license (krea2_turbo)
#   VIGENFLOW_WEBUI=0                               do not start OpenWebUI
#   VIGENFLOW_IMAGE, VIGENFLOW_WEBUI_IMAGE           other images
set -euo pipefail

IMAGE="${VIGENFLOW_IMAGE:-ghcr.io/jia1217/vigenflow:latest}"
WEBUI_IMAGE="${VIGENFLOW_WEBUI_IMAGE:-ghcr.io/open-webui/open-webui:v0.11.4}"
WEIGHTS_VOLUME=vigenflow-weights
WEIGHTS_DIR=/opt/vigenflow/all_model_weights
PORT=11283
WEBUI_NAME=vigenflow-webui  # not open-webui, so an OpenWebUI the user already runs is left alone

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\nError: %s\n' "$*" >&2; exit 1; }

# Questions go to the terminal: under curl | bash, stdin is this script.
TTY=""
if { : < /dev/tty; } 2>/dev/null; then TTY=/dev/tty; fi
ask() {  # ask QUESTION DEFAULT: prints the answer, or DEFAULT without a terminal or an answer
    local answer=""
    if [ -n "$TTY" ]; then
        printf '%s ' "$1" > "$TTY"
        read -r answer < "$TTY" || true
    fi
    printf '%s' "${answer:-$2}"
}
is_ubuntu() { grep -qs '^ID=ubuntu' /etc/os-release; }

[ "$(uname -s)" = Linux ] && [ "$(uname -m)" = x86_64 ] || die "VigenFlow's image runs on x86_64 Linux."

# --- NPU driver ---------------------------------------------------------------------------------------
if [ ! -e /dev/accel/accel0 ]; then
    say "No NPU found: /dev/accel/accel0 is missing, so the AMD XDNA driver is not loaded."
    if is_ubuntu; then
        case "$(ask 'Install the AMD XDNA driver from the Lemonade PPA now? [Y/n]' y)" in
            [Nn]*) ;;
            *)
                sudo add-apt-repository -y ppa:lemonade-team/stable
                sudo apt-get update
                sudo apt-get install -y amdxdna-dkms
                say "The driver is installed. Reboot, then run this installer again."
                exit 0
                ;;
        esac
    fi
    die "Install the AMD XDNA driver (amdxdna-dkms from ppa:lemonade-team/stable on Ubuntu), reboot and run this again."
fi
npu_id="$(cat /sys/class/accel/accel0/device/device 2>/dev/null || true)"
if [ -n "$npu_id" ] && [ "$npu_id" != 0x17f0 ]; then
    say "This NPU ($npu_id) is not a Ryzen AI 300-series XDNA 2 NPU, which VigenFlow's NPU files are built for."
    case "$(ask 'Continue anyway? [y/N]' n)" in [Yy]*) ;; *) exit 1 ;; esac
fi

# --- Docker or Podman ---------------------------------------------------------------------------------
ENGINE=""  # the command to run
KIND=""    # docker or podman: podman-docker's docker command is Podman
if command -v docker >/dev/null 2>&1; then
    ENGINE=docker KIND=docker
    if docker --version 2>/dev/null | grep -qi podman; then KIND=podman; fi
elif command -v podman >/dev/null 2>&1; then
    ENGINE=podman KIND=podman
elif is_ubuntu; then
    say "Neither Docker nor Podman is installed."
    case "$(ask 'Install Docker now (apt install docker.io)? [Y/n]' y)" in
        [Nn]*) die "Install Docker Engine or Podman, then run this again." ;;
    esac
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo usermod -aG docker "$USER"  # takes effect at the next login; until then this script uses sudo
    ENGINE=docker KIND=docker
else
    die "Install Docker Engine (https://docs.docker.com/engine/install/) or Podman, then run this again."
fi

DOCKER=("$ENGINE")
if [ "$KIND" = docker ]; then
    if ! docker info >/dev/null 2>&1; then
        sudo docker info >/dev/null 2>&1 || die "Docker is not running. Start it with: sudo systemctl start docker"
        DOCKER=(sudo docker)
    fi
    [ "$("${DOCKER[@]}" context show 2>/dev/null)" = desktop-linux ] &&
        die "Docker Desktop runs containers in a virtual machine that cannot see the NPU. Use Docker Engine: docker context use default"
fi
if [ "$KIND" = podman ] && [ "$(id -u)" != 0 ]; then
    # Rootless Podman cannot raise the locked-memory limit above the user's own: keep it, and the render group.
    NPU_FLAGS=(--device /dev/accel --ulimit host --group-add keep-groups)
else
    NPU_FLAGS=(--device /dev/accel --ulimit memlock=-1:-1)
fi
D="${DOCKER[*]}"  # for the commands printed at the end

# --- Models -------------------------------------------------------------------------------------------
ALL_MODELS=(flux2_klein_4B z_image_turbo krea2_turbo flux1_schnell)
MODELS="${VIGENFLOW_MODELS:-}"
if [ -z "$MODELS" ]; then
    if [ -n "$TTY" ]; then
        cat > "$TTY" <<'EOF'

Which models do you want to download?
  1) FLUX.2-klein-4B    8 GB   fastest (about 9 s per image), also edits images
  2) Z-Image-Turbo     12 GB
  3) Krea 2 Turbo      19 GB   has its own license, which you are asked to accept
  4) FLUX.1-schnell    27 GB
  a) all four          65 GB
EOF
    fi
    MODELS=""
    for choice in $(ask 'Type the numbers separated by spaces [1]:' 1); do
        case "$choice" in
            1|2|3|4) MODELS="$MODELS,${ALL_MODELS[choice - 1]}" ;;
            a|A|all) MODELS=all ;;
            *) die "Unknown choice: $choice" ;;
        esac
        [ "$MODELS" = all ] && break
    done
    MODELS="${MODELS#,}"
fi
if [ "$MODELS" != all ]; then
    for m in ${MODELS//,/ }; do
        [[ " ${ALL_MODELS[*]} " == *" $m "* ]] || die "Unknown model $m (choose from: ${ALL_MODELS[*]})"
    done
fi

# --- VigenFlow image and weights ----------------------------------------------------------------------
say "Downloading the VigenFlow image"
if ! "${DOCKER[@]}" pull "$IMAGE"; then
    "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || die "Cannot download $IMAGE."
    say "Using the copy of $IMAGE that is already here"
fi

say "Downloading the model weights (the first time takes a while; an interrupted download resumes)"
PREPARE_ARGS=()
[ "$MODELS" != all ] && PREPARE_ARGS+=(--only "$MODELS")
[ "${VIGENFLOW_ACCEPT_KREA_LICENSE:-}" = yes ] && PREPARE_ARGS+=(--accept_license)
prepare_ok=1
if [ -n "$TTY" ]; then  # a terminal, so prepare can ask for Krea's license
    "${DOCKER[@]}" run --rm -it -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" "$IMAGE" prepare "${PREPARE_ARGS[@]}" < "$TTY" || prepare_ok=0
else
    "${DOCKER[@]}" run --rm -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" "$IMAGE" prepare "${PREPARE_ARGS[@]}" || prepare_ok=0
fi

# A model folder exists only once its download is complete (prepare unpacks into <id>.partial first).
listing="$("${DOCKER[@]}" run --rm -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" "$IMAGE" ls "$WEIGHTS_DIR")"
INSTALLED=()
for m in "${ALL_MODELS[@]}"; do
    if grep -qx "$m" <<< "$listing"; then INSTALLED+=("$m"); fi
done
[ ${#INSTALLED[@]} -gt 0 ] || die "No model weights were downloaded; see the messages above."
# The server's default model: z-image-turbo, vgf-serve's own default, when it is here.
DEFAULT_MODEL="${INSTALLED[0]//_/-}"
for m in "${INSTALLED[@]}"; do
    if [ "$m" = z_image_turbo ]; then DEFAULT_MODEL=z-image-turbo; fi
done
[ "$prepare_ok" = 1 ] || say "Some models were not downloaded (see above). Starting with the ones that are here."

# --- Start VigenFlow ----------------------------------------------------------------------------------
say "Starting VigenFlow (default model: $DEFAULT_MODEL)"
"${DOCKER[@]}" rm -f vigenflow >/dev/null 2>&1 || true
"${DOCKER[@]}" run -d --name vigenflow "${NPU_FLAGS[@]}" \
    -p "127.0.0.1:$PORT:$PORT" \
    -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" \
    --restart unless-stopped \
    "$IMAGE" serve --model "$DEFAULT_MODEL" >/dev/null
for _ in $(seq 30); do
    curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 1
done
if ! curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    "${DOCKER[@]}" logs --tail 30 vigenflow >&2 || true
    die "VigenFlow did not start; its log is above."
fi

# --- OpenWebUI ----------------------------------------------------------------------------------------
WEBUI_URL=""
if [ "${VIGENFLOW_WEBUI:-1}" != 0 ]; then
    "${DOCKER[@]}" rm -f "$WEBUI_NAME" >/dev/null 2>&1 || true
    WEBUI_PORT=8080
    while ss -ltnH "sport = :$WEBUI_PORT" 2>/dev/null | grep -q .; do WEBUI_PORT=$((WEBUI_PORT + 1)); done
    say "Starting OpenWebUI on port $WEBUI_PORT (the first start downloads it and takes a few minutes)"
    # Host network: OpenWebUI reaches VigenFlow at 127.0.0.1, an address the browser can open too. The
    # connection settings apply on the first start only; later they are kept in the vigenflow-webui volume.
    "${DOCKER[@]}" run -d --name "$WEBUI_NAME" --network host \
        -v vigenflow-webui:/app/backend/data \
        -e HOST=127.0.0.1 -e PORT="$WEBUI_PORT" \
        -e ENABLE_OLLAMA_API=False \
        -e OPENAI_API_BASE_URL="http://127.0.0.1:$PORT/v1" -e OPENAI_API_KEY=vigenflow \
        -e ENABLE_IMAGE_GENERATION=True -e IMAGE_GENERATION_ENGINE=openai \
        -e IMAGES_OPENAI_API_BASE_URL="http://127.0.0.1:$PORT/v1" -e IMAGES_OPENAI_API_KEY=vigenflow \
        --restart unless-stopped \
        "$WEBUI_IMAGE" >/dev/null
    WEBUI_URL="http://localhost:$WEBUI_PORT"
    for _ in $(seq 180); do
        curl -fsS "http://127.0.0.1:$WEBUI_PORT/health" >/dev/null 2>&1 && break
        sleep 2
    done
    curl -fsS "http://127.0.0.1:$WEBUI_PORT/health" >/dev/null 2>&1 ||
        say "OpenWebUI is still starting; if the page does not open yet, wait a minute ($D logs -f $WEBUI_NAME)"
fi

CONTAINERS="vigenflow${WEBUI_URL:+ $WEBUI_NAME}"

# --- Done ---------------------------------------------------------------------------------------------
say "VigenFlow is running"
printf '  Models downloaded: %s\n' "${INSTALLED[*]}"
if [ -n "$WEBUI_URL" ]; then
    cat <<EOF
  1. Open $WEBUI_URL and create an account (the first one is the admin).
  2. Start a new chat and pick a VigenFlow model you downloaded, for example $DEFAULT_MODEL.
  3. Turn on Image under Integrations in the message box, and type what you want to see.
     The time each image took is shown under it.
EOF
else
    printf '  Connect your OpenWebUI to http://127.0.0.1:%s/v1 as in the VigenFlow README.\n' "$PORT"
fi
cat <<EOF

  Log:                               $D logs -f vigenflow
  Stop / start:                      $D stop $CONTAINERS   /   $D start $CONTAINERS
  Update, or add models:             run this installer again.
EOF
