#!/usr/bin/env bash
# VigenFlow quick install with Docker or Podman (docs/docker.md):
#
#   curl -fsSL https://raw.githubusercontent.com/jia1217/vigenflow/main/docker/install.sh | bash
#
# Checks the NPU driver (and offers to install it on Ubuntu), installs Docker on Ubuntu when there is no
# Docker or Podman, asks which models to download, then starts VigenFlow, whose browser UI is at
# http://localhost:11283/. Run it again to update VigenFlow, update the weights or add models;
# downloaded weights and LoRAs are kept.
#
# Unattended runs:
#   VIGENFLOW_MODELS=flux2_klein_4B,z_image_turbo   models to download, or "all" (default: ask; 1 = FLUX.2)
#   VIGENFLOW_ACCEPT_KREA_LICENSE=yes               accept Krea 2 Turbo's license (krea2_turbo)
#   VIGENFLOW_ACCEPT_QWEN_LICENSE=yes               accept Qwen's license (qwen_image_2_1_turbo)
#   CIVITAI_TOKEN, HF_TOKEN                         passed to VigenFlow for LoRA downloads
#   VIGENFLOW_IMAGE                                 another image
set -euo pipefail

IMAGE="${VIGENFLOW_IMAGE:-ghcr.io/jia1217/vigenflow:latest}"
WEIGHTS_VOLUME=vigenflow-weights
WEIGHTS_DIR=/opt/vigenflow/all_model_weights
PORT=11283
OLD_WEBUI_NAME=vigenflow-webui  # the OpenWebUI container that installers before Beta 4 started

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
ALL_MODELS=(flux2_klein_4B z_image_turbo krea2_turbo flux1_schnell qwen_image_2_1_turbo)
MODELS="${VIGENFLOW_MODELS:-}"
if [ -z "$MODELS" ]; then
    if [ -n "$TTY" ]; then
        cat > "$TTY" <<'EOF'

Which models do you want to download?
  1) FLUX.2-klein-4B    8 GB   fastest (about 9 s per image), also edits images
  2) Z-Image-Turbo     12 GB
  3) Krea 2 Turbo      19 GB   has its own license, which you are asked to accept
  4) FLUX.1-schnell    27 GB
  5) Qwen Image 2.1 Turbo  19.6 GB   text-to-image and editing; asks you to accept its license
  a) all five          about 85 GB
EOF
    fi
    MODELS=""
    for choice in $(ask 'Type the numbers separated by spaces [1]:' 1); do
        case "$choice" in
            1|2|3|4|5) MODELS="$MODELS,${ALL_MODELS[choice - 1]}" ;;
            a|A|all) MODELS=all ;;
            *) die "Unknown choice: $choice" ;;
        esac
        [ "$MODELS" = all ] && break
    done
    MODELS="${MODELS#,}"
fi
if [ "$MODELS" = all ]; then
    SELECTED_MODELS=("${ALL_MODELS[@]}")
else
    IFS=',' read -r -a SELECTED_MODELS <<< "$MODELS"
fi
for m in "${SELECTED_MODELS[@]}"; do
    [[ -n "$m" && " ${ALL_MODELS[*]} " == *" $m "* ]] || die "Unknown model $m (choose from: ${ALL_MODELS[*]})"
done

# --- VigenFlow image and weights ----------------------------------------------------------------------
say "Downloading the VigenFlow image"
if ! "${DOCKER[@]}" pull "$IMAGE"; then
    "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 || die "Cannot download $IMAGE."
    say "Using the copy of $IMAGE that is already here"
fi

say "Downloading the model weights (the first time takes a while; an interrupted download resumes)"
prepare_ok=1
for m in "${SELECTED_MODELS[@]}"; do
    # One model per prepare run, so that each license acceptance applies to its own model only.
    PREPARE_ARGS=(--only "$m")
    case "$m" in
        krea2_turbo)
            [ "${VIGENFLOW_ACCEPT_KREA_LICENSE:-}" = yes ] && PREPARE_ARGS+=(--accept_license)
            ;;
        qwen_image_2_1_turbo)
            [ "${VIGENFLOW_ACCEPT_QWEN_LICENSE:-}" = yes ] && PREPARE_ARGS+=(--accept_license)
            ;;
    esac
    if [ -n "$TTY" ]; then  # a terminal, so prepare can ask for the model's license
        "${DOCKER[@]}" run --rm -it -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" "$IMAGE" prepare "${PREPARE_ARGS[@]}" < "$TTY" || prepare_ok=0
    else
        "${DOCKER[@]}" run --rm -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" "$IMAGE" prepare "${PREPARE_ARGS[@]}" || prepare_ok=0
    fi
done

# A model folder exists only once its download is complete (prepare unpacks into <id>.partial first).
listing="$("${DOCKER[@]}" run --rm -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" "$IMAGE" ls "$WEIGHTS_DIR")"
INSTALLED=()
for m in "${ALL_MODELS[@]}"; do
    if grep -qx "$m" <<< "$listing"; then INSTALLED+=("$m"); fi
done
[ ${#INSTALLED[@]} -gt 0 ] || die "No model weights were downloaded; see the messages above."
# The server's default model: z-image-turbo, vgf-serve's own default, when it is here.
DEFAULT_MODEL="${INSTALLED[0]//_/-}"
if [ "${INSTALLED[0]}" = qwen_image_2_1_turbo ]; then DEFAULT_MODEL=qwen-image-2.1-turbo; fi
for m in "${INSTALLED[@]}"; do
    if [ "$m" = z_image_turbo ]; then DEFAULT_MODEL=z-image-turbo; fi
done
[ "$prepare_ok" = 1 ] || say "Some models were not downloaded (see above). Starting with the ones that are here."

# --- Start VigenFlow ----------------------------------------------------------------------------------
say "Starting VigenFlow (default model: $DEFAULT_MODEL)"
# LoRAs download inside the container, into the weights volume; most Civitai LoRAs need a token.
TOKEN_FLAGS=()
for token in CIVITAI_TOKEN HF_TOKEN; do
    if [ -n "${!token:-}" ]; then TOKEN_FLAGS+=(-e "$token=${!token}"); fi
done
"${DOCKER[@]}" rm -f vigenflow >/dev/null 2>&1 || true
"${DOCKER[@]}" run -d --name vigenflow "${NPU_FLAGS[@]}" \
    -p "127.0.0.1:$PORT:$PORT" \
    -v "$WEIGHTS_VOLUME:$WEIGHTS_DIR" \
    ${TOKEN_FLAGS[@]+"${TOKEN_FLAGS[@]}"} \
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

# --- Done ---------------------------------------------------------------------------------------------
say "VigenFlow is running"
printf '  Models downloaded: %s\n' "${INSTALLED[*]}"
cat <<EOF
  Open http://localhost:$PORT/ in your browser, pick a model you downloaded, type a prompt and
  generate. The image size and LoRA styles are on the same page; the editing models take an image.

  Log:                               $D logs -f vigenflow
  Stop / start:                      $D stop vigenflow   /   $D start vigenflow
  Update, or add models:             run this installer again.
EOF
# Installers before Beta 4 also started OpenWebUI. VigenFlow no longer uses it; it is left as it is,
# because its volume holds the user's chats.
if "${DOCKER[@]}" container inspect "$OLD_WEBUI_NAME" >/dev/null 2>&1; then
    cat <<EOF

  VigenFlow no longer uses OpenWebUI. To remove the one an earlier install started:
  Remove it:                         $D rm -f $OLD_WEBUI_NAME
  Then remove its chats too:         $D volume rm $OLD_WEBUI_NAME
EOF
fi
