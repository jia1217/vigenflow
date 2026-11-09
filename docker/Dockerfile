# VigenFlow server image: the published Ubuntu release package (vgf-serve with its built-in browser UI,
# the seven NPU model workers, their NPU files and the weight downloader) on Ubuntu 24.04 with XRT.
# The model weights are not in the image; `prepare` downloads them into a volume mounted at
# /opt/vigenflow/all_model_weights. The host needs only the amdxdna driver: XRT is in the image.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t vigenflow .
#
# For another release, set VIGENFLOW_VERSION and PACKAGE_SHA256 (the release asset's sha256 digest).
# How to run it: docs/docker.md.

ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS package

ARG XRT_PPA=ppa:lemonade-team/stable
ARG VIGENFLOW_VERSION=0.2.0-beta.4
ARG PACKAGE_URL=https://github.com/jia1217/vigenflow/releases/download/v${VIGENFLOW_VERSION}/vgf_${VIGENFLOW_VERSION}_ubuntu_amd64.zip
ARG PACKAGE_SHA256=d53ceb34f7a6b3c268bde5eb1f4cfd16fbe795d418e64e5046b66ee8f2f5855d
ENV DEBIAN_FRONTEND=noninteractive

# The XRT PPA is added here so that the runtime stage can copy its apt source.
RUN apt-get update \
    && apt-get install -y ca-certificates curl unzip software-properties-common \
    && add-apt-repository -y "$XRT_PPA" \
    && rm -rf /var/lib/apt/lists/*

# The zip holds one folder; it becomes /out/vigenflow whatever its name.
RUN set -eu; \
    curl -fsSL --retry 3 -o /tmp/vigenflow.zip "$PACKAGE_URL"; \
    echo "$PACKAGE_SHA256  /tmp/vigenflow.zip" | sha256sum --check --quiet; \
    mkdir -p /tmp/unpacked /out; \
    unzip -q /tmp/vigenflow.zip -d /tmp/unpacked; \
    set -- /tmp/unpacked/*; \
    test "$#" -eq 1; \
    test -d "$1"; \
    mv "$1" /out/vigenflow; \
    rm -rf /tmp/vigenflow.zip /tmp/unpacked


FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="VigenFlow"
LABEL org.opencontainers.image.description="FLUX, Z-Image, Krea 2 and Qwen Image generation and editing on the AMD Ryzen AI NPU, with a built-in browser UI and an OpenAI-compatible API"
LABEL org.opencontainers.image.source="https://github.com/jia1217/vigenflow"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The XRT PPA's apt source with its signing key, as add-apt-repository wrote it in the package stage.
COPY --from=package /etc/apt/sources.list.d/ /etc/apt/sources.list.d/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxrt-npu2 \
        libboost-program-options1.83.0 \
        libgomp1 \
        libjpeg-turbo8 \
        libssl3t64 \
        libpng16-16t64 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=package /out/vigenflow /opt/vigenflow/app
COPY docker/entrypoint.sh /usr/local/bin/vigenflow

# images/ is where the workers write their PNGs, so any --user can run the server. The workers need
# AVX-512 (as on every Ryzen AI 300-series CPU), so they run only when the build machine has it.
RUN set -eu; \
    chmod 0755 /usr/local/bin/vigenflow; \
    chmod 1777 /opt/vigenflow/app/server_running/images; \
    mkdir -p /opt/vigenflow/all_model_weights; \
    cd /opt/vigenflow/app; \
    sha256sum --check --quiet SHA256SUMS; \
    for binary in server_running/vgf-serve server_running/exe_models/*/run.exe prepare/prepare_base.exe; do \
        dependencies="$(ldd "$binary")"; \
        if printf '%s\n' "$dependencies" | grep -q 'not found'; then \
            printf 'Missing runtime libraries for %s:\n%s\n' "$binary" "$dependencies" >&2; exit 1; \
        fi; \
    done; \
    if grep -qw avx512f /proc/cpuinfo; then \
        for worker in server_running/exe_models/*/run.exe; do \
            "$worker" --help >/dev/null; \
        done; \
    else \
        echo "This build machine has no AVX-512, which the workers need: skipping their --help check."; \
    fi; \
    test ! -e server_running/lib; \
    for task in base edit; do \
        test -d "all_npu_files/qwen_image/$task"; \
    done

WORKDIR /opt/vigenflow/app/server_running
EXPOSE 11283

# tini is PID 1: vgf-serve installs no signal handlers, so as PID 1 it would ignore docker stop.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/vigenflow"]
CMD ["serve"]
