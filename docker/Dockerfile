# VigenFlow server image: vgf-serve, the five NPU model workers, their NPU files and the weight
# downloader, laid out as in the Ubuntu release package. The model weights are not in the image;
# `prepare` downloads them into a volume mounted at /opt/vigenflow/all_model_weights.
# The host needs only the amdxdna driver: XRT is in the image.
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t vigenflow .
#
# How to run it: docs/docker.md.

ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS build

ARG XRT_PPA=ppa:lemonade-team/stable
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y ca-certificates software-properties-common \
    && add-apt-repository -y "$XRT_PPA" \
    && apt-get update \
    && apt-get install -y \
        build-essential \
        g++-13 \
        zip \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-program-options-dev \
        nlohmann-json3-dev \
        libstb-dev \
        libpng-dev \
        uuid-dev \
        libxrt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# The launchers as in ubuntu-build.yml, then vgf-serve, prepare_base.exe and the package folder as in
# the release zip.
RUN for model in flux1_schnell flux2_klein_4B flux2_klein_4B_edit krea2_turbo z_image_turbo; do \
        make -C "src/models/$model" clean all; \
    done \
    && PACKAGE_NAME=vigenflow OUT_ROOT=/out bash src/server_unified/package_ubuntu.sh


FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="VigenFlow"
LABEL org.opencontainers.image.description="FLUX, Z-Image and Krea 2 image generation on the AMD Ryzen AI NPU, served to OpenWebUI"
LABEL org.opencontainers.image.source="https://github.com/jia1217/vigenflow"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The XRT PPA's apt source with its signing key, as add-apt-repository wrote it in the build stage.
COPY --from=build /etc/apt/sources.list.d/ /etc/apt/sources.list.d/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libxrt-npu2 \
        libboost-program-options1.83.0 \
        libgomp1 \
        libpng16-16t64 \
        curl \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/vigenflow /opt/vigenflow/app
COPY docker/entrypoint.sh /usr/local/bin/vigenflow

# images/ is where the workers write their PNGs, so any --user can run the server.
RUN chmod 0755 /usr/local/bin/vigenflow \
    && chmod 1777 /opt/vigenflow/app/server_running/images \
    && mkdir -p /opt/vigenflow/all_model_weights \
    && cd /opt/vigenflow/app \
    && if ldd server_running/vgf-serve server_running/exe_models/*/run.exe server_running/lib/*.so \
            prepare/prepare_base.exe | grep "not found"; then \
        echo "A runtime library is missing from the image" >&2; exit 1; \
    fi

WORKDIR /opt/vigenflow/app/server_running
EXPOSE 11283

# tini is PID 1: vgf-serve installs no signal handlers, so as PID 1 it would ignore docker stop.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/vigenflow"]
CMD ["serve"]
