ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.description="host_flux_klein_bf16_img_lora build environment"
LABEL org.opencontainers.image.source="local"

ENV DEBIAN_FRONTEND=noninteractive

ARG UBUNTU_PPA=""
ARG BACKPORTS=""
ARG INSTALL_XRT_DEV=1
ARG PREFETCH_CARGO_DEPS=0

RUN if [ -n "$UBUNTU_PPA" ]; then \
        apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository -y "$UBUNTU_PPA"; \
    fi

RUN if [ -n "$BACKPORTS" ]; then \
        echo "deb http://deb.debian.org/debian $BACKPORTS main" >> /etc/apt/sources.list; \
    fi

RUN apt-get update && apt-get install -y \
        build-essential \
        gcc-13 \
        g++-13 \
        make \
        cmake \
        ninja-build \
        git \
        curl \
        ca-certificates \
        python3 \
        pkg-config \
        libboost-dev \
        libboost-program-options-dev \
        libboost-filesystem-dev \
        nlohmann-json3-dev \
        libstb-dev \
        uuid-dev \
        libdrm-dev \
    && if [ "$INSTALL_XRT_DEV" = "1" ]; then \
        if [ -n "$BACKPORTS" ]; then \
            apt-get install -t "$BACKPORTS" -y libxrt-dev; \
        else \
            apt-get install -y libxrt-dev; \
        fi; \
    fi \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        RUSTUP_INIT_SKIP_PATH_CHECK=yes sh -s -- -y --profile minimal --default-toolchain stable

COPY external_lib/tokenizers-cpp/rust/ /tmp/tokenizers-cpp-rust/
RUN if [ "$PREFETCH_CARGO_DEPS" = "1" ]; then \
        cargo fetch --manifest-path /tmp/tokenizers-cpp-rust/Cargo.toml --locked; \
    else \
        echo "Skipping Cargo dependency prefetch"; \
    fi

WORKDIR /workspace

CMD ["/bin/bash"]
