# Quick Install

[← Back to VigenFlow](../README.md)

For **Beta 4's FLUX.1-schnell browser UI**, use the [Beta 4 release instructions](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.4). The commands below install the multi-model release with OpenWebUI.

The quickest way to try VigenFlow: one command installs it, downloads the models you choose, and starts it together with OpenWebUI. Press Enter at the model question for FLUX.2-klein-4B (8 GB), the fastest. Run the same command again to update VigenFlow or add models.

## 🪟 Windows

Install the [NPU driver](../README.md#-install-the-npu-driver-one-time) first. Then, in PowerShell:

```powershell
irm https://raw.githubusercontent.com/jia1217/vigenflow/main/install.ps1 | iex
```

The installer checks the NPU driver, downloads the latest release, asks for an install folder and which models to download, and starts VigenFlow and OpenWebUI in their own windows. OpenWebUI runs with [uv](https://github.com/astral-sh/uv), so you need neither Docker nor Python; its first start downloads about 2 GB and takes several minutes, then it opens in your browser. If Windows Firewall asks about `vgf-serve.exe`, either answer works for use on this PC. After a restart, start both again from the Start menu, under VigenFlow.

## 🐧 Ubuntu (Docker or Podman)

```bash
curl -fsSL https://raw.githubusercontent.com/jia1217/vigenflow/main/docker/install.sh | bash
```

The installer checks for the NPU driver and offers to install it, installs Docker if you have neither Docker nor Podman, asks which models to download, and starts VigenFlow together with OpenWebUI. When it finishes, open the address it prints, usually http://localhost:8080. The [Docker and Podman guide](docker.md) shows the same steps by hand, with Podman and Docker Compose.

To use the release packages instead, follow the [manual installation guide](installation.md).
