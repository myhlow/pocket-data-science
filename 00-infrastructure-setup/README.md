# Track 0: Mobile Edge Infrastructure & Setup

This module contains the setup scripts, environment configurations, and bootstrapping workflows used by team **Pocket Data Lab** to execute competitive machine learning on Android smartphones.

---

## 📱 Hardware & OS Architecture

* **Host Environment:** Android 14 / 15 (ARM64 aarch64)
* **Terminal Shell:** Termux
* **Container Layer:** Termux PRoot Linux (Debian 12 / Ubuntu 24.04 LTS rootfs)
* **Agent Orchestrator:** Google Antigravity CLI (`agy`)
* **Hardware Constraints:** 8-core ARM64 CPU, 2–4 GB usable RAM envelope (strictly non-root)

---

## 🛠️ Bootstrapping Workflow

### 1. Install PRoot Linux & Core Compilers
```bash
pkg update && pkg upgrade -y
pkg install proot-distro git curl wget -y
proot-distro install debian
proot-distro login debian
```

### 2. Native Compilation Toolchain
Inside the PRoot container, install build dependencies for native C/C++ compilation:
```bash
apt update && apt install -y build-essential python3 python3-pip python3-dev \
    libopenblas-dev liblapack-dev gfortran libomp-dev cmake
```

### 3. Native Python Wheels
Build and install edge-optimized data science packages:
```bash
pip3 install --upgrade pip setuptools wheel
pip3 install numpy scipy pandas scikit-learn lightgbm catboost
```

### 4. Headless Kaggle API Configuration
Place your API key at `~/.kaggle/kaggle.json`:
```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
pip3 install kaggle
```

---

## 📖 Full Tutorial Reference
* [Run Google Antigravity CLI on Termux: Android PRoot Guide](https://malcolmlow.com/2026/09/18/run-google-antigravity-cli-termux-proot/)
* [Connect Codex CLI to WordPress from Android Termux](https://malcolmlow.com/2026/07/27/how-to-connect-codex-to-wordpress-com-and-post-from-termux-on-android/)
