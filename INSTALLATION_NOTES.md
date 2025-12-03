# Installation & Troubleshooting Guide — CARLA + DQN Project

This document explains how to set up the environment, install CARLA 0.9.15, Python packages, and solve common errors for the Deep Reinforcement Learning for Autonomous Vehicle Safety project.

---

## 1️⃣ System Requirements

- **OS:** 13th Gen Intel(R) Core™ i9-1390HX 2.20 GHz.
- **Python:** 3.7.16
- **CARLA:** 0.9.15
- **GPU:** NVIDIA Geforce RTX 4090 Laptop GPU
- **RAM:** 32 Go 
- **Libraries:** TensorFlow 2.10.1, Keras 2.10.0, OpenCV 4.10.0, Pygame 1.21.6, tqdm 4.67.1
- **Optional Tools:** Jupyter Notebook, Matplotlib

---


## 2️⃣ Install CARLA

Download CARLA 0.9.15:

[https://github.com/carla-simulator/carla/releases/tag/0.9.15](https://github.com/carla-simulator/carla/releases/tag/0.9.15)

---

### 3️⃣ Install DirectX End-User Runtime

[https://www.microsoft.com/en-us/download/details.aspx?id=35](https://www.microsoft.com/en-us/download/details.aspx?id=35)

---

## 4️⃣ Solve Common CARLA Launch Error

**Error:** `out of video memory trying to allocate a rendering resource`

**Solution:**

1. Go to `CarlaUE4.exe` → Right-click → **Create Shortcut** → place it on Desktop.  
2. Right-click shortcut → **Properties** → Add `-dx11` at the end of the target:

