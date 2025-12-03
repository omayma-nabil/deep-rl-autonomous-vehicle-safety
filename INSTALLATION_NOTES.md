# Installation & Troubleshooting Guide — CARLA + DQN Project

This document explains how to set up the environment, install CARLA 0.9.15, Python packages, and solve common errors for the Deep Reinforcement Learning for Autonomous Vehicle Safety project.

---

## 1️⃣ System Requirements
# Deep Reinforcement Learning for Autonomous Vehicle Safety

This repository contains the implementation of the Master’s Thesis project on **Deep Reinforcement Learning (DQN)** for autonomous vehicle safety using the CARLA simulator.

---

## 1. System Requirements

- **OS:** Windows 10 Professionnel N
- **CPU:** 13th Gen Intel(R) Core™ i9-1390HX 2.20 GHz
- **GPU:** NVIDIA GeForce RTX 4090 Laptop GPU
- **RAM:** 32 GB
- **Python:** 3.7.16
- **CARLA:** 0.9.15
- **Libraries:** 
  - TensorFlow 2.10.1
  - Keras 2.10.0
  - OpenCV 4.10.0
  - Pygame 1.21.6
  - tqdm 4.67.1



## 2.  CARLA Simulator Installation

> If you face issues with CARLA 0.9.15, you can also try version 0.9.14:  
> [CARLA 0.9.14 Release](https://github.com/carla-simulator/carla/releases/tag/0.9.14/)

1. Download CARLA from the official release page.  
2. Install **DirectX End-User Runtime Web Installer**:  
   [Download Link](https://www.microsoft.com/en-us/download/details.aspx?id=35)  

> ⚠️ Common error: `out of video memory trying to allocate a rendering resource`  
> **Solution:**  
> - Create a shortcut of `CarlaUE4.exe`  
> - Right-click → Properties → add `-dx11` at the end of Target path  
>   Example: `"C:\Path\To\Carla\CarlaUE4.exe" -dx11`



## 3. Conda Environment Setup

Open **Anaconda Prompt**:

```bash
# Check Conda version
conda --version
# Example output: conda 23.7.4

# Create virtual environment for CARLA
conda create --name carla-sim python=3.7

# Activate environment
conda activate carla-sim

## 4. How to Display Diagrams & Open Jupyter Notebook
**Activate your CARLA environment in Anaconda Prompt:**

```bash
conda activate carla-sim

Navigate to the CARLA PythonAPI examples folder:

```bash
cd C:\path\to\CarlaSimulator\PythonAPI\examples


Run TensorBoard to visualize training logs and diagrams:

```bash
tensorboard --logdir=logs/


Open the URL displayed in the terminal (usually http://localhost:6006) in your browser to see loss, rewards, and other metrics.

Launch Jupyter Notebook to run your scripts interactively:

```bash
jupyter notebook
