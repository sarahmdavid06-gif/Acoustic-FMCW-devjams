# Acoustic FMCW Radar — Blind-Spot Detection for Micro-Mobility

**A sub-$15 acoustic radar that gives e-bikes and scooters the same blind-spot awareness that $8,000 LiDAR systems provide on cars.**

Built for Smart India Hackathon (SIH) 2026, Smart Vehicles domain, by a team from Vellore Institute of Technology (VIT).

---

## The Problem

Two-wheeler riders account for 44.8% of India's road fatalities — over 172,000 deaths in 2023 alone. Riders currently rely on small vibrating mirrors to judge blind-spot traffic, which makes it nearly impossible to gauge how fast a vehicle in the rear-left or rear-right blind spot is actually closing in on them.

Cars solve this with Advanced Driver Assistance Systems (ADAS) — usually 77 GHz millimeter-wave (mmWave) radar or LiDAR. Those systems cost anywhere from $60 to over $8,000 per unit, which completely breaks the economics of a $1,000 e-scooter. Cheaper alternatives, like the common HC-SR04 ultrasonic sensor, only measure raw distance (no direct velocity) and have a 20–30 cm "ring-down" dead zone right next to the vehicle, since the same transducer has to stop vibrating before it can listen again.

## Our Solution

This project takes the mathematical principles behind expensive automotive radio-frequency (RF) radar and applies them to high-frequency sound instead.

- **Frequency-Modulated Continuous Wave (FMCW):** Rather than a single "beep," the sensor transmits a continuous, sweeping tone — a linear chirp — from 35 kHz to 45 kHz, inaudible to humans.
- **Full duplex transmit/receive:** A dedicated transmitter and a separate dedicated receiver run at the same time, which removes the ring-down dead zone entirely. The blind zone shrinks from 20–30 cm down to 1–5 cm.
- **Coherent processing:** Because the system knows exactly what frequency it's transmitting at any given microsecond, it can extract both the target's distance (Time of Flight) and its velocity (Doppler phase shift) from the same signal, at the same time.
- **Cost:** All of this comes in at under $15 in parts.

## How It Works

- **The chirp:** sweeps a 10 kHz bandwidth (35–45 kHz) over a 20 millisecond (ms) sweep time.
- **Range, via dechirping:** the received echo is multiplied against the reference chirp to produce a low-frequency "beat signal." Its frequency is directly proportional to target distance, giving a theoretical range resolution of about 1.72 cm.
- **Velocity, via phase shift:** as a target moves, the beat signal's phase shifts slightly between one chirp and the next. Tracking that phase shift across many chirps (called "slow time") gives instantaneous velocity, without ever having to differentiate distance readings (which is noisy).

## Hardware Architecture

**Transmit path:** ESP32-S3 → streams the chirp via Inter-IC Sound (I2S) Direct Memory Access (DMA) → MAX98357 I2S Digital-to-Analog Converter (DAC) and Class-D amplifier → 40T-16 piezoelectric transmitter.

**Receive path:** 40R-16 piezoelectric receiver → NE5532 preamplifier (single-supply, ~40 dB / 100x gain, biased to a 2.5V virtual ground) → ESP32-S3's built-in 12-bit Analog-to-Digital Converter (ADC), sampling at 96,000 samples per second (96 kSPS).

**Host:** the ESP32-S3 streams raw samples to a laptop over USB, where a Python script (NumPy/SciPy) runs the signal processing and renders a live dashboard.

## The Signal Processing Pipeline

1. **Windowing** — the raw signal is multiplied by a Hanning window to suppress FFT sidelobes, so loud nearby targets don't mask quieter distant ones.
2. **Fast-time FFT** — a Fast Fourier Transform run on each individual chirp; the resulting frequency peaks correspond to target distances.
3. **Range-bin matrix** — many consecutive chirps are stacked into a 2D matrix (fast-time vs. slow-time).
4. **Slow-time FFT** — a second FFT run across that matrix extracts Doppler velocity for every range bin.
5. **CA-CFAR thresholding** — Cell-Averaging Constant False Alarm Rate detection dynamically calculates the local noise floor, filtering out static clutter and wind noise so only genuinely moving targets trigger a warning.

## Repo Structure

- `src/main.cpp` — ESP32-S3 firmware: builds the chirp lookup table, streams it out over I2S, and captures synchronized receiver samples via the ADC continuous-mode driver.
- `src/radar_dsp.py` — desktop-side Python pipeline: reads the serial stream, dechirps it, runs the 2D FFT, and renders the live range-Doppler dashboard with a collision-warning readout.
- `platformio.ini` — PlatformIO build configuration, targeting the ESP32-S3 DevKitC-1 board.

## Getting Started

**Firmware:**
1. Install [PlatformIO](https://platformio.org/) (as a VS Code extension, or standalone).
2. Connect the ESP32-S3 DevKitC-1 over USB.
3. Open this folder in VS Code and use PlatformIO's Build, then Upload — or from the command line, `pio run --target upload`.
4. Check `platformio.ini` for the `upload_port` / `monitor_port` values and change them to match your machine's actual port.

**Signal processing dashboard:**
1. Install the Python dependencies: `pip install numpy scipy pyserial matplotlib`
2. With the ESP32-S3 connected and flashed, run: `python src/radar_dsp.py`
3. The script auto-detects the serial port and opens a live dashboard window.

## Bill of Materials

- ESP32-S3 DevKitC-1
- 40T-16 / 40R-16 open-type ultrasonic transducer pair
- MAX98357 I2S Class-D amplifier module
- NE5532 (or TL072) preamp module
- Passives (resistors, capacitors) and a breadboard for the preamp bias network

Full prototype cost: roughly ₹1,000–1,150 (about $15).


## Team

Varshith Reddy, Tanmay Yogesh Bharat, Atharva Kukreti, Sarah David
