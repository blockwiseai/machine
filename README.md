<p align="center">
  <img src="static/zeus-icon.png" alt="Zeus Logo" width="150"/>
</p>
<h1 align="center">SN18: Zeus Environmental Forecasting Subnet<br><small>Ørpheus AI</small></h1>


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
<style>h1,h2,h3,h4 { border-bottom: 0; } </style>

Welcome to the Zeus Subnet! This repository contains all the necessary information to get started, understand our subnet architecture, and contribute.


### Quick Links
- [Incentive mechanism 🎁](docs/IncentiveMechanism.md)
- [Scoring system breakdown 📈](docs/DifficultyScoring.ipynb)
- [Mining Guide ⛏️](docs/Mining.md)
- [Validator Guide 🔧](docs/Validating.md)

> [!IMPORTANT]
> If you are new to Bittensor, we recommend familiarizing yourself with the basics on the [Bittensor Website](https://bittensor.com/) before proceeding.

---
# Predicting future environmental variables within a decentralized framework

### Overview
The Zeus subnet leverages advanced AI models within the Bittensor network to forecast environmental data at two distinct scales: Global and Hyperlocal.
Our platform is engineered to replace expensive, physics-based simulations with efficient, data-driven AI architectures. To achieve this, the subnet operates as a unified forecasting engine with two parallel challenges:
1. **Global (On-the-Grid):** Forecasting macro-scale variables using ERA5 reanalysis data from Copernicus (ECMWF). This is the largest global environmental dataset to date.
2. **Hyperlocal (Off-the-Grid):** Forecasting precise weather conditions at specific coordinates using real-time data from the WeatherXM network.

Validators stream data from these sources in real-time, querying miners on gigabytes of data across both modalities.

### Purpose
Traditionally, environmental forecasting relies on physics-based Numerical Weather Prediction (NWP). While accurate, NWP is highly cost-ineffective, requiring massive supercomputers and hours of simulation time for a single forecast.

Zeus incentives the development of AI-driven forecasting. These models are faster, cheaper, and increasingly more accurate than traditional methods. By operating distinct lanes for global and local data, we allow miners to specialize in different model architectures, solving the problem of planetary climate trends and street-level weather simultaneously.

### How it Works: Artificial Metagraph Separation
To maintain high performance across both data modalities, Zeus uses a system of distinct challenges. Miners cannot be generalists; they must specialize in one domain to compete effectively.

**1. Synapse Selection**

When running a miner, you must explicitly choose your path. You will send a synapse to the validators stating which challenge you are targeting: ERA5 (Global) or WeatherXM (Hyperlocal).

**2. Targeted Weights**

We enforce strict separation to ensure fair competition. You can switch preferences at any time, but you cannot accumulate scores in both lanes simultaneously.
 - Targeted Weights: You only compete against other miners in your chosen track.
 - If you choose the WeatherXM track, your score for the ERA5 track is mathematically forced to zero (and vice versa).

**3. Incentivized Slots & The Safe Zone**

The subnet supports **512 UIDs** to allow for robust competition without immediate deregistration pressure. The metagraph is divided into two sections:

- **Incentivized Slots (Top 256):**
    - 128 Slots are reserved for the top-performing ERA5 miners.
    - 128 Slots are reserved for the top-performing WeatherXM miners.
    - Only miners in these slots receive TAO emissions.

- **The Safe Zone (Bottom 256):**
    - If you rank below number 128 in your chosen mechanism (or are a newly registered miner), your weight is set to zero and you fall into the Safe Zone.
    - **Purpose:** This "buffer" eliminates immediate registration pressure. It gives new miners a safe space to deploy, test, and prove their performance.
    - **Deregistration:** Deregistration occurs from the bottom of the Safe Zone. Unless a new miner is strictly better than the existing top 128 performers in their lane, they will eventually be deregistered, protecting the established, high-quality miners in the incentivized slots.

### Core Components
- **Miners:** Tasked with running forecasting algorithms for either ERA5 (global) or WeatherXM (local) challenges. Miners must select a specific lane to specialize in.
- **Validators:** Responsible for challenging miners with subsets of environmental data, evaluating performance on held-out data, and enforcing the metagraph separation logic.
- **Resource Expansion:** We continuously add new environmental variables and data modalities to our subnet in order to evolve our subnet and solve a multitude of distinct problems.

---
### Community
For real-time discussions, community support, and regular updates, <a href="https://discord.com/invite/bittensor">join the bittensor discord</a>. Connect with developers, researchers, and users to get the most out of the Zeus Subnet.

### License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
