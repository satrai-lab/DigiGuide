# DigiGuide - Occupant Guiding System

## Project Overview

This repository provides an open-source implementation of DigiGuide, a system designed to optimize occupant comfort and energy efficiency within buildings by guiding occupants toward spaces optimized for their individual comfort needs. This project accompanies our research paper, with additional details to be published upon paper acceptance.

## Comfort Configuration and Modeling

Before deployment, DigiGuide requires an initialization phase involving Digital Twin modeling and comfort configurations. This is demonstrated in the papers with examples. In the following, we extend the examples with more details.

### **Digital Twin (DT) Modeling**

As explained in the paper, modeling DTs for the building and its occupants is important to create a realistic environment and predict future conditions accurately. This can be implemented in two ways: using real-time sensor observations and using simulations.

The environmental conditions are modeled in `spaces.py`, which contains methods to calculate and update different environmental parameters. For example:
- `update_crowd_level()` models the crowdedness level.
- `update_noise_level()` models the indoor noise level.
- `update_environment()` calls these methods to refresh the environmental conditions before guidance decisions.

To incorporate new environmental conditions, users need to create corresponding functions and add them to `update_environment()`.

By utilizing real-time sensing from sensors, users can process sensor readings in these functions to provide real observations of each environmental condition.

For cases where users do not have sensors for all locations, they can use mathematical models to estimate or simulate the corresponding conditions.

Here, we introduce two examples utilizing mathematical models for crowdedness and noise level estimations. These models are implemented in `update_crowd_level()` and `update_noise_level()`, respectively.

<!-- - **Noise level:**
  - Measured using Sound Pressure Level (SPL, in dB).
  - Aggregated noise from multiple sources (static background noise and occupant-generated sounds) is calculated using:
    $$
    L_{\Sigma} = 10 \log_{10} \left( 10^{\frac{L_1}{10}} + 10^{\frac{L_2}{10}} + \dots + 10^{\frac{L_n}{10}} \right)
    $$

- **Crowdedness level:**
  - Calculated based on the number of occupants relative to room capacity:
    $$
    crowdedness = \frac{\text{occupant number}}{\text{room capacity}}
    $$
  - Room capacity is estimated from room size and required area per occupant (e.g., office: 5 m²/person, meeting room: 2 m²/person). -->

Simulators are also helpful for modeling environmental conditions. For example, we use **pyEnergyPlus** to simulate the thermal environment. This project is built upon **Co-zyBench**, an open-source benchmarking tool for DT modeling and building management system evaluation. The thermal environment is modeled in `sim_ep.py`. If users have alternative approaches for modeling energy consumption of the HVAC system and indoor temperature, we recommend updating the `self.temperature` and `self.energy` parameters in `spaces.py`.

### **Occupant Comfort Configurations**

The next step involves occupant comfort configuration, including the modeling of occupant comfort preferences and estimating their comfort levels.

As introduced in the paper, we categorize environmental conditions into several categories. This is also modeled in `spaces.py`, with functions like `update_crowd_level()`.

Occupant comfort level estimation, along with energy consumption, is defined as multi-objective optimization in `./H2H_guidance/multi_objectives.py`. For example:
- `eval_noise()` evaluates the acoustic comfort of occupants.

## Usage

Run the system with the following command:
```bash
python main.py [options]
