# DigiGuide - Occupant Guiding System

## Project Overview

This repository provides an open-source implementation of DigiGuide, a system designed to optimize occupant comfort and energy efficiency within buildings by guiding occupants toward locations optimized for their individual comfort needs. This project accompanies our research paper, with additional details to be published upon paper acceptance.

## Comfort Configuration and Modeling

Before deployment, DigiGuide requires an initialization phase involving Digital Twin (DT) modeling for the building and the occupants. 
This can be divided into two parts: building modeling and occupant modeling. This is already demonstrated in the papers with examples. In the following, we extend the examples with more details.

### **Building Environmental Conditions Modeling**

Modeling DTs for the building is vital to create a realistic environment and predict future environmental conditions accurately. The building DT includes its static building model and dynamic environment. The static building information includes spatial layouts, climate type, and energy-related devices like HVAC system capabilities and their energy consumption. This project is built on [**Co-zyBench**](https://github.com/satrai-lab/cozybench), an open-source benchmarking tool for DT modeling and building management system evaluation, see more details in its correlated papers (Co-zyBench: Using Co-Simulation and Digital Twins to Benchmark Thermal Comfort Provision in Smart Buildings).

Buildings dynamic environment can be implemented in two ways: using real-time sensor observations and using simulations.
The environmental conditions are modeled in `spaces.py`, which contains methods to calculate and update different environmental parameters. For example in the Space class:
- `update_crowd_level()` models the indoor crowdedness level.
- `update_noise_level()` models the indoor noise level.

These methods are called by the `update_environment()` function to refresh the environmental conditions when generating guidance decisions.

To incorporate new environmental conditions, users need to create corresponding functions and add them to `update_environment()`.
For example, by utilizing real-time sensing from sensors, users can process sensor readings in these functions to provide real observations of each environmental condition. While for cases where users do not have devices for all locations and require using simulations to compensate, they can use mathematical models or simulators to estimate the corresponding conditions.

Here, we introduce two examples utilizing mathematical models for crowdedness and noise level estimations. These models are implemented in `update_crowd_level()` and `update_noise_level()`, respectively.

  - **Noise level:**
  - Measured using Sound Pressure Level (SPL, in dB).
  - Aggregated noise from multiple sources (static background noise and occupant-generated sounds) is calculated using:
    $$
    L_{\Sigma} = 10 \log_{10} \left( 10^{\frac{L_1}{10}} + 10^{\frac{L_2}{10}} + \dots + 10^{\frac{L_n}{10}} \right)
    $$
  - Sound level of each occupant is modeled in `occupant.py`.

- **Crowdedness level:**
  - Calculated based on the number of occupants relative to room capacity:
    $$
    crowdedness = \frac{\text{occupant number}}{\text{room capacity}}
    $$
  - Room capacity is estimated from room size and required area per occupant (e.g., office: 5 m²/person, meeting room: 2 m²/person).

Simulators can also help on modeling environmental conditions. users just need to add interfaces with other simulators in the same file `spaces.py` as described above. 

However, this project is built upon **pyEnergyPlus**, the thermal environment is modeled separately in `sim_ep.py`. If users have alternative approaches for modeling energy consumption of the HVAC system and indoor temperature, we recommend updating the `self.temperature` and `self.energy` parameters in `update_environment()`.

### **Occupant Dynamism Configurations**

Occupant DT modeling also contain two steps: static occupant profile and dynamic occupant behavior. The static profile decides how occupant would feel in different environment, such as whether they prefer colder environment or warmer. On the other hand, for the dynamic behavior, we need to model occupant comfort.

As introduced in the paper, we categorize occupant comfort preference and tolerance into several categories. For example, occupant thermal preferences can be either preferred cold, neutral or preferred warm. While crowdedness and noise level tolerance is categorized into 4 levels to represent different sensitivity. This categorization is introduced to follow the real-world condition where providing occupant preferred levels is more convenient and intuitive for them than providing exact preferred values such as their preferred temperature. This is modeled in the `occupants.py` file.


Occupant comfort level estimation, along with energy consumption, is defined as multi-objective optimization in `./H2H_guidance/multi_objectives.py`. For example, `eval_noise()` evaluates the acoustic comfort of occupants by counting ow much the current noise level (from `spaces.py`) in the place is higher than occupant oise tolerance (from `occupants.py`).


## System Requirements
- Python 3.x

## Setting Up the Conda Environment

To set up the required environment using Conda, follow these steps:

1. Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

2. Navigate to the project directory and create the environment using the provided `./env/environment.yml` file:
   ```bash
   cd ./env
   conda env create -f environment.yml
   ```

## Usage
Run the system with the following command:
```bash
python main.py [options]
```

### Command Line Options
- -f, --h2h: Enable/disable occupant guidance (default: True)
- -c, --comment: Add a comment to the output file name
- -a, --algorithm: Select envcironmental control approaches (majority/drift)

### Example
```bash
python main.py -f true -a majority -c "test_run"
```
