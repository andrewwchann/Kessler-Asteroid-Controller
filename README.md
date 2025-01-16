# Kessler Asteroid Controller

## Overview

The **WhiteDove Controller** is a fuzzy logic-based agent developed for the Kessler Asteroids game. Its goal was to compete against a controller built by Professor Scott Dick, utilizing a combination of **fuzzy logic** and **genetic algorithms** for optimizing ship control strategies.

## Features

- **Fuzzy Logic**:
  - Implements a rule-based fuzzy system to handle dynamic scenarios in the asteroid field.
  - Controls key ship actions:
    - **Thrust**: Adjusts ship speed and direction based on asteroid proximity and velocity.
    - **Turn Rate**: Rotates the ship for precise targeting.
    - **Fire**: Determines optimal firing angles and timing for bullets.
    - **Mine Deployment**: Decides when to drop mines to neutralize threats.

- **Genetic Algorithm Optimization**:
  - A genetic algorithm was used to fine-tune the fuzzy system's parameters.
  - The final optimized chromosome (`[7, 100, 153, 0, 162, 181]`) maximized the controller's efficiency, achieving:
    - **491 asteroids hit**
    - **Extended survival time**

## How It Works

1. **Fuzzy Control System**:
   - Inputs: 
     - Asteroid distance and speed.
     - Bullet travel time and angle adjustments.
   - Outputs:
     - Thrust power, turning rate, firing decision, and mine deployment.

2. **Genetic Algorithm**:
   - Evolved parameters for ship speed and thrust levels.
   - Evaluated performance in simulated asteroid fields to identify the best chromosome.

3. **Performance**:
   - The WhiteDove Controller demonstrated its performance in test scenarios, balancing aggressive asteroid elimination with survival-focused maneuvering.
