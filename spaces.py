import math

import numpy as np


class Space:
    def __init__(self, room_num, temperature, setpoint, capacity):

        self.room_num = room_num
        self.temperature = temperature
        self.energy = 0
        self.setpoint = setpoint
        self.capacity = capacity
        if self.capacity == 0:
            self.capacity = 1
        self.initial_noise = np.random.normal(40, 10)
        self.crowd = 0
        self.noise = 0
        self.people_num = 0
        self.type = None
        self.sound_pressure = self.initial_noise
        # Dictionary to store historical energy consumption data
        # Key: temperature difference (setpoint - outdoor temperature), limited to ≤ 0
        # Value: List of historical energy consumption values (max 10 entries per key)
        self.historical_energy_data = {}
        self.outdoor_temperature = 0  # Initialize outdoor temperature

    def update_environment(self, occupant_location, occupant_noise_level):
        self.update_crowd_level(occupant_location)
        self.update_noise_level(occupant_location, occupant_noise_level)

    def _count_people(self, occupant_location):
        occupant_count = 0
        for loc in occupant_location.values():
            if loc == self.room_num:
                occupant_count += 1
        self.people_num = occupant_count

    def update_crowd_level(self, occupant_location):
        self._count_people(occupant_location)
        crowd_ratio = (self.people_num / self.capacity)

        self.crowd = np.digitize(crowd_ratio, [0.25, 0.5, 0.75]) + 1
        return self.crowd

    def update_noise_level(self, occupant_location, occupant_noise_level):
        """Calculate noise level based on acoustic formula"""
        # Convert base noise to sound intensity
        total_intensity = 10 ** (self.initial_noise / 10)

        # Add noise contribution from each person
        for p_id, loc in occupant_location.items():
            if loc == self.room_num:
                total_intensity += 10 ** (occupant_noise_level[p_id] / 10)

        # Add noise contribution from each person
        # for person in range(self.people_num):
        #     # Ensure individual noise contribution is non-negative
        #     individual_noise = max(np.random.normal(30, 15), 0)
        #     total_intensity += 10 ** (individual_noise / 10)

        # Convert back to decibels (avoid division by zero)
        total_db = 10 * np.log10(total_intensity) if total_intensity > 0 else -np.inf

        # Classification thresholds (customizable)
        self.sound_pressure = total_db
        self.noise = np.digitize(total_db, [40, 55, 60]) + 1

        return self.noise

    def calculate_next_noise_level(self, num_people):
        """Calculate noise level based on acoustic formula"""
        # Convert base noise to sound intensity
        total_intensity = 10 ** (self.sound_pressure / 10)

        # Add noise contribution from each person
        for person in range(num_people):
            # Ensure individual noise contribution is non-negative
            individual_noise = max(np.random.normal(30, 0), 0)
            total_intensity += 10 ** (individual_noise / 10)

        # Convert back to decibels (avoid division by zero)
        total_db = 10 * np.log10(total_intensity) if total_intensity > 0 else -np.inf

        # Classification thresholds (customizable)
        return np.digitize(total_db, [40, 55, 60]) + 1

    def set_outdoor_temperature(self, outdoor_temp):
        """Set the current outdoor temperature"""
        self.outdoor_temperature = outdoor_temp
        
    def record_energy_consumption(self, energy_value):
        """
        Record actual energy consumption for the current temperature difference
        between setpoint and outdoor temperature.
        
        Args:
            energy_value: The actual energy consumption value to record
        """
        # Calculate temperature difference (setpoint - outdoor temperature)
        temp_diff = self.setpoint - self.outdoor_temperature
        
        # If setpoint > outdoor temperature, use difference, otherwise use 0
        diff_key = temp_diff if temp_diff > 0 else 0
        
        # Initialize list for this difference if it doesn't exist
        if diff_key not in self.historical_energy_data:
            self.historical_energy_data[diff_key] = []
            
        # Add the new energy consumption value
        self.historical_energy_data[diff_key].append(energy_value)
        
        # Keep only the 10 most recent values
        if len(self.historical_energy_data[diff_key]) > 10:
            self.historical_energy_data[diff_key].pop(0)
    
    def predict_energy_consumption(self, preferred_temp):
        """
        Predict energy consumption based on historical data for a given
        temperature preference.
        
        Args:
            preferred_temp: The preferred temperature setting
            
        Returns:
            Predicted energy consumption value
        """
        # Calculate temperature difference
        temp_diff = self.setpoint - preferred_temp
        
        # If no data is available or the key doesn't exist, return the simple difference
        if not self.historical_energy_data:
            return temp_diff if temp_diff > 0 else 0
            
        # Return the simple difference for penalty cases
        if self.setpoint == 30:  # Special case for empty spaces
            return temp_diff + 1 if temp_diff > 0 else 1
            
        # Try to find the closest key in the historical data
        closest_diff = min(self.historical_energy_data.keys(), 
                          key=lambda x: abs(x - temp_diff) if x > 0 else float('inf'))
                          
        # Check if there's sufficient data for this or similar temperature differences
        if closest_diff in self.historical_energy_data and len(self.historical_energy_data[closest_diff]) >= 2:
            # Use the average of historical values
            return sum(self.historical_energy_data[closest_diff]) / len(self.historical_energy_data[closest_diff])
        
        # Default to the simple difference if no historical data is available
        return temp_diff if temp_diff > 0 else 0

# class Space:
#     def __init__(self):
#         self.id = ''
#         self.setpoint = -1
#         self.temperature = -1
#         self.cap = -1
#         self.luminosity = -1
#
#     def getID(self):
#         return self.id
#
#     def setID(self, id):
#         self.id = id
#
#     def setCap(self,cap):
#         self.cap = cap
#
#     def getCap(self):
#         return self.cap
#
#     def setSetpoint(self, setpoint):
#         self.setpoint = setpoint
#
#     def getSetpoint(self):
#         return self.setpoint
#
#     def setTemperature(self, temp):
#         self.temperature = temp
#
#     def getTemperature(self):
#         return self.temperature
#
#     def setLuminosity(self, luminosity):
#         self.luminosity = luminosity
#
#     def getLuminosity(self):
#         return self.luminosity
