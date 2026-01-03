"""
Template engine for generating disease explanations.
Combines visual symptoms with environmental sensor data to create coherent explanations.
"""

import random
from typing import Dict, List, Optional
import numpy as np

from .template_definitions import (
    VISUAL_SYMPTOMS,
    ENVIRONMENTAL_ANALYSIS,
    TEMPLATE_PATTERNS,
    RECOMMENDATIONS,
    get_environmental_risk_level,
    get_template_components
)


class TemplateEngine:
    """Engine for generating disease explanations from templates."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize template engine.

        Args:
            seed: Random seed for reproducible template selection
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def generate_explanation(
        self,
        disease: str,
        temperature: float,
        humidity: float,
        soil_moisture: float,
        include_recommendation: bool = False,
        pattern_idx: Optional[int] = None,
        visual_idx: Optional[int] = None
    ) -> str:
        """
        Generate a complete explanation for a disease detection.

        Args:
            disease: Disease name ('Healthy', 'Alternaria', 'Stemphylium', 'Marssonina')
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            soil_moisture: Soil moisture percentage
            include_recommendation: Whether to append management recommendation
            pattern_idx: Specific pattern index to use (None for random)
            visual_idx: Specific visual symptom index to use (None for random)

        Returns:
            Complete explanation string
        """
        # Get template components
        components = get_template_components(disease)

        if not components['patterns']:
            raise ValueError(f"No templates defined for disease: {disease}")

        # Select visual symptom description
        if visual_idx is not None:
            visual = components['visual_symptoms'][visual_idx % len(components['visual_symptoms'])]
        else:
            visual = random.choice(components['visual_symptoms'])

        # Determine environmental risk level
        risk_level = get_environmental_risk_level(disease, temperature, humidity, soil_moisture)

        # Get environmental analysis template
        env_template = components['environmental_analysis'].get(risk_level)
        if env_template is None:
            # Fallback to first available risk level
            env_template = list(components['environmental_analysis'].values())[0]

        # Format environmental analysis with sensor values
        environmental = env_template.format(
            temp=temperature,
            humidity=humidity,
            moisture=soil_moisture
        )

        # Select main pattern template
        if pattern_idx is not None:
            pattern = components['patterns'][pattern_idx % len(components['patterns'])]
        else:
            pattern = random.choice(components['patterns'])

        # Generate main explanation
        explanation = pattern.format(
            visual=visual,
            environmental=environmental
        )

        # Optionally add recommendation
        if include_recommendation and components['recommendations']:
            recommendation = random.choice(components['recommendations'])
            explanation += f" {recommendation}"

        return explanation

    def generate_batch_explanations(
        self,
        diseases: List[str],
        temperatures: np.ndarray,
        humidities: np.ndarray,
        soil_moistures: np.ndarray,
        include_recommendation: bool = False
    ) -> List[str]:
        """
        Generate explanations for a batch of samples.

        Args:
            diseases: List of disease names
            temperatures: Array of temperatures
            humidities: Array of humidity values
            soil_moistures: Array of soil moisture values
            include_recommendation: Whether to include recommendations

        Returns:
            List of explanation strings
        """
        explanations = []

        for disease, temp, humidity, moisture in zip(diseases, temperatures, humidities, soil_moistures):
            explanation = self.generate_explanation(
                disease=disease,
                temperature=float(temp),
                humidity=float(humidity),
                soil_moisture=float(moisture),
                include_recommendation=include_recommendation
            )
            explanations.append(explanation)

        return explanations

    def get_all_possible_explanations(self, disease: str, num_sensor_samples: int = 10) -> List[str]:
        """
        Generate all possible explanation variations for a disease.
        Useful for vocabulary building.

        Args:
            disease: Disease name
            num_sensor_samples: Number of different sensor value combinations to try

        Returns:
            List of all possible explanations
        """
        components = get_template_components(disease)
        explanations = []

        # Define sensor value ranges based on disease
        sensor_ranges = {
            "Healthy": (20, 25, 60, 70, 40, 60),
            "Alternaria": (22, 28, 75, 90, 30, 50),
            "Stemphylium": (18, 24, 80, 95, 40, 60),
            "Marssonina": (15, 22, 70, 85, 50, 70),
        }

        temp_min, temp_max, hum_min, hum_max, moist_min, moist_max = sensor_ranges.get(
            disease, (15, 30, 50, 90, 30, 70)
        )

        # Generate samples across the range
        for _ in range(num_sensor_samples):
            temp = random.uniform(temp_min, temp_max)
            humidity = random.uniform(hum_min, hum_max)
            moisture = random.uniform(moist_min, moist_max)

            # Try all pattern and visual combinations
            for pattern_idx in range(len(components['patterns'])):
                for visual_idx in range(len(components['visual_symptoms'])):
                    explanation = self.generate_explanation(
                        disease=disease,
                        temperature=temp,
                        humidity=humidity,
                        soil_moisture=moisture,
                        include_recommendation=False,
                        pattern_idx=pattern_idx,
                        visual_idx=visual_idx
                    )
                    explanations.append(explanation)

        return list(set(explanations))  # Remove duplicates

    def get_template_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about available templates.

        Returns:
            Dictionary with template counts per disease
        """
        from .template_definitions import get_all_diseases

        stats = {}

        for disease in get_all_diseases():
            components = get_template_components(disease)
            stats[disease] = {
                'visual_symptoms': len(components['visual_symptoms']),
                'environmental_levels': len(components['environmental_analysis']),
                'patterns': len(components['patterns']),
                'recommendations': len(components['recommendations']),
                'total_combinations': (
                    len(components['visual_symptoms']) *
                    len(components['environmental_analysis']) *
                    len(components['patterns'])
                )
            }

        return stats


def create_template_engine(seed: Optional[int] = None) -> TemplateEngine:
    """
    Factory function to create a template engine.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Configured TemplateEngine instance
    """
    return TemplateEngine(seed=seed)


if __name__ == "__main__":
    # Test template engine
    print("Testing Template Engine...")
    print("=" * 80)

    # Create engine with seed for reproducibility
    engine = TemplateEngine(seed=42)

    # Test single explanation generation
    print("\nTest 1: Single explanation generation")
    print("-" * 80)

    test_cases = [
        ("Healthy", 22.5, 65.0, 50.0),
        ("Alternaria", 25.0, 85.0, 40.0),
        ("Stemphylium", 20.0, 90.0, 50.0),
        ("Marssonina", 18.0, 78.0, 55.0),
    ]

    for disease, temp, humidity, moisture in test_cases:
        explanation = engine.generate_explanation(disease, temp, humidity, moisture)
        print(f"\n{disease}:")
        print(f"Sensors: T={temp}°C, H={humidity}%, M={moisture}%")
        print(f"Explanation: {explanation}")

    # Test batch generation
    print("\n\nTest 2: Batch explanation generation")
    print("-" * 80)

    diseases = ["Healthy", "Alternaria", "Stemphylium", "Marssonina"]
    temps = np.array([22.5, 25.0, 20.0, 18.0])
    humidities = np.array([65.0, 85.0, 90.0, 78.0])
    moistures = np.array([50.0, 40.0, 50.0, 55.0])

    batch_explanations = engine.generate_batch_explanations(
        diseases, temps, humidities, moistures
    )

    print(f"Generated {len(batch_explanations)} explanations")

    # Test template statistics
    print("\n\nTest 3: Template statistics")
    print("-" * 80)

    stats = engine.get_template_statistics()
    for disease, disease_stats in stats.items():
        print(f"\n{disease}:")
        for key, value in disease_stats.items():
            print(f"  {key}: {value}")

    # Test all possible explanations for vocabulary building
    print("\n\nTest 4: All possible explanations (for vocabulary)")
    print("-" * 80)

    all_explanations = engine.get_all_possible_explanations("Alternaria", num_sensor_samples=3)
    print(f"Generated {len(all_explanations)} unique explanations for Alternaria")
    print(f"Sample: {all_explanations[0]}")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
