"""
Template definitions for generating disease explanations.
Provides disease-specific templates with environmental context.
"""

from typing import Dict, List


# Disease-specific visual symptom descriptions
VISUAL_SYMPTOMS = {
    "Healthy": [
        "normal green coloration and no visible disease symptoms",
        "uniform leaf structure with no lesions or discoloration",
        "healthy appearance with intact leaf margins",
        "vibrant green color without spots or blemishes",
    ],
    "Alternaria": [
        "brown lesions with concentric rings, typically starting at leaf margins",
        "target-spot patterns with dark brown centers",
        "necrotic spots with distinct concentric zones",
        "brown to black lesions showing bull's-eye pattern",
    ],
    "Stemphylium": [
        "dark brown to black irregular spots on the leaf surface",
        "scattered necrotic lesions with irregular margins",
        "brown blotches that may coalesce into larger necrotic areas",
        "irregular dark spots with potential leaf curling",
    ],
    "Marssonina": [
        "small brown spots with purple halos",
        "reddish-brown lesions with distinct purple borders",
        "numerous small spots that may merge over time",
        "characteristic purple-edged lesions on leaf surface",
    ]
}

# Environmental analysis templates based on sensor conditions
ENVIRONMENTAL_ANALYSIS = {
    "Healthy": {
        "optimal": "Environmental conditions are optimal: temperature {temp:.1f}°C, humidity {humidity:.1f}%, and soil moisture {moisture:.1f}% support healthy plant growth.",
        "suboptimal": "Environmental conditions are acceptable: temperature {temp:.1f}°C, humidity {humidity:.1f}%, soil moisture {moisture:.1f}%. Monitor for potential stress factors.",
    },
    "Alternaria": {
        "high_risk": "High humidity ({humidity:.1f}%) combined with moderate temperature ({temp:.1f}°C) creates favorable conditions for Alternaria fungal pathogen. Soil moisture at {moisture:.1f}%.",
        "moderate_risk": "Temperature {temp:.1f}°C and humidity {humidity:.1f}% present moderate risk for Alternaria development. Current soil moisture: {moisture:.1f}%.",
        "favorable": "Environmental factors favor fungal growth: temperature {temp:.1f}°C, high humidity {humidity:.1f}%, and soil moisture {moisture:.1f}% support pathogen development.",
    },
    "Stemphylium": {
        "high_risk": "Very high humidity ({humidity:.1f}%) at temperature {temp:.1f}°C provides optimal conditions for Stemphylium leaf blight. Soil moisture: {moisture:.1f}%.",
        "moderate_risk": "Current conditions (temperature {temp:.1f}°C, humidity {humidity:.1f}%) are moderately conducive to Stemphylium infection. Soil moisture at {moisture:.1f}%.",
        "favorable": "Cool to moderate temperature ({temp:.1f}°C) with elevated humidity ({humidity:.1f}%) favors this fungal pathogen. Soil moisture: {moisture:.1f}%.",
    },
    "Marssonina": {
        "high_risk": "Cool temperatures ({temp:.1f}°C) with high humidity ({humidity:.1f}%) create ideal conditions for Marssonina blotch. Soil moisture: {moisture:.1f}%.",
        "moderate_risk": "Environmental conditions (temperature {temp:.1f}°C, humidity {humidity:.1f}%) present moderate infection risk. Soil moisture at {moisture:.1f}%.",
        "favorable": "Moderate temperatures ({temp:.1f}°C) and humidity levels ({humidity:.1f}%) favor pathogen activity. Current soil moisture: {moisture:.1f}%.",
    }
}

# Complete template patterns
TEMPLATE_PATTERNS = {
    "Healthy": [
        "The leaf appears healthy with {visual}. {environmental}",
        "No disease detected. Leaf shows {visual}. {environmental}",
        "Healthy leaf condition observed. {visual}. {environmental}",
    ],
    "Alternaria": [
        "Alternaria leaf spot detected. The leaf shows {visual}. {environmental}",
        "Alternaria alternata infection identified. {visual} are visible. {environmental}",
        "Early signs of Alternaria blight. Leaf exhibits {visual}. {environmental}",
    ],
    "Stemphylium": [
        "Stemphylium leaf blight identified. {visual} visible on the leaf surface. {environmental}",
        "Stemphylium vesicarium infection detected. The leaf shows {visual}. {environmental}",
        "Stemphylium disease present. {visual} observed. {environmental}",
    ],
    "Marssonina": [
        "Marssonina blotch detected. {visual} are characteristic of this disease. {environmental}",
        "Marssonina coronaria infection identified. Leaf exhibits {visual}. {environmental}",
        "Fabraea leaf spot (Marssonina) observed. {visual} present on leaf. {environmental}",
    ]
}

# Recommended actions based on disease and severity
RECOMMENDATIONS = {
    "Healthy": [
        "Continue current management practices.",
        "Monitor regularly for early disease detection.",
        "Maintain optimal growing conditions.",
    ],
    "Alternaria": [
        "Apply appropriate fungicide and improve air circulation.",
        "Remove infected leaves and reduce overhead irrigation.",
        "Consider copper-based fungicide application.",
    ],
    "Stemphylium": [
        "Implement fungicide treatment and reduce humidity.",
        "Prune affected areas and improve ventilation.",
        "Apply protective fungicides during susceptible periods.",
    ],
    "Marssonina": [
        "Apply fungicide and remove fallen leaves to reduce inoculum.",
        "Improve drainage and reduce leaf wetness duration.",
        "Consider resistant varieties for future plantings.",
    ]
}


def get_environmental_risk_level(disease: str, temp: float, humidity: float, moisture: float) -> str:
    """
    Determine environmental risk level based on sensor values.

    Args:
        disease: Disease name
        temp: Temperature in Celsius
        humidity: Humidity percentage
        moisture: Soil moisture percentage

    Returns:
        Risk level string ('high_risk', 'moderate_risk', 'favorable', 'optimal', 'suboptimal')
    """
    if disease == "Healthy":
        # Optimal ranges for healthy plants
        if 20 <= temp <= 25 and 60 <= humidity <= 70 and 40 <= moisture <= 60:
            return "optimal"
        else:
            return "suboptimal"

    elif disease == "Alternaria":
        # Alternaria favors moderate temp (22-28°C) and high humidity (75-90%)
        if 22 <= temp <= 28 and humidity >= 80:
            return "high_risk"
        elif 20 <= temp <= 30 and humidity >= 70:
            return "moderate_risk"
        else:
            return "favorable"

    elif disease == "Stemphylium":
        # Stemphylium favors cool-moderate temp (18-24°C) and very high humidity (80-95%)
        if 18 <= temp <= 24 and humidity >= 85:
            return "high_risk"
        elif 15 <= temp <= 26 and humidity >= 75:
            return "moderate_risk"
        else:
            return "favorable"

    elif disease == "Marssonina":
        # Marssonina favors cool temp (15-22°C) and high humidity (70-85%)
        if 15 <= temp <= 22 and humidity >= 75:
            return "high_risk"
        elif 13 <= temp <= 24 and humidity >= 65:
            return "moderate_risk"
        else:
            return "favorable"

    return "moderate_risk"  # Default


def get_template_components(disease: str) -> Dict[str, List[str]]:
    """
    Get all template components for a disease.

    Args:
        disease: Disease name

    Returns:
        Dictionary with visual symptoms, environmental analysis, patterns, and recommendations
    """
    return {
        "visual_symptoms": VISUAL_SYMPTOMS.get(disease, []),
        "environmental_analysis": ENVIRONMENTAL_ANALYSIS.get(disease, {}),
        "patterns": TEMPLATE_PATTERNS.get(disease, []),
        "recommendations": RECOMMENDATIONS.get(disease, [])
    }


def get_all_diseases() -> List[str]:
    """Get list of all disease names."""
    return list(TEMPLATE_PATTERNS.keys())


if __name__ == "__main__":
    # Test template definitions
    print("Testing template definitions...")
    print("=" * 80)

    diseases = get_all_diseases()
    print(f"\nSupported diseases: {diseases}")

    # Test for each disease
    for disease in diseases:
        print(f"\n{disease}:")
        print(f"  Visual symptoms: {len(VISUAL_SYMPTOMS[disease])} variations")
        print(f"  Environmental templates: {len(ENVIRONMENTAL_ANALYSIS[disease])} risk levels")
        print(f"  Pattern templates: {len(TEMPLATE_PATTERNS[disease])} variations")
        print(f"  Recommendations: {len(RECOMMENDATIONS[disease])} options")

    # Test risk level calculation
    print("\n" + "=" * 80)
    print("Testing risk level calculation:")

    test_cases = [
        ("Alternaria", 25, 85, 40),
        ("Stemphylium", 20, 90, 50),
        ("Marssonina", 18, 80, 55),
        ("Healthy", 22, 65, 50),
    ]

    for disease, temp, humidity, moisture in test_cases:
        risk = get_environmental_risk_level(disease, temp, humidity, moisture)
        print(f"  {disease}: temp={temp}°C, humidity={humidity}%, moisture={moisture}% -> {risk}")
