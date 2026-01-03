from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if fh else ""

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-leaf-disease",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multi-modal leaf disease detection system combining images and environmental sensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MultiModalLeafDiseaseDetection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-train=scripts.train:main",
            "multimodal-evaluate=scripts.evaluate:main",
            "multimodal-inference=scripts.inference:main",
        ],
    },
)
