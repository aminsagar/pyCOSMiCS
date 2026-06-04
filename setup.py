from setuptools import setup, find_packages

setup(
    name="cosmics",
    version="1.0.0",
    description="COSMiCS — Chemometric decomposition Of SAXS data (Python port)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8",
        "matplotlib>=3.5",
    ],
    entry_points={
        "console_scripts": [
            "cosmics=cosmics.main:main",
        ],
    },
)
