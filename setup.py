# setup.py
from setuptools import setup, find_packages

setup(
    name="eeg_knn_bhho",
    version="0.1.0",
    description="CiSSA‐based EEG decomposition + KNN+B-HHO pipeline",
    author="Your Name",
    author_email="you@example.com",

    # Tell setuptools “look under src/ for our package code”
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "joblib",
        "typeguard",    # needed by test_feature_selection_toy.py
    ],

    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
