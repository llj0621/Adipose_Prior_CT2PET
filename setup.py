from setuptools import setup, find_packages

setup(
    name="adipose-ct2pet",
    version="0.1.0",
    description="Two-stage CT-to-PET synthesis with adipose-aware refinement",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "nibabel>=4.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-image>=0.20",
        "tqdm>=4.65",
    ],
)
