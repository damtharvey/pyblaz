from setuptools import setup, find_packages

setup(
    name="pyblaz",
    version="0.0.1",
    description="Arbitrary-dimensional, floating-point array compressor that supports transformations of compressed arrays.",
    url="https://github.com/damtharvey/pyblaz",
    author="Harvey Dam",
    author_email="",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "dev": [
            "matplotlib",
            "numpy",
            "tqdm",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
