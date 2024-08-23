from setuptools import setup, find_packages

setup(
    name="q-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "networkx",
        "scikit-learn"
    ],
    author="Nikita Smirnov",
    # author_email="your.email@example.com",
    description="A package for Q-analysis of complex networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pakrentos/q-analysis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
