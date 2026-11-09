from setuptools import setup, find_packages

setup(
    name="jetbot",
    version="1.0.0",
    description="Corrected Jetbot library and autonomous visual docking pipeline for Yahboom YB-EJF01 carrier board",
    author="Azam Rafique",
    packages=find_packages(),
    install_requires=[
        "Adafruit-MotorHAT>=1.4.0",
        "traitlets>=5.0.0",
        "smbus2>=0.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)
