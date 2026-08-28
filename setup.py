from setuptools import find_packages, setup

setup(
    name="cdm",
    version="0.1",
    packages=find_packages(include=["cdm", "cdm.*"]),
    zip_safe=False,
)
