from setuptools import Command, find_packages, setup

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = "STALocator",
    version = "1.0.0",
    description = "Spatial Transcriptomics-Aided Localization for Single-Cell Transcriptomics with STALocator",
    url = "https://github.com/Zhanglabtools/STALocator",
    author = "Shang Li",
    author_email = "lishang@amss.ac.cn",
    license = "MIT",
    packages = ['STALocator'],
    install_requires = ["requests",],
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)
