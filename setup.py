from setuptools import Command, find_packages, setup

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = "STALOC",
    version = "1.0.0",
    description = "Spatial transcriptomics-assisted localization for single-cell transcriptomics with STALOC",
    url = "https://github.com/Zhanglabtools/STALOC",
    author = "Shang Li",
    author_email = "lishang@amss.ac.cn",
    license = "MIT",
    packages = ['STALOC'],
    install_requires = ["requests",],
    zip_safe = False,
    include_package_data = True,
    long_description = __long_description__
)
