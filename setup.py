import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Rad-Age", # Replace with your own username
    version="0.0.1",
    author="Adrian Tasistro-Hart",
    description="This python package provides classes and functions for working with isotopic measurements of uranium, thorium, and lead to produce radiometric ages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarttiso/rad-age",
    packages=setuptools.find_packages(),
    license="GPL v3",
    install_requires=[]
)
