from setuptools import find_packages, setup


setup(
    name="hide-llava",
    version="0.1.0",
    description="HiDe-LLaVA local package scaffold",
    packages=find_packages(include=["HiDe", "HiDe.*", "llava", "llava.*"]),
    include_package_data=True,
)
