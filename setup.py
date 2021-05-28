import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_runstats",
    version="0.1.0",
    url="https://github.com/mir-group/pytorch_runstats",
    description="Running/online statistics for PyTorch ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    license_files="LICENSE",
    project_urls={
        "Bug Tracker": "https://github.com/mir-group/pytorch_runstats/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.6",
    install_requires=[],
    packages=["torch_runstats"],
)
