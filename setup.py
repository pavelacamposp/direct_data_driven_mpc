import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="direct_data_driven_mpc",
    version='1.0',
    description=("A Python implementation of Nominal and Robust Direct "
                 "Data-Driven MPC Controllers proposed by Julian Berberich, "
                 "Johannes Köhler, Matthias A. Müller, and Frank Allgöwer."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Pável A. Campos-Peña',
    author_email='pcamposp@uni.pe',
    url='https://github.com/pavelacamposp/direct-data-driven-mpc',
    packages=setuptools.find_packages(include=["direct_data_driven_mpc*",
                                               "utilities*",
                                               "examples*"]),
    install_requires=['numpy',
                      'matplotlib>=3.9.0',
                      'cvxpy',
                      'tqdm',
                      'PyYAML',
                      'PyQt5'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    license='MIT'
)