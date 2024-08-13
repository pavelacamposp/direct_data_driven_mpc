import setuptools

setuptools.setup(
    name="direct_data_driven_mpc",
    version='1.0',
    description=("A Python implementation of a Nominal and Robust Direct"
                 "Data-Driven MPC Control system proposed by "),
    author='Pável A. Campos-Peña',
    author_email='pcamposp@uni.pe',
    packages=setuptools.find_packages(include=["direct_data_driven_mpc*",
                                               "models*"]),
    install_requires=['numpy', 'matplotlib', 'cvxpy'],
    license='MIT'
)