from setuptools import find_packages,setup

setup(
    name='Gemstone Price Prediction',
    version='1.1.1',
    author='HailHydra',
    author_email='darklususnaturae@gmail.com',
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'seaborn',
        'flask',
        'mlflow',
        'ipykernel',
        'xgboost',
    ],
    packages=find_packages()
)