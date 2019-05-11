from setuptools import setup

setup(
    name='League-of-Legends-Predict-Lane',
    version='0.01',
    url='https://github.com/Lkgsr/League-of-Legends-Predict-Lane',
    license='MIT',
    author='Lukas',
    author_email='',
    description='A CNN that predict\'s the lanes for leaug of legends ',
    packages=[''],
    package_dir={'': 'League-of-Legends-Predict-Lane'},
    install_requires=[
        'tensorflow-gpu',
        'keras',
        'scikit-learn',
        'cassiopeia'
    ]
)
