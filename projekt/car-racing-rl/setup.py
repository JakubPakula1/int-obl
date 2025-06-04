from setuptools import setup, find_packages

setup(
    name='car-racing-rl',
    version='0.1.0',
    author='Twoje Imię',
    author_email='twoj_email@example.com',
    description='Projekt wykorzystujący algorytmy uczenia maszynowego do sterowania samochodem w grze wyścigowej.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'gym',
        'numpy',
        'torch',
        'matplotlib',
        'pandas',
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            'car-racing-rl=main:main',
        ],
    },
)