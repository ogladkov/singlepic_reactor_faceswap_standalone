from setuptools import setup, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='reactor_standalone',
    version='0.1',
    packages=find_packages(),
    package_data={'reactor_standalone': [
        'components/*',
    ]},
    description='A standalone implementation of ReActor faceswap',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=required,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
