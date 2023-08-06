import sys
from setuptools import setup, find_packages

def get_requirements():

    static_requirements = [
        'datasets>=2.12.0',
        'evaluate>=0.4.0',
        'torch>=2.0.0',
        'wrapt>=1.15.0',
        'pydantic>=1.10.7'
    ]

    if not any('adapters' in v for v in sys.argv):
        # add transformers requirements
        return static_requirements + [
            'transformers>=4.30.1',
            'accelerate>=0.20.3'
        ]

    return static_requirements

setup(
    name="hyped",
    version="0.0.5",
    description="A collection of data pipelines to ease the training of transformer models",
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    author="Niclas Doll",
    author_email="niclas@amazonis.net",
    url="https://github.com/ndoll1998/hyped/tree/master",
    packages=find_packages(exclude='tests'),
    package_dir={'hyped': 'hyped'},
    classifiers=[
        "License :: Freely Distributable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10"
    ],
    install_requires=get_requirements(),
    extras_require={
        'adapters': ['adapter-transformers>=3.2.1'],
        'peft': ['peft>=0.4.0']
    },
    entry_points={
        "console_scripts": [
            'hyped = hyped.cli:main'
        ]
    }
)
