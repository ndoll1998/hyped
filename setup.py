from setuptools import setup, find_packages

setup(
    name="hyped",
    version="0.0.4",
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
    install_requires=[
        'adapter-transformers>=3.2.1',
        'datasets>=2.12.0',
        'evaluate>=0.4.0',
        'torch>=2.0.0',
        'pydantic>=1.10.7'
    ]
)
