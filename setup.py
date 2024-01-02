from setuptools import setup, find_packages

setup(
    name="hyped",
    version="0.1.0",
    description="",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    author="Niclas Doll",
    author_email="niclas@amazonis.net",
    url="https://github.com/ndoll1998/hyped/tree/master",
    packages=find_packages(exclude="tests"),
    package_dir={"hyped": "hyped"},
    classifiers=[
        "License :: Freely Distributable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=[
        "datasets>=2.16.1",
        "fsspec<=2023.9.2",
        "transformers>=4.36.2",
        "networkx>=3.1",
    ],
)
