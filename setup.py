from setuptools import setup, find_packages

setup(
    name="CommML",  
    version="0.1.0",
    author="TheCSGuy25",
    author_email="pandz1802@gmail.com",
    description="Common ML utilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CommML",
    packages=find_packages(),  
    install_requires=[
        "numpy",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
