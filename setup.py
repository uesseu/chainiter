import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="chainiter",  # Replace with your own username
    version="0.0.1",
    author="ninja",
    author_email="sheepwing@kyudai.jp",
    description="My iterator object.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uesseu/chainiter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.7',
)
