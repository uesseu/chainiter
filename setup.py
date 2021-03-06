from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="chainiter",  # Replace with your own username
    author="ninja",
    author_email="sheepwing@kyudai.jp",
    description="Iterator which can use multicore, method chain, coroutine, and progress bar.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uesseu/chainiter",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.7',
)
