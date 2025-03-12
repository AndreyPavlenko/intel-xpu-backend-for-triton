from setuptools import setup, find_packages

setup(
    name="pytest-select",
    version="0.2.1",
    description="A pytest plugin which allows to (de-)select tests from a file.",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Plugins",
        "Framework :: Pytest",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pytest",
    ],
    extras_require={"dev": [
        "black",
        "coverage",
        "flake8",
        "flake8-bugbear",
        "mutmut",
        "bump2version",
    ]},
    entry_points={"pytest11": [
        "pytest-select = pytest_select.plugin",
    ]},
    keywords=["pytest", "test", "plugin"],
)
