from setuptools import setup, find_packages

def parse_requirements(path):
    with open(path) as f:
        lines = f.read().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

if __name__ == "__main__":
    setup(
        name="nrn-beta",
        version="0.1.1",
        description="A PyTorch framework for rapidly developing Neural Reasoning Networks.",
        author="Anonymous",
        python_requires=">=3.6",
        packages=find_packages(),
        package_dir={"": "."},
        install_requires=parse_requirements("requirements.txt"),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ],
        include_package_data=True,
    )
