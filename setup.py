from setuptools import setup, find_packages

def parse_requirements(path):
    with open(path) as f:
        lines = f.read().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

if __name__ == "__main__":
    setup(
        name="centaur-nrn",
        version="0.1.2",
        description="A PyTorch framework for rapidly developing Neural Reasoning Networks.NRN module under Centaur organization",
        author="Centaur Team",
        author_email="team@centaur.org",
        url="https://github.com/centaur/nrn",
        project_urls={
        "Documentation": "https://docs.centaur.org/nrn",
        "Source": "https://github.com/centaur/nrn",
        "Issues": "https://github.com/centaur/nrn/issues"},
        packages=find_packages(),
        package_dir={"": "."},
        install_requires=parse_requirements("requirements.txt"),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ],
        include_package_data=True,
    )