from setuptools import setup, find_packages

def parse_requirements(path):
    with open(path) as f:
        lines = f.read().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

if __name__ == "__main__":
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()

    setup(
        name="nrn-beta",
        packages=find_packages('nrn-beta'),
        package_dir={'': 'nrn'},
        version="0.1.0",
        description="A PyTorch framework for rapidly developing Neural Reasoning Networks.",
        classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
        author="Anonymous",
        python_requires=">=3.6",
        install_requires="""
            aix360==0.3.0
            cffi==1.17.1
            clarabel==0.11.1
            colorama==0.4.6
            contourpy==1.3.2
            coverage==7.9.2
            cvxpy==1.6.6
            cycler==0.12.1
            filelock==3.18.0
            fonttools==4.58.5
            fsspec==2025.5.1
            iniconfig==2.1.0
            Jinja2==3.1.6
            joblib==1.5.1
            kiwisolver==1.4.8
            MarkupSafe==3.0.2
            matplotlib==3.10.3
            mpmath==1.3.0
            networkx==3.5
            numpy==2.3.1
            osqp==1.0.4
            packaging==25.0
            pandas==2.3.1
            pillow==11.3.0
            pluggy==1.6.0
            pycparser==2.22
            Pygments==2.19.2
            pyparsing==3.2.3
            pytest==8.4.1
            pytest-cov==6.2.1
            python-dateutil==2.9.0.post0
            pytorch_optimizer==3.6.1
            pytz==2025.2
            scikit-learn==1.7.0
            scipy==1.16.0
            scs==3.2.7.post2
            setuptools==80.9.0
            six==1.17.0
            sympy==1.14.0
            threadpoolctl==3.6.0
            torch==2.7.1
            torchvision==0.22.1
            typing_extensions==4.14.1
            tzdata==2025.2
            xgboost==3.0.2
        """,
    )
