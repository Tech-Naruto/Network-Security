from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    try:
        requirements: List[str] = []

        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip().replace("\n", "") for req in requirements if req != "-e ." and req != "\n"]
    
        return requirements
    except Exception as e:
        raise e

setup(
    name="network_security",
    version="0.0.1",
    author="Krish Vardhan Pal",
    author_email="krishvardhan9369@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)
