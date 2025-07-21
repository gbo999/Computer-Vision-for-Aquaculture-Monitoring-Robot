from setuptools import find_packages, setup
import os

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [line for line in requirements if line and not line.startswith("#")]

setup(
    name='counting-research-algorithms',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version='0.1.0',
    description='Automated counting and measurement algorithms for aquaculture monitoring',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Gil Ben Or',
    author_email='your.email@example.com',  # Replace with your email
    license='MIT',
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    keywords='computer vision, object detection, aquaculture, counting, measurement',
    project_urls={
        'Source': 'https://github.com/yourusername/counting_research_algorithms',
        'Bug Reports': 'https://github.com/yourusername/counting_research_algorithms/issues',
    },
    entry_points={
        'console_scripts': [
            'analyze-sizes=scripts.analyze_sizes:main',
            'filter-predictions=scripts.filter_predictions:main',
            'segment-molt=scripts.segment_molt:main',
        ],
    },
)
