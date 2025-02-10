import setuptools

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='genomeocean',
    version='0.5.0',
    author='Zhong Wang',
    author_email='zhongwang@lbl.gov',
    description='A Python library for GenomeOcean inference and fine-tuning.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jgi-genomeocean/genomeocean",
    packages=setuptools.find_packages(),
    scripts=[
        "go_generate.py"
    ],
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3.8', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10', 
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
