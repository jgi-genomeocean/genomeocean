import setuptools

setuptools.setup(
    name='genomeocean',
    version='0.1.0',
    author='Zhong Wang',
    author_email='zhongwang@lbl.gov',
    description='A Python library for GenomeOcean inference and fine-tuning.',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jgi-genomeocean/genomeocean",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
