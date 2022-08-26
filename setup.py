
from pathlib import Path
from setuptools import setup

description = ['Tool for citywide mapping']

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()
with open(str(root / 'requirements.txt'), 'r') as f:
    dependencies = f.read().split('\n')

setup(
    name='cityslam',
    version='0.1',
    packages=['cityslam'],
    python_requires='>=3.7',
    install_requires=dependencies,
    authors=['Forgaard, T.', 'Heine, T.', 'Kalananthan, S.', 'Steinsland, K.'],
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/tforgaard/zuricityslam/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)