from distutils.core import setup

requirements = ['numpy', 'scipy']
test_requirements = ['pytest', 'hypothesis']
bench_requirements = ['asv']

setup(
    name=['yarrow-diagrams'],
    version='0.1',
    description='Yarrow Diagrams',
    author='Paul Wilson',
    author_email='paul@statusfailed.com',
    url='yarrow.id',
    packages=['yarrow.diagrams'],
    install_requires=requirements,
    extras_require={
        'dev': test_requirements,
        'bench', bench_requirements,
    }
)

