import os
import sys
from setuptools import setup
from setuptools.command.test import test


test_args = [
    '--verbose',
    '-W ignore',  # ignore warnings
    '--capture=sys',
    '--log-level=INFO',
    '--log-file-level=INFO',
    '--cov-report=html',
    '--cov-report=term',
    '--cov=dl',
    'test',
]


class PyTest(test):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        os.environ['ENV'] = 'test'
        test.initialize_options(self)
        self.pytest_args = test_args

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='dl',
    version='0.0.2',
    description='Example of a lean deep learning project with a config-driven approach.',
    long_description='https://github.com/kengz/lean-dl-example',
    keywords='dl',
    url='https://github.com/kengz/lean-dl-example',
    author='kengz',
    author_email='kengzwl@gmail.com',
    packages=['dl'],
    install_requires=[],
    zip_safe=False,
    include_package_data=True,
    dependency_links=[],
    extras_require={
        'dev': [],
        'docs': [],
        'testing': [],
    },
    classifiers=[],
    tests_require=[
        'autopep8',
        'flake8',
        'pytest',
        'pytest-cov',
    ],
    test_suite='test',
    cmdclass={'test': PyTest},
)
