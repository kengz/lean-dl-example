from setuptools import setup


setup(
    name='dl',
    version='0.0.3',
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
)
