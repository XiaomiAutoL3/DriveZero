import os

import setuptools

# Change directory to allow installation from anywhere.
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Keep one dependency source for editable installs, wheels, and the documented
# no-deps workflow.
requirements_path = os.path.join(script_folder, 'requirements.txt')
with open(requirements_path, encoding='utf-8') as requirements_file:
    runtime_requirements = [
        line.strip()
        for line in requirements_file
        if line.strip() and not line.lstrip().startswith('#')
    ]

setuptools.setup(
    name='nuplan-devkit',
    version='1.2.2',
    author='The nuPlan team @ Motional',
    author_email='nuscenes@motional.com',
    description='Public nuPlan runtime companion for DriveRL closed-loop inference.',
    url='https://github.com/motional/nuplan-devkit',
    python_requires='>=3.11,<3.12',
    install_requires=runtime_requirements,
    packages=setuptools.find_packages(script_folder),
    # Hydra discovers the simulation/experiment configuration through the
    # installed ``nuplan`` package.  Keep the non-Python runtime resources in
    # wheels as well as in source checkouts; without these globs an installed
    # companion silently loses all YAML configs and NuBoard templates.
    package_data={
        '': [
            '**/*.css',
            '**/*.html',
            '**/*.js',
            '**/*.png',
            '**/*.yaml',
        ]
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
    ],
    entry_points={"console_scripts": ["nuplan_cli = nuplan.cli.nuplan_cli:main"]},
    license='apache-2.0',
)
