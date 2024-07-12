from setuptools import find_packages, setup

package_name = 'wheel_leg_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/model', ['model/actor.onnx']),
        ('share/' + package_name + '/model', ['model/encoder.onnx']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xinchen',
    maintainer_email='yao29@illinois.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wheel_leg_rl = wheel_leg_rl.wheel_leg_rl:main',
        ],
    },
)
