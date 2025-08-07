from setuptools import setup, find_packages

setup(
    name='quest',
    version='0.1.0',
    description='A package for WebRTC headset utilities and transformations',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),  # 自动查找包含 __init__.py 的包
    include_package_data=True,
    install_requires=[
        'numpy>=1.18.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
