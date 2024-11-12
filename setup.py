from setuptools import setup, find_packages

setup(
    name='your_package_name',  # Required: The name of your package
    version='0.1.0',          # Required: The initial release version
    #author='Your Name',       # Optional: The author's name
    #author_email='your.email@example.com',  # Optional: The author's email
    description='A brief description of your package',  # Required: A short summary of what your package does
    #long_description=open('README.md').read(),  # Optional: A detailed description; typically from a README file
    #long_description_content_type='text/markdown',  # Optional: The format of the long description
    #url='Tobe',  # Optional: URL for the package's homepage
    packages=find_packages(),  # Automatically find packages in the current directory
    #classifiers=[  # Optional: A list of classifiers that provide metadata
    #    'Programming Language :: Python :: 3',
    #    'License :: OSI Approved :: MIT License',
    #    'Operating System :: OS Independent',
    #],
    #python_requires='>=3.6',  # Optional: Minimum Python version required
    install_requires=[  # Optional: List of dependencies required to run the package
        'numpy',  # Example: List your package dependencies
        'requests',
    ],
)
