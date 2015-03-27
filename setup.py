from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='changepoint',
      version='0.1',
      description='Implement change point model in Python',
      url='https://github.com/LaDilettante/changepoint_model',
      author='Anh Le',
      author_email='anh.le91@gmail.com',
      license='MIT',
      packages=['changepoint'],
      install_requires=[
        'numpy',
        'scipy'
      ],
      zip_safe=False)
