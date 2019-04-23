from setuptools import setup

setup(name='deepframeinterpolation', 
      version='0.0.1', 
      description='Deep Frame Interpolation', 
      author='Arjun-Arora', 
      author_email='me@GitHub.com', 
      packages=['src','models','datasets', 'datasets/frames/train','datasets/frames/test', 'datasets/frames/val'], 
      zip_safe=False,
      include_package_data=True)
