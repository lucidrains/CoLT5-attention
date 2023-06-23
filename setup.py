from setuptools import setup, find_packages

setup(
  name = 'CoLT5-attention',
  packages = find_packages(),
  version = '0.10.12',
  license='MIT',
  description = 'Conditionally Routed Attention',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/CoLT5-attention',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'dynamic routing'
  ],
  install_requires=[
    'einops>=0.6.1',
    'local-attention>=1.8.6',
    'packaging',
    'torch>=1.10'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
