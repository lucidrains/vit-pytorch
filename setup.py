from setuptools import setup, find_packages

setup(
  name = 'vit-pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.24.2',
  license='MIT',
  description = 'Vision Transformer (ViT) - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/vit-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image recognition'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6',
    'torchvision'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
