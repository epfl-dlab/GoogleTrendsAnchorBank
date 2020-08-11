from setuptools import setup

setup(name = 'gtab',
      version = '0.1',
      description = 'Google Trends Anchor Banks (G-TAB) is a tool that allows users calibrate Google Trends data in order to enable further analyses.',
      url = 'https://github.com/epfl-dlab/GoogleTrendsAnchorBank',
      packages = ['gtab'],
      include_package_data = True
    #   entry_points = {
    #       'console_scripts': 'gtab-'
    #   }
    )