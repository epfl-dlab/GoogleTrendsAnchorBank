from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gtab',
      version='0.1',
      author="EPFL DLAB",
      author_email="epfl.dlab@gmail.com",
      description='gtab allows users to obtain precise results from Google Trends queries.',
      url='https://github.com/epfl-dlab/GoogleTrendsAnchorBank',
      packages=['gtab'],
      include_package_data=True,
      entry_points={
          'console_scripts': ['gtab-init=gtab.command_line:init_dir',
                              'gtab-print-options=gtab.command_line:print_options',
                              'gtab-set-options=gtab.command_line:set_options',
                              'gtab-set-blacklist=gtab.command_line:set_blacklist',
                              'gtab-set-hitraffic=gtab.command_line:set_hitraffic',
                              'gtab-list=gtab.command_line:list_gtabs',
                              'gtab-rename=gtab.command_line:rename_gtab',
                              'gtab-delete=gtab.command_line:delete_gtab',
                              'gtab-set-active=gtab.command_line:set_active_gtab',
                              'gtab-create=gtab.command_line:create_gtab',
                              'gtab-query=gtab.command_line:new_query'
                              ]
      },
      classifiers=["License :: OSI Approved :: MIT License"],
      python_requires='>=3.6',
      )
