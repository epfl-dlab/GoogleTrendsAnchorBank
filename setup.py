from setuptools import setup

setup(name='gtab',
      version='0.7',
      author="EPFL DLAB",
      author_email="epfl.dlab@gmail.com",
      description='gtab (Google Trends Anchor Bank) allows users to obtain precisely calibrated time series of search interest from Google Trends.',
      long_description='For a project description see https://github.com/epfl-dlab/GoogleTrendsAnchorBank/.',
      url='https://github.com/epfl-dlab/GoogleTrendsAnchorBank',
      packages=['gtab'],
      include_package_data=True,
      install_requires=[
          'networkx',
          'pytrends',
          'tqdm',
          'pandas',
          'numpy',
      ],
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
