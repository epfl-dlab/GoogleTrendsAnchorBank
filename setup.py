from setuptools import setup

setup(name='gtab',
      version='0.1',
      description='Google Trends Anchor Banks (G-TAB) is a tool that allows users calibrate Google Trends data in order to enable further analyses.',
      url='https://github.com/epfl-dlab/GoogleTrendsAnchorBank',
      packages=['gtab'],
      include_package_data=True,
      entry_points={
          'console_scripts': ['gtab-init=gtab.command_line:init_dir',
                              'gtab-print-options=gtab.command_line:print_options',
                              'gtab-set-options=gtab.command_line:set_options',
                              'gtab-set-blacklist=gtab.command_line:set_blacklist',
                              'gtab-list=gtab.command_line:list_gtabs',
                              'gtab-rename=gtab.command_line:rename_gtab',
                              'gtab-delete=gtab.command_line:delete_gtab',
                              'gtab-set-active=gtab.command_line:set_active_gtab',
                              'gtab-create=gtab.command_line:create_gtab',
                              'gtab-query=gtab.command_line:new_query'
                              ]
      }
      )
