test = {
  'name': 'q2.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(concentration('HCl', 'c/16') - c_stock['HCl'] / 16) < 1e-12
          True
          >>> abs(concentration('NaAc', 'c') - c_stock['NaAc']) < 1e-12
          True
          """,
          'hidden': False,
          'locked': False
        }
      ],
      'scored': True,
      'setup': '',
      'teardown': '',
      'type': 'doctest'
    }
  ]
}
