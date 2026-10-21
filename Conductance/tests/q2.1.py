test = {
  'name': 'q2.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(c_HCl - 0.020570) < 1e-5
          True
          >>> sorted(c_stock.keys()) == ['HAc', 'HCl', 'NaAc', 'NaCl']
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
