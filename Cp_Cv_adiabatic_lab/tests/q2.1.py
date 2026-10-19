test = {
  'name': 'q2.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(vib_temperature(1000) - 1438.78) < 0.1
          True
          >>> abs(einstein_cv(1.0, 1e4) - 1.0) < 1e-6
          True
          >>> abs(einstein_cv(1438.78, 295) - 0.1840) < 0.001
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
