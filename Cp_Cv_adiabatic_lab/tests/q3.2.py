test = {
  'name': 'q3.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(gamma_exact(789.86, 771.14, 776.86) - 1.4445) < 0.001
          True
          >>> abs(gamma_approx(24.25, 7.40) - 1.4392) < 0.001
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
