test = {
  'name': 'q1.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(kappa_NaCl - 0.12640) < 0.0002
          True
          >>> abs(kappa_HCl - 0.42589) < 0.0005
          True
          >>> abs(Lambda_NaCl - 126.40) < 0.2
          True
          >>> abs(Lambda_HCl - 425.89) < 0.5
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
