test = {
  'name': 'q1.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(gamma_mono - 5/3) < 1e-6
          True
          >>> abs(gamma_diatomic_rigid - 7/5) < 1e-6
          True
          >>> abs(gamma_diatomic_full - 9/7) < 1e-6
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
