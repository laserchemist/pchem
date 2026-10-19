test = {
  'name': 'q1.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> n_vib_CO2
          4
          >>> abs(gamma_CO2_rigid - 7/5) < 1e-6
          True
          >>> abs(gamma_CO2_full - 15/13) < 1e-6
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
