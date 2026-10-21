test = {
  'name': 'q1.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(molar_to_SI(0.02) - 20) < 1e-9
          True
          >>> abs(Lambda_SI_to_cgs(0.01264) - 126.4) < 1e-6
          True
          >>> abs(resistivity_MOhm_cm(5.5e-6) - 18.18) < 0.01
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
