test = {
  'name': 'q3.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(oil_cm_to_mmHg(24.25) - 18.72) < 0.01
          True
          >>> abs(inHg_to_mmHg(30.45) - 773.43) < 0.01
          True
          >>> abs(mmHg_to_Pa(760) - 101325) < 0.5
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
