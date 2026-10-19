test = {
  'name': 'q4.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> int(_np.sum(~_np.isnan(cpcv.column('dh1_cm')) & ~_np.isnan(cpcv.column('dh3_cm')))) >= 3
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
