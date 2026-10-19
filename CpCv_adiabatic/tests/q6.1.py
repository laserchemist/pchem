test = {
  'name': 'q6.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> Q, s = q_statistic(_np.array([1.0, 2.0, 3.0, 10.0]))
          >>> abs(Q - 7/9) < 1e-6 and s == 10.0
          True
          >>> Q, s = q_statistic(_np.array([1.0, 8.0, 9.0, 10.0]))
          >>> abs(Q - 7/9) < 1e-6 and s == 1.0
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
