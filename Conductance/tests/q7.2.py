test = {
  'name': 'q7.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> bool(_np.allclose(alpha, Lambda_HAc / Lambda0_HAc))
          True
          >>> bool(_np.allclose(Kc, alpha**2 * c_HAc / (1 - alpha)))
          True
          >>> bool(_np.allclose(x_DH, _np.sqrt(alpha * c_HAc))) and bool(_np.allclose(log_Kc, _np.log10(Kc)))
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
