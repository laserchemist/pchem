test = {
  'name': 'q3.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> bool(_np.allclose(G_mean, _np.nanmean(G_runs, axis=1), equal_nan=True))
          True
          >>> bool(_np.allclose(G_sd, _np.nanstd(G_runs, axis=1, ddof=1), equal_nan=True))
          True
          >>> bool(_np.allclose(T_mean, _np.nanmean(T_runs, axis=1), equal_nan=True))
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
