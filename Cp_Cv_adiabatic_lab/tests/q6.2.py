test = {
  'name': 'q6.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> drop_suspect in (True, False)
          True
          >>> abs(gamma_mean - _np.mean(gamma_used)) < 1e-9
          True
          >>> abs(gamma_sd - _np.std(gamma_used, ddof=1)) < 1e-9
          True
          >>> abs(ci_half - t_crit * gamma_sd / _np.sqrt(n_used)) < 1e-9
          True
          >>> abs(ci_low - (gamma_mean - ci_half)) < 1e-9 and abs(ci_high - (gamma_mean + ci_half)) < 1e-9
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
