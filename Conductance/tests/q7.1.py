test = {
  'name': 'q7.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> abs(Lambda0_HAc - (Lambda0['HCl'] + Lambda0['NaAc'] - Lambda0['NaCl'])) < 1e-9
          True
          >>> abs(Lambda0_HAc_err - _np.sqrt(sum(Lambda0_err[e]**2 for e in ['HCl', 'NaAc', 'NaCl']))) < 1e-9
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
