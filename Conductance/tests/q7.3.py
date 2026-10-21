test = {
  'name': 'q7.3',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> from scipy import stats as _st
          >>> f = _st.linregress(x_DH, log_Kc)
          >>> abs(log_Ka - f.intercept) < 1e-9 and abs(log_Ka_err - f.intercept_stderr) < 1e-9
          True
          >>> abs(Ka - 10**f.intercept) < 1e-12 and abs(pKa + f.intercept) < 1e-9
          True
          >>> abs(Ka_err - Ka * _np.log(10) * f.intercept_stderr) < 1e-12
          True
          >>> abs(slope_DH - f.slope) < 1e-9
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
