test = {
  'name': 'q6.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> sub = results.where('Solution', 'NaCl')
          >>> from scipy import stats as _st
          >>> f = _st.linregress(sub.column('sqrt c'), sub.column('Lambda (S cm2/mol)'))
          >>> abs(Lambda0['NaCl'] - f.intercept) < 1e-6 and abs(K_kohl['NaCl'] + f.slope) < 1e-6
          True
          >>> abs(Lambda0_err['NaCl'] - f.intercept_stderr) < 1e-6
          True
          >>> sorted(Lambda0.keys()) == ['HCl', 'NaAc', 'NaCl']
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
