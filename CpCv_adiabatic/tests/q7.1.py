test = {
  'name': 'q7.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(percent_diff - 100 * (gamma_mean - gamma_lit[gas]) / gamma_lit[gas]) < 1e-6
          True
          >>> abs(vdw_factor - (1 + 2 * a_SI * P_Pa / (R**2 * T_K**2))) < 1e-9
          True
          >>> abs(gamma_real - (1 + (gamma_ideal - 1) * vdw_factor)) < 1e-9
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
