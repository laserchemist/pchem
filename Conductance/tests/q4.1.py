test = {
  'name': 'q4.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> abs(kappa_KCl(25) - 0.2767) < 1e-9
          True
          >>> abs(kappa_KCl(20) - 0.2767 * (1 - 5 * 0.0193)) < 1e-9
          True
          >>> s0 = sets[0]
          >>> G0 = lookup(s0, 'KCl', 'G_mean (µS)') * G_UNIT
          >>> abs(k_cell[s0] - kappa_KCl(lookup(s0, 'KCl', 'T_mean (°C)')) / G0) < 1e-6 * k_cell[s0]
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
