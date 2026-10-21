test = {
  'name': 'q5.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> bool(_np.allclose(G_s, G_bar - Gw_row))
          True
          >>> bool(_np.allclose(kappa, k_row * (G_bar - Gw_row)))
          True
          >>> bool(_np.allclose(Lambda_m, k_row * (G_bar - Gw_row) / (1000 * c_M)))
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
