test = {
  'name': 'q5.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> import numpy as _np
          >>> bool(_np.allclose(P2, inHg_to_mmHg(cpcv.column('Patm_inHg'))))
          True
          >>> bool(_np.allclose(P1 - P2, oil_cm_to_mmHg(cpcv.column('dh1_cm'))))
          True
          >>> bool(_np.allclose(P3 - P2, oil_cm_to_mmHg(cpcv.column('dh3_cm'))))
          True
          >>> 'gamma' in results.labels
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
