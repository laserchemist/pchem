test = {
  'name': 'q3.1',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> sols = list(cond.column('Solution'))
          >>> all(s in sols for s in ['Water', 'KCl'])
          True
          >>> all(sols.count(e) >= 3 for e in ['HCl', 'NaCl', 'NaAc', 'HAc'])
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
