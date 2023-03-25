


from _util.util_v1 import * ; import _util.util_v1 as uutil



DEBUG = False

bns = uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv') \
    if not DEBUG else ['6152365338188306398',]
for bn in bns:
    querystr = ','.join([
        'rutileE',
        bn[-1],
        bn,
        f'./_data/lustrous/raw/vroid/{bn[-1]}/{bn}/{bn}.vrm',
    ])
    for dtype in [
        'ortho',
        'ortho_xyza',
        'rgb',
        'xyza',
    ]:
        print(f'{bn}: {dtype}')
        os.system(' '.join([
            'DEBUG=' if DEBUG else '',
            f'python3 -m _scripts.render_{dtype}',
            querystr,
        ]))











