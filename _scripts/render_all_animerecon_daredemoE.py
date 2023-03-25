


from _util.util_v1 import * ; import _util.util_v1 as uutil



DEBUG = False

bns = uutil.read_bns('./_data/lustrous/subsets/daredemoE_test.csv') \
    if not DEBUG else ['hololive/inugami_korone_v1.03',]
for bn in bns:
    franch,idx = bn.split('/')
    querystr = ','.join([
        'daredemoE',
        franch,
        idx,
        f'./_data/lustrous/raw/dssc/{franch}/{idx}.vrm',
    ])
    for dtype in [
        'ortho',
        'ortho_xyza',
        'rgb60',
        'xyza60',
    ]:
        print(f'{bn}: {dtype}')
        os.system(' '.join([
            'DEBUG=' if DEBUG else '',
            f'python3 -m _scripts.render_{dtype}',
            querystr,
        ]))











