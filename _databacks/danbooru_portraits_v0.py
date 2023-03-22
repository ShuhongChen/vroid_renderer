


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d


class DatabackendDanpor:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(),
    )
    def __init__(self, args=None, collate=False):
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/danbooru/raw/portraits'
        self.bns = sorted([
            uutil.fnstrip(fn)
            for fn in os.listdir(self.dn)
        ])
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        ret = Dict({
            'bn': bn,
            'image': I(f'{self.dn}/{bn}.jpg'),
        })
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret










