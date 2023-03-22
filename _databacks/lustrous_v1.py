




from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d


cam60 = torch.tensor(np.stack(np.meshgrid(
    np.linspace(60, -20,  5),
    np.linspace(-180, 150, 12),
)).T.reshape(60,-1)).float()

camsubs = {
    'all': list(range(60)),
    'front1': [42,],
    'front15': [
        28, 29, 30, 31, 32,
        40, 41, 42, 43, 44,
        52, 53, 54, 55, 56,
    ],
}




















