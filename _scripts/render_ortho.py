


# early check
import os, argparse, time
TIME = time.time()

# DEBUG = False
DEBUG = 'DEBUG' in os.environ
# name = 'phosA'

# pick vrm
ap = argparse.ArgumentParser()
ap.add_argument('name')
# name = 'daredemoE'
# name,franch,bn = ap.parse_args().name.split(',')
name,franch,bn,ifn = ap.parse_args().name.split(',')
# bn = '6152365338188306398'  # vanilla
# bn = '5057537072278330978'
# bn = '8015923251209631708'
odn = f'./_data/lustrous/renders/{name}' if not DEBUG else f'/dev/shm/_mgl_test/{name}'

render_dtype = 'ortho'
if os.path.isfile(f'{odn}/{render_dtype}/_logs/succ/{bn}.txt'):
    print(f'JOB {bn} ALREADY COMPLETED')
    exit(0)
if os.path.isfile(f'{odn}/{render_dtype}/_logs/fails/{bn}.txt'):
    print(f'JOB {bn} ALREADY FAILED')
    exit(0)




from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
from pathlib import Path

from _databacks import lustrous_gltf_v0 as uvrm


# from _scripts.gltf_scenes_amber import CubeModel
import pyrr
from pyrr import Matrix44

import moderngl
import moderngl_window as mglw
from moderngl_window.scene.camera import KeyboardCamera, OrbitCamera


# render params
# elevations = [-30, 0, 30,]
# azimuths = np.linspace(0,360,8+1)[:-1]
# elevations = [0,]
# azimuths = 180 + np.asarray([0, 45, 90])
# if name=='rutileD':
#     # n_generations = 8 if not DEBUG else 16
#     elevation_mu, elevation_sig = 0.0, 20
#     azimuth_mu, azimuth_sig = 180.0, 100000  # basically uniform
#     distance_mu, distance_sig = 1.0, 0.0
#     fov_mu, fov_sig = 30.0, 0.0  # full-fov, only zero std supported atm
#     # fov_mu, fov_sig = 24.0, 0.0  # full-fov, only zero std supported atm
# else:
#     assert 0
boxwarp = 0.7
frust_near,frust_far = 0.5, 1.5
# frust_near,frust_far = 1.5, 0.5
assert name in {'daredemoE', 'rutileE'}
elevations = [90, 0,0,0,0]
azimuths = np.asarray([0, -180,-90,0,90])
distances = [1.0,]*5
fovs = [30.0,]*5
render_idxs = ['top', 'back', 'right', 'front', 'left']
# c = 0
# for elev in np.linspace(60, -60, 7):
#     for azim in np.linspace(-180, 150, 12):
#         if c in [6, 36, 39, 42, 45]:
#             elevations.append(elev)
#             azimuths.append(azim-180)
#             distances.append(distance_mu)
#             fovs.append(fov_mu)
#             render_idxs.append(c)
#         c += 1
# for i in range(n_generations):
#     with np_seed(f'{bn}/{i}'):
#         elevations.append(elevation_mu + elevation_sig*np.random.randn())
#         azimuths.append(azimuth_mu + azimuth_sig*np.random.randn())
#         distances.append(distance_mu + distance_sig*np.random.randn())
#         fovs.append(fov_mu + fov_sig*np.random.randn())

# other render params
offset_head = np.asarray([0, 0.1, 0])
size = 1024
# thresh_alpha = 4/16, 12/16
thresh_alpha = None
thresh_mesh_size = None
# thresh_mesh_size = np.asarray([
#     [-1.0, 1.0],  # arms: -0.6 to 0.6
#     [-0.1, 2.0],  # height: 0 to 1.7
#     [-0.5, 0.75], # belly: -0.2 to 0.2 (front, back)
# ] )
dn_temp_gltf = f'/dev/shm/__shu_gltf_noext/{name}_{isonow()}'

# mini render
mini_size = 512
mini_mode = 'bilinear'
# mini_bg = 1.0


def projection_matrix_ortho():
    bw2 = boxwarp / 2.0
    l,r = -bw2, bw2
    b,t = -bw2, bw2
    n,f = frust_near, frust_far
    return Matrix44(np.asarray([
        [2/(r-l), 0, 0, -(r+l)/(r-l)],
        [0, 2/(t-b), 0, -(t+b)/(t-b)],
        [0, 0, -2/(f-n), -(f+n)/(f-n)],
        [0, 0, 0, 1, ],
    ]).T.copy())
projection_matrix_ortho = projection_matrix_ortho()



# debug grid
if DEBUG:
    igrid = I.blank(size, 'a')
    sq = size-1
    for q in np.linspace(0,sq,5):
        igrid = igrid.line((0,q), (sq,q))
        igrid = igrid.line((q,0), (q,sq))


class CameraWindow(mglw.WindowConfig):
    """Base class with built in 3D camera support"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera = KeyboardCamera(self.wnd.keys, aspect_ratio=self.wnd.aspect_ratio)
        self.camera_enabled = True

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if self.camera_enabled:
            self.camera.key_input(key, action, modifiers)

        if action == keys.ACTION_PRESS:
            if key == keys.C:
                self.camera_enabled = not self.camera_enabled
                self.wnd.mouse_exclusivity = self.camera_enabled
                self.wnd.cursor = not self.camera_enabled
            if key == keys.SPACE:
                self.timer.toggle_pause()
        # print(key, type(key))
        # print(action, type(action))
        # print(modifiers, type(modifiers))

    def mouse_position_event(self, x: int, y: int, dx, dy):
        if self.camera_enabled:
            self.camera.rot_state(-dx, -dy)

    def resize(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self.wnd.aspect_ratio)

class CustomProgram(mglw.scene.programs.MeshProgram):
    """
    Simple texture program
    """

    def __init__(self, program=None, loc_origin=None, **kwargs):
        super().__init__(program=None)
        self.program = mglw.resources.programs.load(
            mglw.meta.ProgramDescription(
                # path=Path("./temp/_moderngl/scene_default_programs/texture_light.glsl").resolve(),
                # path=Path("./_cells/shaders/_test/test.glsl").resolve(),
                path=Path("./_cells/shaders/rutileE.glsl").resolve(),
            )
        )
        self.loc_origin = loc_origin.astype(np.float32)

    def draw(
        self,
        mesh,
        projection_matrix=None,
        model_matrix=None,
        camera_matrix=None,
        time=0,
    ):
        # if mesh.material.double_sided:
        #     self.ctx.disable(moderngl.CULL_FACE)
        # else:
        #     self.ctx.enable(moderngl.CULL_FACE)

        mesh.material.mat_texture.texture.use()
        self.program["texture0"].value = 0
        self.program["m_proj"].write(projection_matrix)
        self.program["m_model"].write(model_matrix)
        self.program["m_cam"].write(camera_matrix)
        self.program['loc_origin'].write(self.loc_origin)
        self.program['boxwarp'].write(np.asarray(boxwarp).astype(np.float32))

        # https://github.com/moderngl/moderngl-window/blob/55d7d5a988564da6afff128b4513e3712fd3312b/moderngl_window/loaders/scene/gltf2.py#L247
        self.program['base_color_factor'].write(np.asarray(mesh.material.color, dtype=np.float32))

        mesh.vao.render(self.program)

    def apply(self, mesh):
        if not mesh.material:
            return None

        # if not mesh.attributes.get("NORMAL"):
        #     return None

        if not mesh.attributes.get("TEXCOORD_0"):
            return None

        if mesh.material.mat_texture is not None:
            return self

        return None

# class GLTFModel(OrbitCameraWindow):
class GLTFModel(CameraWindow):
    """
    In oder for this example to work you need to clone the gltf
    model samples repository and ensure resource_dir is set correctly:
    https://github.com/KhronosGroup/glTF-Sample-Models/tree/master/2.0
    """
    title = 'GL Transmission Format (glTF) 2.0 Scene'
    window_size = (size, size)
    # window_size = 1280, 720
    aspect_ratio = None
    # resource_dir = Path(__file__, '../../../glTF-Sample-Models/2.0').resolve()
    # resource_dir = Path('./_data/gltf_samples/2.0').resolve()
    # resource_dir = Path('./temp/_moderngl').resolve()

    # def __init__(self, fn_gltf, spin=15, **kwargs):
    def __init__(self, fn_gltf, fov, loc_origin, **kwargs):
        super().__init__(**kwargs)
        self.wnd.mouse_exclusivity = True
        self.fn_gltf = fn_gltf
        # self.spin = spin
        self.loc_origin = loc_origin

        # --- glTF-Sample-Models ---
        self.scene = self.load_scene(self.fn_gltf)
        # self.scene = self.load_scene('amber_noext.gltf')
        self.scene.apply_mesh_programs(mesh_programs=[CustomProgram(loc_origin=self.loc_origin)], clear=True)
        # self.scene.apply_mesh_programs(mesh_programs=[mglw.scene.programs.TextureLightProgram()], clear=True)
        # self.scene.apply_mesh_programs(mesh_programs=[mglw.scene.programs.ColorLightProgram()], clear=True)
        # self.scene.apply_mesh_programs(mesh_programs=[mglw.scene.programs.FallbackProgram()], clear=True)

        # self.scene = self.load_scene('2CylinderEngine/glTF-Binary/2CylinderEngine.glb')
        # self.scene = self.load_scene('CesiumMilkTruck/glTF-Embedded/CesiumMilkTruck.gltf')
        # self.scene = self.load_scene('CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb')
        # self.scene = self.load_scene('CesiumMilkTruck/glTF/CesiumMilkTruck.gltf')
        # # self.scene = self.load_scene('Sponza/glTF/Sponza.gltf')
        # self.scene = self.load_scene('Lantern/glTF-Binary/Lantern.glb')
        # self.scene = self.load_scene('Buggy/glTF-Binary/Buggy.glb')
        # self.scene = self.load_scene('VC/glTF-Binary/VC.glb')
        # self.scene = self.load_scene('DamagedHelmet/glTF-Binary/DamagedHelmet.glb')
        # self.scene = self.load_scene('BoxInterleaved/glTF/BoxInterleaved.gltf')
        # self.scene = self.load_scene('OrientationTest/glTF/OrientationTest.gltf')
        # self.scene = self.load_scene('AntiqueCamera/glTF/AntiqueCamera.gltf')
        # self.scene = self.load_scene('BoomBox/glTF/BoomBox.gltf')
        # self.scene = self.load_scene('Box/glTF/Box.gltf')
        # self.scene = self.load_scene('BoxTextured/glTF/BoxTextured.gltf')
        # self.scene = self.load_scene('BoxTexturedNonPowerOfTwo/glTF/BoxTexturedNonPowerOfTwo.gltf')
        # self.scene = self.load_scene('BoxVertexColors/glTF/BoxVertexColors.gltf')
        # self.scene = self.load_scene('BrainStem/glTF/BrainStem.gltf')
        # self.scene = self.load_scene('Corset/glTF/Corset.gltf')
        # self.scene = self.load_scene('FlightHelmet/glTF/FlightHelmet.gltf')
        # self.scene = self.load_scene('Fox/glTF/Fox.gltf')
        # self.scene = self.load_scene('GearboxAssy/glTF/GearboxAssy.gltf')
        # self.scene = self.load_scene('ReciprocatingSaw/glTF/ReciprocatingSaw.gltf')
        # self.scene = self.load_scene('RiggedFigure/glTF/RiggedFigure.gltf')
        # self.scene = self.load_scene('RiggedSimple/glTF/RiggedSimple.gltf')
        # self.scene = self.load_scene('SciFiHelmet/glTF/SciFiHelmet.gltf')
        # self.scene = self.load_scene('SimpleMeshes/glTF/SimpleMeshes.gltf')
        # self.scene = self.load_scene('SimpleSparseAccessor/glTF/SimpleSparseAccessor.gltf')
        # self.scene = self.load_scene('Suzanne/glTF/Suzanne.gltf')
        # self.scene = self.load_scene('TextureCoordinateTest/glTF/TextureCoordinateTest.gltf')
        # self.scene = self.load_scene('TextureSettingsTest/glTF/TextureSettingsTest.gltf')
        # self.scene = self.load_scene('VertexColorTest/glTF/VertexColorTest.gltf')
        # self.scene = self.load_scene('WaterBottle/glTF/WaterBottle.gltf')

        self.camera = KeyboardCamera(self.wnd.keys, fov=fov, aspect_ratio=self.wnd.aspect_ratio, near=frust_near, far=frust_far)
        # self.camera.velocity = 7.0
        # self.camera.mouse_sensitivity = 0.3

        # Use this for gltf scenes for better camera controls
        if self.scene.diagonal_size > 0:
            self.camera.velocity = self.scene.diagonal_size / 5.0

        self.generation_counter = 0
        return

    def render(self, time: float, frame_time: float):
        """Render the scene"""
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        # self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        # self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        # self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND)
        # self.ctx.enable_only(moderngl.BLEND)  # cursed

        # elev,azim = time, frame_time
        elev = elevations[self.generation_counter]
        azim = azimuths[self.generation_counter] - 180
        dist = distances[self.generation_counter]
        fov = fovs[self.generation_counter]  # unused

        # Move camera in on the z axis slightly by default
        # translation0 = Matrix44.from_translation((0, -0.75, 0), dtype='f4')
        # move camera to neck bone
        # print(loc_bone_head)
        # print(loc_bone_neck)
        # translation0 = Matrix44.from_translation(-loc_bone_neck, dtype='f4')
        translation0 = Matrix44.from_translation(-self.loc_origin, dtype='f4')
        # rotation = pyrr.Matrix44.from_eulers(np.asarray([0,0,135])*np.pi/180, dtype='f4')
        rotation = pyrr.Matrix44.from_eulers(np.asarray([-elev,0,azim])*np.pi/180, dtype='f4')
        translation1 = Matrix44.from_translation((0, 0, -dist), dtype='f4')
        camera_matrix = self.camera.matrix * translation1 * rotation * translation0
        # print(np.asarray(self.camera.matrix).T)
        # camera_matrix = translation1 * rotation * translation0
        # uutil.pdump(self.camera.projection.matrix, mkfile('./temp/_cameras/intrinsic.pkl'))
        # uutil.pdump(translation1 * rotation * translation0, mkfile('./temp/_cameras/extrinsic.pkl'))
        # uutil.pdump({
        #     'loc_origin': self.loc_origin,
        #     'elev': elev,
        #     'azim': azim,
        #     'dist': dist,
        # }, mkfile('./temp/_cameras/params.pkl'))
        # exit(0)

        # def _node_with_mesh(node):
        #     if node._mesh is None:
        #         for ch in node.children:
        #             out = _node_with_mesh(ch)
        #             if out is not None:
        #                 return out
        #     else:
        #         return node
        # for node in self.scene.root_nodes:
        #     out = _node_with_mesh(node)
        #     if out is not None:
        #         node = out
        #         break
        # print(node._mesh.mesh_program)
        # exit(0)

        self.scene.draw(
            # projection_matrix=self.camera.projection.matrix,
            projection_matrix=projection_matrix_ortho,
            camera_matrix=camera_matrix,
            time=time,
        )

        # Draw bounding boxes
        # self.scene.draw_bbox(
        #     projection_matrix=self.camera.projection.matrix,
        #     camera_matrix=camera_matrix,
        #     children=True,
        #     color=(0.75, 0.75, 0.75),
        # )

        # self.scene.draw_wireframe(
        #     projection_matrix=self.camera.projection.matrix,
        #     camera_matrix=camera_matrix,
        #     color=(1, 1, 1, 1),
        # )

        self.generation_counter += 1
        return



def main():

    ################ GLTF BONE STUFF ################

    # load gltf
    # try:
    vrm = LustrousGLTF(ifn)
    # vrm = LustrousGLTF(f'./_data/lustrous/raw/dssc/{franch}/{bn}.vrm')
    # vrm = LustrousGLTF(f'./_data/lustrous/raw/vroid/{franch}/{bn}/{bn}.vrm')
    gltf = vrm.gltf
    # gltf = pygltflib.GLTF2().load_binary(f'./_data/lustrous/raw/vroid/{franch}/{bn}/{bn}.vrm')
    # except:
    #     print(f'VRM {bn} failed to open')
    #     print(f'./_data/lustrous/raw/vroid/{franch}/{bn}/{bn}.vrm')
    #     if not DEBUG:
    #         write(isonow(), mkfile(f'{odn}/fails/{bn}.txt'))
    #     exit(0)

    # apply lbs to put hands down
    fn_gltf = f'{dn_temp_gltf}/{bn.replace(".","-")}/{bn.replace(".","-")}_noext.gltf'
    _write_lbs_gltf(vrm, fn_gltf)

    # # remove extensions
    # def _remove_gltf_ext(gltf):
    #     fn_gltf = f'{dn_temp_gltf}/{bn}/{bn}_noext.gltf'
    #     gltf_noext = copy.deepcopy(gltf)
    #     gltf_noext.extensionsUsed = gltf_noext.extensionsRequired = []
    #     gltf_noext.save_json(mkfile(fn_gltf))
    #     return fn_gltf
    # fn_gltf = _remove_gltf_ext(gltf)

    # remove hugemeshes
    def _get_verts(gltf):
        def gltf_accessor(gltf, accessor_idx):
            acc = gltf.accessors[accessor_idx]
            bv = gltf.bufferViews[acc.bufferView]
            blob = gltf.binary_blob()
            # blob = bins[bv.buffer] if bins else gltf.load_file_uri(gltf.buffers[bv.buffer].uri)
            arr = np.frombuffer(
                blob,
                dtype=u3d._component_dtypes[acc.componentType],
                count=acc.count*np.prod(u3d._accessor_ncomps[acc.type]),
                offset=bv.byteOffset+acc.byteOffset,
            ).reshape(acc.count, *u3d._accessor_ncomps[acc.type])
            return arr
        _verts = []
        # _norms = []
        # _faces = []
        # _uvcol = []
        # vc = 0
        # timgs = []
        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                # for triangle meshes
                # assert u3d._mesh_primitive_modes[prim.mode]=='TRIANGLES'
                if u3d._mesh_primitive_modes[prim.mode]!='TRIANGLES': continue
                
                # attributes = {
                #     k: gltf_accessor(gltf, v)
                #     for k,v in json.loads(prim.attributes.to_json()).items()
                #     if v is not None
                # }
                # indices = gltf_accessor(gltf, prim.indices)

                # material = gltf.materials[prim.material]
                # texture = gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index]
                # texture_img = gltf_image(gltf, texture.source)
                # texture_set = material.pbrMetallicRoughness.baseColorTexture.texCoord
                # timgs.append(texture_img)

                try:
                    a = json.loads(prim.attributes.to_json())['POSITION']
                    verts = gltf_accessor(gltf, a)
                except:
                    continue
                vmin = verts.min(axis=0)
                vmax = verts.max(axis=0)
                if np.any(vmin<thresh_mesh_size[:,0]) or np.any(thresh_mesh_size[:,1]<vmax):
                    print(f'VRM {bn} failed hugemesh test')
                    # window.close(); timer.stop(); window.destroy()
                    if not DEBUG: shutil.rmtree(f'{dn_temp_gltf}/{bn.replace(".","-")}')
                    assert 0, f'VRM {bn} failed hugemesh test'
                    # exit(0)
                # verts = attributes['POSITION']
                # norms = attributes['NORMAL']
                # faces = indices.reshape(-1, 3)
                # uvmap = attributes[f'TEXCOORD_{texture_set}']
                # uvcol = sample_texture_uv(texture_img, uvmap)
                _verts.append(verts)
                # _norms.append(norms)
                # _faces.append(faces)
                # _uvcol.append(uvcol)
                # vc += len(verts)
        # return np.concatenate(_verts)
        return
    # _get_verts(gltf)
    # # verts = _get_verts(gltf)
    # # vmin = verts.min(axis=0)
    # # vmax = verts.max(axis=0)
    # # if np.any(vmin<thresh_mesh_size[:,0]) or np.any(thresh_mesh_size[:,1]<vmax):
    # #     print(f'VRM {bn} failed hugemesh test')
    # #     # window.close(); timer.stop(); window.destroy()
    # #     if not DEBUG: shutil.rmtree(f'{dn_temp_gltf}/{bn.replace(".","-")}')
    # #     assert 0, f'VRM {bn} failed hugemesh test'
    # #     # exit(0)

    # get neck and head locations
    def _get_bone_locs(gltf):
        def gltf_accessor(gltf, accessor_idx):
            acc = gltf.accessors[accessor_idx]
            bv = gltf.bufferViews[acc.bufferView]
            blob = gltf.binary_blob()
            # blob = bins[bv.buffer] if bins else gltf.load_file_uri(gltf.buffers[bv.buffer].uri)
            arr = np.frombuffer(
                blob,
                dtype=u3d._component_dtypes[acc.componentType],
                count=acc.count*np.prod(u3d._accessor_ncomps[acc.type]),
                offset=bv.byteOffset+acc.byteOffset,
            ).reshape(acc.count, *u3d._accessor_ncomps[acc.type])
            return arr

        # get interesting nodes
        inodes = Dict()
        hbones = gltf.extensions['VRM']['humanoid']['humanBones']
        for hb in hbones:
            if hb['bone']=='neck':
                inodes['neck'] = hb['node']
            elif hb['bone']=='head':
                inodes['head'] = hb['node']
            elif hb['bone']=='leftEye':
                inodes['eye_left'] = hb['node']
            elif hb['bone']=='rightEye':
                inodes['eye_right'] = hb['node']
        assert 'neck' in inodes.keys() and 'head' in inodes.keys()
        # assert set(inodes.keys())=={'neck', 'head', 'eye_left', 'eye_right'}

        g_skin = gltf.skins[0]
        ibms = np.transpose(gltf_accessor(gltf, accessor_idx=g_skin.inverseBindMatrices), (0,2,1))
        ibm_neck = ibms[g_skin.joints.index(inodes['neck'])]
        ibm_head = ibms[g_skin.joints.index(inodes['head'])]
        loc_bone_neck = -ibm_neck[:3,-1]
        loc_bone_head = -ibm_head[:3,-1]
        return loc_bone_neck, loc_bone_head
    loc_bone_neck, loc_bone_head = _get_bone_locs(gltf)
    loc_origin = loc_bone_head+offset_head


    ################ MODERNGL STUFF ################

    config_cls = GLTFModel
    timer = None
    args = None

    # parse
    parser = mglw.create_parser()
    config_cls.add_arguments(parser)
    values = parser.parse_args([])
    values.window = 'headless'
    config_cls.argv = values
    window_cls = mglw.get_local_window_cls(values.window)

    # # Calculate window size
    # size = values.size or config_cls.window_size
    # size = int(size[0] * values.size_mult), int(size[1] * values.size_mult)

    # Resolve cursor
    show_cursor = values.cursor
    if show_cursor is None:
        show_cursor = config_cls.cursor

    window = window_cls(
        title=config_cls.title,
        size=(size, size),
        fullscreen=config_cls.fullscreen or values.fullscreen,
        resizable=values.resizable if values.resizable is not None else config_cls.resizable,
        gl_version=config_cls.gl_version,
        aspect_ratio=config_cls.aspect_ratio,
        vsync=values.vsync if values.vsync is not None else config_cls.vsync,
        samples=values.samples if values.samples is not None else config_cls.samples,
        cursor=show_cursor if show_cursor is not None else True,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    timer = timer or mglw.Timer()
    config = config_cls(
        fn_gltf=fn_gltf,
        fov=fovs[0],
        loc_origin=loc_origin,
        # spin=0,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )

    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    import weakref
    window._config = weakref.ref(config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    # while not window.is_closing:
    #     current_time, delta = timer.next_frame()

    #     if config.clear_color is not None:
    #         window.clear(*config.clear_color)

    #     # Always bind the window framebuffer before calling render
    #     window.use()

    #     window.render(current_time, delta)
    #     if not window.is_closing:
    #         window.swap_buffers()
        
    #     data = copy.deepcopy(window.fbo.read(components=4))
    #     window.close()
    #     # window.key_event(65307, 'ACTION_PRESS', mglw.context.base.keys.KeyModifiers())
    outs = []
    # for elev in tqdm(elevations, desc=bn):
        # for azim in azimuths:
    # for i in trange(n_generations, desc=bn):
    for i in trange(len(elevations), desc=bn):
        current_time, delta = timer.next_frame()

        if config.clear_color is not None:
            window.clear(*config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()
        
        data = copy.deepcopy(window.fbo.read(components=4))
        # data = np.frombuffer(data, dtype=np.uint8).reshape(*size[::-1],4)[::-1]
        data = np.frombuffer(data, dtype=np.uint8).reshape(size,size,4)[::-1]
        data = I(data)
        # I(data).acomp(igrid).save(mkfile(f'{odn}/{bn}_{int(elev):+04d}_{int(azim):03d}.png'))
        acheck = data['a'].np().mean()
        if not DEBUG and thresh_alpha!=None and not (thresh_alpha[0] < acheck < thresh_alpha[1]):
            print(f'VRM {bn} failed transparency test')
            window.close(); timer.stop(); window.destroy()
            if not DEBUG: shutil.rmtree(f'{dn_temp_gltf}/{bn.replace(".","-")}')
            assert 0, f'VRM {bn} failed transparency test'
            # exit(0)
        outs.append(data)
    for i,out in enumerate(outs):
        # if DEBUG: out = out.acomp(igrid)
        # out.save(mkfile(f'{odn}/{render_dtype}/{franch}/{bn}/{i:04d}.png'))
        out.resize(mini_size, mini_mode).save(mkfile(f'{odn}/{render_dtype}/{franch}/{bn}/{render_idxs[i]}.png'))
        # out.resize(mini_size, mini_mode).bg(mini_bg).save(mkfile(f'{odn}/{render_dtype}/{franch}/{bn}/{render_idxs[i]}.png'))
        # out.save(mkfile(f'{odn}/{render_dtype}/{franch}/{bn}/{i:04d}.png'))
        # out.resize(mini_size, mini_mode).bg(mini_bg).save(mkfile(f'{odn}/{mini_size}/{franch}/{bn}/{i:04d}.png'))
    window.close()

    _, duration = timer.stop()
    window.destroy()
    # if duration > 0:
    #     logger.info(
    #         "Duration: {0:.2f}s @ {1:.2f} FPS".format(
    #             duration, window.frames / duration
    #         )
    #     )
    if not DEBUG: shutil.rmtree(f'{dn_temp_gltf}/{bn.replace(".","-")}')

    if DEBUG:
        I.grid(
            chunk_cols(
                [out.acomp(igrid).resize(256, 'bilinear') for out in outs],
                int(np.sqrt(len(elevations))),
                # int(np.sqrt(n_generations)),
            )
        ).save(f'{odn}/{render_dtype}/{franch}/{bn}/_all.png')




    # q = np.frombuffer(data, dtype=np.uint8).reshape(*size[::-1],4)[::-1]#.transpose(2,0,1)[[2,1,0,3]]
    # I(q).border().save('./temp/_moderngl/headless_amber_fuckyes.exr')
    # I(q).border().save('./temp/_moderngl/headless_amber_fuckyes.png')
    return


################ GLTF FOR LBS ################

def _write_lbs_gltf(vrm, fn_out):
    self = vrm
    hbones = self.gltf.extensions['VRM']['humanoid']['humanBones']
    hindex = {
        hb['bone']: hb['node']
        for hb in hbones
    }
    if name=='daredemoE':
        theta = {
            # hololive/genshin mmds
            hindex['leftUpperArm']: [0, 0, (60-45)],
            hindex['rightUpperArm']: [0, 0, -(60-45)],
        }
    elif name=='rutileE':
        theta = {
            # vroids
            hindex['leftUpperArm']: [0, 0, (60)],
            hindex['rightUpperArm']: [0, 0, -(60)],
        }
    else:
        assert 0
    M = apply_lbs(vrm.arm_rest, theta)
    jorder = {
        v['joint_idx']: k
        for k,v in vrm.arm_rest.items()
    }
    Ms = np.stack([
        M[jorder[j]]
        for j in range(len(jorder))
        # M[i] if i in M else np.eye(4)
        # for i in range(max(M.keys()))
    ])
    out = copy.deepcopy(self.gltf)
    refd = set()
    for mesh in self.gltf.meshes:
        for prim in mesh.primitives:
            # for triangle meshes
            assert u3d._mesh_primitive_modes[prim.mode]=='TRIANGLES'

            # grab all attributes
            attributes = {
                k: gltf_accessor(self.gltf, v)
                for k,v in json.loads(prim.attributes.to_json()).items()
                if v is not None and k in ['JOINTS_0', 'WEIGHTS_0', 'POSITION']
                # if v is not None
            }
            # indices = gltf_accessor(self.gltf, prim.indices)
            verts = attributes['POSITION']
            
            # skin weights
            swj = attributes['JOINTS_0']
            sww = attributes['WEIGHTS_0']
            # swj = gltf_accessor(self.gltf, prim.attributes.JOINTS_0)
            # sww = gltf_accessor(self.gltf, prim.attributes.WEIGHTS_0)
            # if 'JOINTS_1' in attributes and 'WEIGHTS_1' in attributes:
            #     swj = np.concatenate([
            #         swj, gltf_accessor(self.gltf, prim.attributes.JOINTS_1),
            #     ], axis=1)
            #     sww = np.concatenate([
            #         sww, gltf_accessor(self.gltf, prim.attributes.WEIGHTS_1),
            #     ], axis=1)
            sww = sww / sww.sum(axis=1, keepdims=True)
            refd = refd.union([i for i in np.unique(swj)])
            
            # apply lbs
            v = np.concatenate([
                verts,
                np.ones((len(verts),1)),
            ], axis=1)
            t = (Ms[swj] * sww[...,None,None]).sum(axis=1)
            v_ = (t @ v[...,None])[:,:3,0]
            gltf_write_blob(out, prim.attributes.POSITION, v_)
            # break
        # break

    # also remove extensions
    out.extensionsUsed = out.extensionsRequired = []
    out.save_json(mkfile(fn_out))
    return

def apply_lbs(arm, theta, idx=None, ans=None):
    # top-level iterate
    if ans is None:
        # convert theta
        T = {}
        for k,v in arm.items():
            q = np.eye(4)
            if k in theta:
                q[:3,:3] = scipy.spatial.transform.Rotation.from_euler(
                    'xyz', theta[k], degrees=True,
                ).as_matrix()
            T[k] = q
        
        # iterate
        ans = {}
        for idx in sorted(arm):
            apply_lbs(arm, T, idx=idx, ans=ans)
    
    # recursion
    else:
        # get parent
        p = arm[idx]['parent_idx']
        # print(idx, p)
        if p==-1:
            ansp = np.eye(4)
            up = np.eye(4)
        else:
            if p not in ans:
                apply_lbs(arm, T, idx=p, ans=ans)
            ansp = ans[p]
            up = arm[p]['u']

        # figure self
        u,r = arm[idx]['u'], arm[idx]['r']
        q = u  # goto origin
        q = theta[idx] @ q  # apply T
        q = np.linalg.inv(up) @ q  # goto parent
        q = ansp @ r @ q  # apply relative to parent
        ans[idx] = q
    
    return ans

def gltf_accessor(gltf, accessor_idx):
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    # blob = bins[bv.buffer] if bins else gltf.load_file_uri(gltf.buffers[bv.buffer].uri)
    arr = np.frombuffer(
        blob,
        dtype=u3d._component_dtypes[acc.componentType],
        count=acc.count*np.prod(u3d._accessor_ncomps[acc.type]),
        offset=bv.byteOffset+acc.byteOffset,
    ).reshape(acc.count, *u3d._accessor_ncomps[acc.type])
    return arr
def gltf_write_blob(gltf, accessor_idx, data):
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    # blob = bins[bv.buffer] if bins else gltf.load_file_uri(gltf.buffers[bv.buffer].uri)
    
    dtype = u3d._component_dtypes[acc.componentType]
    count = acc.count*np.prod(u3d._accessor_ncomps[acc.type])
    offset = bv.byteOffset+acc.byteOffset
    data = data.flatten().astype(dtype)
    arr = data.tobytes()
    assert len(arr)==count*data.dtype.itemsize, 'can only write same size for now'

    gltf.set_binary_blob(blob[:offset] + arr + blob[offset+len(arr):])
    return

# class container
class LustrousGLTF:
    # load gltf
    def __init__(self, fn, scale_factor=1.0):
        
        self.fn = fn
        self.gltf = gltf = pygltflib.GLTF2().load_binary(self.fn)
        self.scale_factor = scale_factor
        
        g_skin = self.gltf.skins[0]
        g_nodes = self.gltf.nodes
        g_nodes_joint = [(i,g_nodes[i]) for i in g_skin.joints]
        
        
        ################ BONES ################
        
        # get inverse bind matrices
        us = gltf_accessor(self.gltf, g_skin.inverseBindMatrices)
        us = np.transpose(us, (0,2,1)).copy()
        us = {
            i: u
            for (i,j),u in zip(g_nodes_joint, us)
        }
        
        # parenting
        ar = {i: {'joint_idx': e} for e,(i,j) in enumerate(g_nodes_joint)}
        for bi,nj in g_nodes_joint:
            chidxs = []
            for chi in nj.children:
                if chi in ar:
                    ar[chi]['parent_idx'] = bi
                    chidxs.append(chi)
            ar[bi]['children_idx'] = sorted(chidxs)
            ar[bi]['u'] = us[bi]
            ar[bi]['u'][:3,3] *= self.scale_factor
        for bi,nj in g_nodes_joint:
            if 'parent_idx' not in ar[bi]:
                ar[bi]['parent_idx'] = -1
                up = np.eye(4)
            else:
                up = ar[ar[bi]['parent_idx']]['u']
            ar[bi]['r'] = up @ np.linalg.inv(ar[bi]['u'])
        self.arm_rest = ar
        
        
        ################ MESH DATA ################
        
        # get combined attributes
        # _verts = []
        # _norms = []
        # _faces = []
        # _uvcol = []
        # _uvmap = []
        # _texidxs = []
        # _basecol = []
        # _swj = []
        # _sww = []
        # vc = 0
        # tc = 0
        # timgs = []
        # for mesh in gltf.meshes:
        #     for prim in mesh.primitives:
        #         # for triangle meshes
        #         assert u3d._mesh_primitive_modes[prim.mode]=='TRIANGLES'

        #         # grab all attributes
        #         attributes = {
        #             k: gltf_accessor(gltf, v)
        #             for k,v in json.loads(prim.attributes.to_json()).items()
        #             if v is not None and k in ['JOINTS_0', 'WEIGHTS_0', ]
        #             # if v is not None
        #         }
        #         # indices = gltf_accessor(gltf, prim.indices)

        #         # # materials + textures
        #         # material = gltf.materials[prim.material]
        #         # texture = gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index]
        #         # texture_img = gltf_image(gltf, texture.source)
        #         # texture_set = material.pbrMetallicRoughness.baseColorTexture.texCoord
        #         # timgs.append(texture_img)
        #         # try:
        #         #     bc = material.pbrMetallicRoughness.baseColorFactor
        #         # except:
        #         #     bc = [1,1,1,1]
                
        #         # skin weights
        #         swj = attributes['JOINTS_0']
        #         sww = attributes['WEIGHTS_0']
        #         # swj = gltf_accessor(gltf, prim.attributes.JOINTS_0)
        #         # sww = gltf_accessor(gltf, prim.attributes.WEIGHTS_0)
        #         # if 'JOINTS_1' in attributes and 'WEIGHTS_1' in attributes:
        #         #     swj = np.concatenate([
        #         #         swj, gltf_accessor(gltf, prim.attributes.JOINTS_1),
        #         #     ], axis=1)
        #         #     sww = np.concatenate([
        #         #         sww, gltf_accessor(gltf, prim.attributes.WEIGHTS_1),
        #         #     ], axis=1)
        #         sww = sww / sww.sum(axis=1, keepdims=True)
        #         # sw = np.zeros((len(attributes['POSITION']), len(g_nodes_joint)))
        #         # sw[np.tile(np.arange(len(sw)),(swj.shape[1],1)).T, swj] = sww
        #         # sw = sw / sw.sum(axis=1, keepdims=True)

        #         # saving
        #         # verts = attributes['POSITION']
        #         # norms = attributes['NORMAL']
        #         # faces = indices.reshape(-1, 3) + vc
        #         # uvmap = attributes[f'TEXCOORD_{texture_set}']
        #         # uvcol = sample_texture_uv(texture_img, uvmap)
        #         # _verts.append(verts)
        #         # _norms.append(norms)
        #         # _faces.append(faces)
        #         # _uvcol.append(uvcol)
        #         # _uvmap.append(uvmap)
        #         # _texidxs.append(tc * np.ones(len(verts), dtype=int))
        #         # _basecol.append(bc)
        #         _swj.append(swj)
        #         _sww.append(sww)
        #         # vc += len(verts)
        #         tc += 1
        #     #     break
        #     # break
        # self.verts = np.concatenate(_verts)
        # self.faces = np.concatenate(_faces)
        # self.normals = np.concatenate(_norms)
        # self.uv_colors = np.concatenate([i[:,:3] for i in _uvcol])
        # self.uv_map = np.concatenate(_uvmap)
        # self.texture_idxs = np.concatenate(_texidxs)
        # self.textures = timgs
        # self.base_colors = np.asarray(_basecol)
        # self.skin_weights_idxs = np.concatenate(_swj)
        # self.skin_weights_vals = np.concatenate(_sww)
        return
    
    # mesh adjustment
    def remove_innards(self, n=1, thresh=1.3):
        (
            self.verts,
            self.faces,
            self.normals,
            self.uv_colors,
            self.uv_map,
            self.texture_idxs,
        ) = remove_innards(
            self.verts,
            self.faces,
            self.normals,
            self.uv_colors,
            self.uv_map,
            self.texture_idxs,
            n=n, thresh=thresh,
        )
        return self


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        tbs = traceback.format_exc()
        write(f'{isonow()}\n{tbs}', mkfile(f'{odn}/{render_dtype}/_logs/fails/{bn}.txt'))
        exit(0)

    write(isonow(), mkfile(f'{odn}/{render_dtype}/_logs/succ/{bn}.txt'))

    print(time.time()-TIME)

