import multiprocessing
import time
import traceback
import uuid

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from util.function import (SetForegroundWindow, communicate_Thread_callFun,
                           vType)


class base_showbyopen3d():
    def __init__(self, queueSend: multiprocessing.Queue, queueRecive: multiprocessing.Queue, connectFun_queueRecive, _data_manager: multiprocessing.Manager, winName='__open3d_win__') -> None:
        self.send = queueSend.put
        self.recive = queueRecive.get
        self._data_manager = _data_manager
        self.winName = winName

        self.args_init()

        self._threadCommunicateFromOpen3d = communicate_Thread_callFun(
            queueRecive, connectFun_queueRecive)
        self._threadCommunicateFromOpen3d.start()

        gui.Application.instance.initialize()
        self.open3d_init()

        self.setWim = SetForegroundWindow()

        gui.Application.instance.post_to_main_thread(self.window, self.setWinTop)

        self.running = True

    def run(self):
        while self.running:
            try:
                gui.Application.instance.run_one_tick()
            except Exception:
                traceback.print_exc()
                continue

        gui.Application.instance.quit()
        gui.Application.instance.run_one_tick()

    def args_init(self):
        self.geometryRawCache = {}    # {uuid: {raw:..., box:..., objType:...}} raw为open3d的原始数据，无法加入到manager             # type, name, path, raw, orignColor
        self.flag_reflashFinish = True

        self.selectgeometryUuid = None

        self.mouseMoveLast_x = None
        self.mouseMoveLast_y = None

        self.isShowCoordinate = True

        self.default_pointSize = 5

    def open3d_init(self):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.window = gui.Application.instance.create_window(
            self.winName, 1500, 1000, 50, 50)
        # Since we want the label on top of the scene, we cannot use a layout,
        # so we need to manually layout the window's children.
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)
        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        self.new_ibl_name = gui.Application.instance.resource_path + "/" + "default"
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.show_skybox(False)                      # 开天空盒
        self.widget3d.scene.scene.enable_sun_light(True)
        self.widget3d.scene.scene.set_indirect_light(self.new_ibl_name)
        self.widget3d.scene.scene.enable_indirect_light(True)
        self.widget3d.scene.scene.set_indirect_light_intensity(45000)

        self.enable_basic_mode(True)
        # self.widget3d.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, (0.577, -0.577, -0.577))

        self.widget3d.scene.update_material(self.creartMaterial())
        # self.widget3d.scene.show_axes(True)

        self.sceneCamera = self.widget3d.scene.camera
        # print(self.sceneCamera.get_model_matrix())
        # print(self.sceneCamera.get_projection_matrix())
        # print(self.sceneCamera.get_view_matrix())

        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        self.widget3d.look_at(center, center - [0, 0, 3], [0, 1, 0])

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

        self.create_coordinate()

    def close(self):
        self.running = False

###############################################################################
# open3d draw Function
###############################################################################
    def setWinTop(self):
        self.setWim.setWin(self.winName)
        time.sleep(0.01)

    def showGeometry(self, objUuid_show):
        name = str(objUuid_show[0])
        if self.widget3d.scene.geometry_is_visible(name) != objUuid_show[1]:
            self._show_geometry(name, objUuid_show[1])
        self.setWinTop()

    def selectGeometry(self, objUuid: list):
        self.selectgeometryUuid = None
        for uid, raw in self.geometryRawCache.items():
            name = str(uid)
            if self.widget3d.scene.geometry_is_visible(name):
                if uid in objUuid:
                    self.selectgeometryUuid = uid
                    if self.widget3d.scene.geometry_is_visible(name + '_box') is False:
                        self.scale_coordinate = self.calculate_coordinateScale(self.coordinate[vType.type.raw], raw[vType.type.raw])
                        self.center_coordinate = raw[vType.type.raw].get_center()
                        self._select_geometry(name + '_box', True)
                else:
                    if self.widget3d.scene.geometry_is_visible(name + '_box') is True:
                        self._select_geometry(name + '_box', False)
        self.setWinTop()

    def addGeometry_byFile(self, uid, show=False):
        filepath = self._data_manager[vType.data.geometryInfoCache][uid][vType.type.path]
        modeType = self._data_manager[vType.data.geometryInfoCache][uid][vType.type.objType]
        raw = None

        try:
            if modeType == vType.type.mesh:
                raw = o3d.io.read_triangle_mesh(filepath)
                if np.asarray(raw.vertices).shape[0] == 0:
                    raw = None
            else:
                file_type = filepath.split('.')[-1]
                if file_type == 'ply':
                    raw = o3d.io.read_point_cloud(filepath)
                    if np.asarray(raw.points).shape[0] == 0:
                        raw = None
                elif file_type == 'txt':
                    data_input = np.loadtxt(filepath).astype(np.float32)
                    if data_input.shape[1] >= 6:
                        raw = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(data_input[:, :3]))
                        colors = data_input[:, 3:6]
                        if np.min(colors) < 0:
                            colors = (colors + 1)/2
                        elif np.max(colors) > 1.2:
                            colors /= 255
                        raw.colors = o3d.utility.Vector3dVector(colors)
                elif file_type == 'pth':
                    pass
                    # data_input = torch.load(filepath)
                    # xyz, colors = data_input[0], data_input[1]
                    # raw = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
                    # if np.min(colors) < 0:
                    #     colors = (colors + 1)/2
                    # elif np.max(colors) > 1.2:
                    #     colors /= 255
                    # raw.colors = o3d.utility.Vector3dVector(colors)
        except Exception as e:
            self.send([vType.log.log, str(e), vType.log.error])

        if raw is not None:
            self.addGeometry_byRaw(
                uid=uid,
                raw=raw,
                show=show
            )
        return raw

    def addGeometry_byRaw(self, uid, raw, show=False):
        # geometryBox = raw.get_oriented_bounding_box()
        geometryBox = raw.get_axis_aligned_bounding_box()
        geometryBox.color = (1, 0, 0)
        modeType = self._data_manager[vType.data.geometryInfoCache][uid][vType.type.objType]
        self.geometryRawCache.update({uid: {vType.type.raw: raw, vType.type.box: geometryBox, vType.type.objType: modeType}})
        self._add_geometry(str(uid), raw, modeType, show, geometryBox)
        if len(self.geometryRawCache) == 1:
            self.cameraReset()
        self.setWinTop()

    def creartMaterial(self, type=vType.type.pointCloud):
        mat = rendering.MaterialRecord()
        mat.base_color = [1.0, 1.0, 1.0, 1.0]

        if type == vType.type.pointCloud:
            mat.shader = "defaultUnlit"           # defaultLit, defaultUnlit, unlitLine, unlitGradient, unlitSolidColor
            mat.point_size = self.default_pointSize * self.window.scaling
        elif type == vType.type.mesh:
            mat.shader = 'defaultLitTransparency'
        elif type == vType.type.line or type == vType.type.box:
            mat.shader = 'unlitLine'
            mat.line_width = 2
        return mat

    def cameraReset(self):
        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)

    def create_coordinate(self):
        raw = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        name = '_coordinate'
        self.coordinate = {vType.type.objName: name, vType.type.raw: raw}
        mat = self.creartMaterial(vType.type.mesh)

        self.widget3d.scene.add_geometry(name, raw, mat)
        self.widget3d.scene.show_geometry(name, False)

    def show_coordinate(self, show=True):
        if self.isShowCoordinate and show:
            self.coordinate[vType.type.raw] = self.coordinate[vType.type.raw].translate(self.center_coordinate, False)
            self.coordinate[vType.type.raw] = self.coordinate[vType.type.raw].scale(self.scale_coordinate, self.center_coordinate)
            self.scale_coordinate = 1
            self.widget3d.scene.remove_geometry(self.coordinate[vType.type.objName])
            mat = self.creartMaterial(vType.type.mesh)
            self.widget3d.scene.add_geometry(self.coordinate[vType.type.objName], self.coordinate[vType.type.raw], mat)
            self.widget3d.scene.show_geometry(self.coordinate[vType.type.objName], True)
        else:
            self.widget3d.scene.show_geometry(self.coordinate[vType.type.objName], False)

    def calculate_coordinateScale(self, coordinate_raw, geometry_raw):
        coordinate_box_points = np.asarray(coordinate_raw.get_oriented_bounding_box().get_box_points())
        geometry_box_points = np.asarray(geometry_raw.get_oriented_bounding_box().get_box_points())

        def fun(boxp):
            extend = set()
            for i, bp in enumerate(boxp):
                for bp0 in np.delete(boxp, i, axis=0):
                    re = '%.10f' % np.sum((bp-bp0)**2)**(1/2)
                    extend.add(float(re))

            return np.asarray(sorted(extend)[:-1]).mean()

        return fun(geometry_box_points) / fun(coordinate_box_points) * 0.4

    def enable_basic_mode(self, enable):
        self.is_enable_basic_mode = enable
        if enable:
            self.widget3d.scene.scene.enable_indirect_light(False)
            self.widget3d.scene.scene.enable_sun_light(True)
            self.widget3d.scene.scene.set_sun_light(
                -self.widget3d.scene.camera.get_view_matrix()[2, :-1].reshape(3),
                [1., 1., 1.],
                160000.0
            )
            self.widget3d.scene.view.set_shadowing(False, o3d.visualization.rendering.View.ShadowType.PCF)
            self.widget3d.scene.view.set_post_processing(False)
        else:
            self.widget3d.scene.scene.enable_indirect_light(True)
            self.widget3d.scene.scene.enable_sun_light(True)
            self.widget3d.scene.scene.set_sun_light(
                [0.577, -0.577, -0.577],
                [1., 1., 1.],
                45000.0
            )
            self.widget3d.scene.view.set_shadowing(True, o3d.visualization.rendering.View.ShadowType.PCF)
            self.widget3d.scene.view.set_post_processing(True)

###############################################################################
# open3d callback Function
###############################################################################
    def _show_geometry(self, geometryName, isShow):
        self.widget3d.scene.show_geometry(geometryName, isShow)
        self.widget3d.scene.show_geometry(geometryName + '_box', False)
        self.show_coordinate(False)

    def _select_geometry(self, geometryName, isSelect):
        self.widget3d.scene.show_geometry(geometryName, isSelect)
        self.show_coordinate(self.selectgeometryUuid is not None)

    def _add_geometry(self, geometryName, geometryRaw, geometryType, isShow, geometryBox):
        print('>>>', geometryName)

        mat = self.creartMaterial(geometryType)

        self.widget3d.scene.add_geometry(
            geometryName, geometryRaw, mat)
        self.widget3d.scene.show_geometry(
            geometryName, isShow)

        mat = self.creartMaterial(vType.type.box)

        self.widget3d.scene.add_geometry(
            geometryName + '_box', geometryBox, mat)
        self.widget3d.scene.show_geometry(
            geometryName + '_box', False)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height,
                                   pref.width,
                                   pref.height)

    def _on_mouse_widget3d(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                else:
                    world = self.widget3d.scene.camera.unproject(
                        event.x, event.y, depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and (event.is_button_down(gui.MouseButton.LEFT) or event.is_button_down(gui.MouseButton.RIGHT)) and event.is_modifier_down(gui.KeyModifier.SHIFT):
            self.mouseMoveLast_x = event.x
            self.mouseMoveLast_y = event.y
            return gui.Widget.EventCallbackResult.CONSUMED
        elif (event.type == gui.MouseEvent.Type.DRAG and (event.is_button_down(gui.MouseButton.LEFT) or event.is_button_down(gui.MouseButton.RIGHT))) or (event.type == gui.MouseEvent.Type.WHEEL):

            if self.is_enable_basic_mode:
                self.widget3d.scene.scene.set_sun_light(
                    -self.widget3d.scene.camera.get_view_matrix()[2, :-1].reshape(3),
                    [1., 1., 1.],
                    160000
                )

            if not event.is_modifier_down(gui.KeyModifier.SHIFT):
                return gui.Widget.EventCallbackResult.HANDLED

            if self.selectgeometryUuid is not None:
                raw = self.geometryRawCache[self.selectgeometryUuid][vType.type.raw]
                box = self.geometryRawCache[self.selectgeometryUuid][vType.type.box]
                center = box.get_center()
                cameraRot = self.sceneCamera.get_model_matrix()[:3, :3]

                if event.type == gui.MouseEvent.Type.WHEEL:             # 滑轮缩放
                    if event.wheel_dy > 0:
                        raw = raw.scale(0.9, center)
                    else:
                        raw = raw.scale(1.1, center)
                else:
                    if self.mouseMoveLast_x is None or self.mouseMoveLast_y is None:
                        self.mouseMoveLast_x, self.mouseMoveLast_y = event.x, event.y
                    x = event.x - self.mouseMoveLast_x
                    y = event.y - self.mouseMoveLast_y
                    self.mouseMoveLast_x = event.x
                    self.mouseMoveLast_y = event.y

                    if event.is_button_down(gui.MouseButton.LEFT):      # 左键旋转
                        rot = raw.get_rotation_matrix_from_axis_angle(np.matmul(cameraRot, np.asarray([y*0.005, x*0.005, 0])))
                        raw = raw.rotate(rot, center)
                    elif event.is_button_down(gui.MouseButton.RIGHT):   # 右键平移
                        raw = raw.translate(np.matmul(cameraRot, np.asarray([x*0.05, -y*0.05, 0])))

                name = str(self.selectgeometryUuid)
                self.geometryRawCache[self.selectgeometryUuid][vType.type.raw] = raw
                # box = raw.get_oriented_bounding_box()
                box = raw.get_axis_aligned_bounding_box()
                box.color = (1, 0, 0)
                self.geometryRawCache[self.selectgeometryUuid][vType.type.box] = box
                self.widget3d.scene.remove_geometry(name)
                self.widget3d.scene.remove_geometry(name + '_box')
                mat = self.creartMaterial(self.geometryRawCache[self.selectgeometryUuid][vType.type.objType])
                self.widget3d.scene.add_geometry(name, raw, mat)
                mat = self.creartMaterial(vType.type.box)
                self.widget3d.scene.add_geometry(name + '_box', box, mat)

                self.reflash_data_manager_geometry_raw(self.selectgeometryUuid, raw)

                self.scale_coordinate = self.calculate_coordinateScale(self.coordinate[vType.type.raw], raw)
                self.center_coordinate = box.get_center()
                self.show_coordinate(True)

            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.IGNORED

###############################################################################################
# open3d Other Function
###############################################################################################
    def get_uuid(self):
        while True:
            new_key = uuid.uuid4()
            if new_key not in self._data_manager[vType.data._uuidSet]:
                self._data_manager[vType.data._uuidSet].add(new_key)
                break
        return new_key

    def reflash_data_manager_geometry_raw(self, uid, raw):
        modeType = self._data_manager[vType.data.geometryInfoCache][uid][vType.type.objType]
        if modeType == vType.type.mesh:
            out_raw = (np.asarray(raw.vertices), np.asarray(raw.vertex_colors), np.asarray(raw.triangles), np.asarray(raw.vertex_normals))
        else:
            out_raw = (np.asarray(raw.points), np.asarray(raw.colors))

        gic = self._data_manager[vType.data.geometryInfoCache]
        gic[uid][vType.type.raw] = out_raw
        self._data_manager.update({vType.data.geometryInfoCache: gic})
