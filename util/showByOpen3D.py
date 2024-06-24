import multiprocessing
import sys
import numpy as np
import open3d as o3d

from util.baseShowByOpen3D import base_showbyopen3d
from util.function import vType


class showbyopen3d(base_showbyopen3d):

    def __init__(self, queueSend: multiprocessing.Queue, queueRecive: multiprocessing.Queue, _data_manager: multiprocessing.Manager, winName='__open3d_win__') -> None:
        super().__init__(queueSend, queueRecive, self.connectFun_queueRecive, _data_manager, winName)

        self.run()
        print('out')
        sys.exit(0)

    def args_init(self):
        super().args_init()

###############################################################################
# appMain Connect Function
###############################################################################
    def connectFun_queueRecive(self, info: list):
        vt = info[0]

        if vt == vType.all.close:
            self.close()

        # geometry
        elif vt == vType.geometry.add:
            self.send([vType.widgets.progressBar, True])
            add_uuid = info[1]
            for id in add_uuid:
                if id in self._data_manager[vType.data.geometryInfoCache].keys():
                    raw = self.addGeometry_byFile(id, True)
                    filepath = self._data_manager[vType.data.geometryInfoCache][id][vType.type.path]
                    if raw is not None:
                        self.reflash_data_manager_geometry_raw(id, raw)
                        self.send([vType.geometry.add, id])
                        self.send([vType.log.log, f'load success [{filepath}]', vType.log.info])
                    else:
                        self.send([vType.log.log, f'load failed [{filepath}]', vType.log.error])

            self.send([vType.widgets.progressBar, False])

        # geometry_mesh
        elif vt == vType.geometry.show:
            self.showGeometry(info[1])
        elif vt == vType.geometry.select:
            self.selectGeometry(info[1])
        # elif vt == vType.geometry_mesh.save:
        #     names = info[2]
        #     meshAll = None
        #     for mesh in self.geometryRawCache:
        #         if mesh[vType.type.objName] in names:
        #             if meshAll is None:
        #                 meshAll = mesh[vType.type.raw]
        #             else:
        #                 meshAll = meshAll + mesh[vType.type.raw]
        #     if meshAll is not None:
        #         o3d.io.write_triangle_mesh(info[1], meshAll, write_ascii=True)

        # geometry_pointcloud
        # elif vt == vType.geometry_pointcloud.save:
        #     names = info[2]
        #     pointCloudAll = None
        #     for pointCloud in self.geometryRawCache:
        #         if pointCloud[vType.type.objName] in names:
        #             if pointCloudAll is None:
        #                 pointCloudAll = pointCloud[vType.type.raw]
        #             else:
        #                 pointCloudAll = pointCloudAll + pointCloud[vType.type.raw]
        #     if pointCloudAll is not None:
        #         o3d.io.write_point_cloud(info[1], pointCloudAll, write_ascii=True)

        # sence
        elif vt == vType.sence.cameraReset:
            self.cameraReset()
        elif vt == vType.sence.showAxis:
            self.isShowCoordinate = info[1]
            self.show_coordinate()
        elif vt == vType.sence.showSkybox:
            self.widget3d.scene.show_skybox(info[1])
        elif vt == vType.sence.showGroundPlane:
            self.widget3d.scene.show_ground_plane(info[1], self.widget3d.scene.scene.GroundPlane.XY)
        elif vt == vType.sence.addPointSize:
            self.default_pointSize += 0.5
            self.default_pointSize = np.clip(self.default_pointSize, 1, 10)
            for uid, geo in self.geometryRawCache.items():
                if geo[vType.type.objType] == vType.type.pointCloud:
                    self.widget3d.scene.modify_geometry_material(str(uid), self.creartMaterial())
        elif vt == vType.sence.dePointSize:
            self.default_pointSize -= 0.5
            self.default_pointSize = np.clip(self.default_pointSize, 1, 10)
            for uid, geo in self.geometryRawCache.items():
                if geo[vType.type.objType] == vType.type.pointCloud:
                    self.widget3d.scene.modify_geometry_material(str(uid), self.creartMaterial())
        elif vt == vType.sence.rawMode:
            self.enable_basic_mode(info[1])
