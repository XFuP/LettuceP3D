import argparse
import glob
import os
import numpy as np
import spconv.pytorch as spconv
import torch
import yaml
from munch import Munch
from tqdm import tqdm
from softgroup.data import build_dataloader, custom
from softgroup.model import SoftGroup
from softgroup.ops import voxelization, voxelization_idx
from softgroup.util import utils 
from softgroup.util import (collect_results_gpu, cuda_cast, rle_decode)
import tempfile
import open3d as o3d

COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        0.876, 0.000, 0.000,
        0.876, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 0.876, 0.000,
        0.000, 0.000, 0.876,
        0.667, 0.000, 0.876,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 0.876, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 0.876, 0.000,
        0.876, 0.333, 0.000,
        0.876, 0.667, 0.000,
        0.876, 0.876, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 0.876, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 0.876, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 0.876, 0.500,
        0.876, 0.000, 0.500,
        0.876, 0.333, 0.500,
        0.876, 0.667, 0.500,
        0.876, 0.876, 0.500,
        0.000, 0.333, 0.876,
        0.000, 0.667, 0.876,
        0.000, 0.876, 0.876,
        0.333, 0.000, 0.876,
        0.333, 0.333, 0.876,
        0.333, 0.667, 0.876,
        0.333, 0.876, 0.876,
        0.667, 0.000, 0.876,
        0.667, 0.333, 0.876,
        0.667, 0.667, 0.876,
        0.667, 0.876, 0.876,
        0.876, 0.000, 0.876,
        0.876, 0.333, 0.876,
        0.876, 0.667, 0.876,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        0.876, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 0.876, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 0.876,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 0.876, 0.876, 0.876
    ]).astype(np.float32).reshape(-1, 3) * 255

SEMANTIC_IDXS = np.array([1, 2])
SEMANTIC_NAMES = np.array(['ground', 'leaf'])
CLASS_COLOR = {'ground': [0, 0, 0],
               'leaf': [143, 223, 142]}
SEMANTIC_IDX2NAME = {1: 'ground',
                     2: 'leaf'}

class shengcaiDataset(custom.CustomDataset):

    CLASSES = ('groud', 'leaf')

    def __init__(self, data_root, prefix, suffix, voxel_cfg=None, training=False, repeat=1, logger=None, files_path: list=None, ):
        self.filenames = files_path

        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.repeat = repeat
        self.mode = 'test'
        # super().__init__(data_root, prefix, suffix, voxel_cfg, training, repeat, logger)

    def load(self, filename):

        data = np.loadtxt(filename)
        xyz, rgb = data[:, :3], data[:, 3:6]

        xyz = np.ascontiguousarray(data[:, :3] - data[:, :3].mean(0))
        rgb = np.ascontiguousarray(data[:, 3:6]) /255 * 2 - 1

        return xyz, rgb

    def get_filenames(self):
        return self.filenames

    def __getitem__(self, index):
        data = self.load(self.filenames[index])
        if data is None:
            return None

        xyz, rgb = data
        xyz_middle = self.dataAugment(xyz, False, False, False)
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)

        coord = torch.from_numpy(xyz).long()             
        coord_float = torch.from_numpy(xyz_middle)      
        feat = torch.from_numpy(rgb).float()
        return (coord, coord_float, feat)

    def collate_fn(self, batch):
        coords = []
        coords_float = []
        feats = []

        batch_id = 0
        for data in batch:
            if data is None:
                continue
            (coord, coord_float, feat) = data
            coords.append(torch.cat([coord.new_full((coord.size(0), 1), batch_id), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            batch_id += 1
        assert batch_id > 0, 'empty batch'
        if batch_id < len(batch):
            print(f'batch is truncated from size {len(batch)} to {batch_id}')

        # merge all the scenes in the batch
        coords = torch.cat(coords, 0)  # long (N, 1 + 3), the batch item idx is put in coords[:, 0]
        batch_idxs = coords[:, 0].int()
        coords_float = torch.cat(coords_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)  # float (N, C)

        spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, self.voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, batch_id)
        return {
            'batch_idxs': batch_idxs,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'coords_float': coords_float,
            'feats': feats,
            'spatial_shape': spatial_shape,
            'batch_size': batch_id,
        }

class softGroupTest(SoftGroup):
    @utils.cuda_cast
    def forward_test(
            self,
            batch_idxs,
            voxel_coords,
            p2v_map,
            v2p_map,
            coords_float,
            feats,
            spatial_shape,
            batch_size,
            **kwargs):
        feats = torch.cat((feats, coords_float), 1)
        voxel_feats = voxelization(feats, p2v_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        semantic_scores, pt_offsets, output_feats = self.forward_backbone(input, v2p_map, x4_split=False)

        semantic_preds = semantic_scores.max(1)[1]
        ret = dict(
            semantic_preds=semantic_preds.cpu().numpy(),
            offset_preds=pt_offsets.cpu().numpy(),
        )
        if not self.semantic_only:
            proposals_idx, proposals_offset = self.forward_grouping(semantic_scores, pt_offsets,
                                                                    batch_idxs, coords_float,
                                                                    self.grouping_cfg)
            inst_feats, inst_map = self.clusters_voxelization(proposals_idx, proposals_offset,
                                                              output_feats, coords_float,
                                                              **self.instance_voxel_cfg)
            _, cls_scores, iou_scores, mask_scores = self.forward_instance(inst_feats, inst_map)
            pred_instances = self.get_instances(0, proposals_idx, semantic_scores,
                                                cls_scores, iou_scores, mask_scores)
            ret.update(dict(pred_instances=pred_instances))
        return ret

def load_checkpoint(checkpoint, model):
    if hasattr(model, 'module'):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    src_state_dict = state_dict['net']
    target_state_dict = model.state_dict()
    skip_keys = []
    # skip mismatch size tensors in case of pretraining
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=False)
    if skip_keys:
        print(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        print(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        print(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

def main_single(file=None, out_path=None,
         path_config='./data/softgroup_shengcai.yaml',
         path_checkpoint='./data/epoch_36.pth',
         return_data=False):
    cfg_txt = open(path_config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    '''model'''
    model = softGroupTest(**cfg.model).cuda()

    '''checkpoint'''
    load_checkpoint(path_checkpoint, model)

    '''dataset'''
    _data_cfg = cfg.data.test.copy()
    _data_cfg['logger'] = None
    _data_cfg.pop('type')
    _data_cfg.pop('with_label')

    _data_cfg['files_path'] = [file]
    dataset = shengcaiDataset(**_data_cfg)

    '''dataloader'''
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)

    results = []

    with torch.no_grad():
        model.eval()
        progress_bar = tqdm(total=len(dataloader), disable=True)

        for i, batch in enumerate(dataloader):
            # try:
            result = model(batch)
            # except Exception as e:
            #     print(f'[Error]: ({i})'+str(e))
            #     continue
            results.append(result)
            progress_bar.update()
        progress_bar.close()

        results = collect_results_gpu(results, len(dataset))
        raw_data = []
        for i, res in enumerate(results):
            data = np.loadtxt(file)
            xyz, color = data[:, :3], data[:, 3:6]
        
            inst_label_pred_rgb = np.zeros(xyz.shape)
            semantic_preds = res['semantic_preds']
            pred_instances = res['pred_instances']
            ins_num = len(pred_instances)

            masks = [mask['conf'] for mask in pred_instances]
            ins_pointnum = np.zeros(ins_num)
            inst_label = -100 * np.ones(inst_label_pred_rgb.shape[0]).astype(np.int32)
            scores = np.array([float(x) for x in masks])
            sort_inds = np.argsort(scores)[::-1]   

            for i_ in range(len(masks) - 1, -1, -1):
                iii = sort_inds[i_]
                mask = rle_decode(pred_instances[iii]['pred_mask'])
                ins_pointnum[iii] = mask.sum()
                inst_label[mask == 1] = iii
            
            sort_idx = np.argsort(ins_pointnum)[::-1]
            for _sort_id in range(ins_num):
                inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[_sort_id % len(COLOR_DETECTRON2)]
                # inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = np.random.randn(3) * 0.6 + 0.2

            sem = -100 * np.ones(xyz.shape[0]).astype(np.int32)
            sem = semantic_preds
            inst = -100 * np.ones(xyz.shape[0]).astype(np.int32)
            inst = inst_label

            if out_path is not None:
                np.savetxt(os.path.join(out_path, os.path.basename(file)), np.c_[xyz, color, inst_label_pred_rgb, sem, inst], fmt="%f %f %f %d %d %d %d %d %d %d %d")
            if return_data:
                raw_data.append((sem, inst))
            continue

            rgb = inst_label_pred_rgb


    if return_data:
        return raw_data


if __name__ == "__main__":
    if True:
        parser = argparse.ArgumentParser('')
        parser.add_argument('--in_path', type=str)
        parser.add_argument('--out_path', type=str)
        args = parser.parse_args()

        file_path = args.in_path
        out_path = args.out_path
    else:
        file_path = ''
        out_path = ''

    os.makedirs(out_path, exist_ok=True)
    files = glob.glob(os.path.join(file_path, '*.txt'))
    for file in tqdm(files):
        main_single(file, out_path)
