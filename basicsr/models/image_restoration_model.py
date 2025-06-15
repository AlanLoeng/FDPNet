# ------------------------------------------------------------------------
# Copyright (c) 2025 Bolun Liang(https://github.com/AlanLoeng) All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import os 
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)
        
        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        
        # Add support for mixed precision training
        self.use_amp = opt.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # load pretrained models
        # ... rest of the function remains the same

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()
        
        if self.opt['train'].get('mixup', False):
            self.mixup_aug()
        
        # Modify the forward pass to use autocast for mixed precision
        if self.use_amp:
            with torch.cuda.amp.autocast():
                preds = self.net_g(self.lq)
                if not isinstance(preds, list):
                    preds = [preds]
                
                self.output = preds[-1]
                
                l_total = 0
                loss_dict = OrderedDict()
                # pixel loss
                if self.cri_pix:
                    l_pix = 0.
                    for pred in preds:
                        l_pix += self.cri_pix(pred, self.gt)
                    
                    l_total += l_pix
                    loss_dict['l_pix'] = l_pix
                
                # perceptual loss
                if self.cri_perceptual:
                    l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                    
                    if l_percep is not None:
                        l_total += l_percep
                        loss_dict['l_percep'] = l_percep
                    if l_style is not None:
                        l_total += l_style
                        loss_dict['l_style'] = l_style
                
                l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
            
            # Use scaler for backward and optimization steps
            self.scaler.scale(l_total).backward()
            
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                self.scaler.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            # Original non-AMP code
            preds = self.net_g(self.lq)
            if not isinstance(preds, list):
                preds = [preds]
            
            self.output = preds[-1]
            
            l_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            if self.cri_pix:
                l_pix = 0.
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)
                
                l_total += l_pix
                loss_dict['l_pix'] = l_pix
            
            # perceptual loss
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output, self.gt)
                
                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep
                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style
            
            l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())
            
            l_total.backward()
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
            self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                # Use autocast for inference as well if AMP is enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.net_g(self.lq[i:j])
                else:
                    pred = self.net_g(self.lq[i:j])
                    
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j
            
            self.output = torch.cat(outs, dim=0)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}.png')
                        save_gt_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_gt.png')

                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    # def nondist_validation(self, *args, **kwargs):
    #     logger = get_root_logger()
    #     logger.warning('nondist_validation is not implemented. Run dist_validation.')
    #     self.dist_validation(*args, **kwargs)
    # def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
    #     dataset_name = dataloader.dataset.opt['name']
    #     with_metrics = self.opt['val'].get('metrics') is not None

    #     if with_metrics:
    #         self.metric_results = {
    #             metric: 0
    #             for metric in self.opt['val']['metrics'].keys()
    #         }

    #     pbar = tqdm(total=len(dataloader), unit='image')

    #     cnt = 0

    #     for idx, val_data in enumerate(dataloader):
    #         img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

    #         self.feed_data(val_data, is_val=True)
    #         if self.opt['val'].get('grids', False):
    #             self.grids()

    #         self.test()

    #         if self.opt['val'].get('grids', False):
    #             self.grids_inverse()

    #         visuals = self.get_current_visuals()
    #         sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
    #         if 'gt' in visuals:
    #             gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
    #             del self.gt

    #         # 释放内存
    #         del self.lq
    #         del self.output
    #         torch.cuda.empty_cache()

    #         if save_img:
    #             if sr_img.shape[2] == 6:
    #                 L_img = sr_img[:, :, :3]
    #                 R_img = sr_img[:, :, 3:]
    #                 visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
    #                 imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
    #                 imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
    #             else:
    #                 save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{current_iter}.png')
    #                 save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{current_iter}_gt.png')
    #                 imwrite(sr_img, save_img_path)
    #                 imwrite(gt_img, save_gt_img_path)

    #         if with_metrics:
    #             opt_metric = deepcopy(self.opt['val']['metrics'])
    #             if use_image:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
    #             else:
    #                 for name, opt_ in opt_metric.items():
    #                     metric_type = opt_.pop('type')
    #                     self.metric_results[name] += getattr(metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

    #         cnt += 1
    #         pbar.update(1)
    #         pbar.set_description(f'Test {img_name}')

    #     pbar.close()

    #     collected_metrics = OrderedDict()
    #     if with_metrics:
    #         for metric in self.metric_results.keys():
    #             collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
    #         collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

    #         self.collected_metrics = collected_metrics

    #     # 计算并记录平均指标
    #     keys = []
    #     metrics = []
    #     for name, value in self.collected_metrics.items():
    #         keys.append(name)
    #         metrics.append(value)

    #     metrics = torch.stack(metrics, 0)

    #     metrics_dict = {}
    #     cnt = 0
    #     for key, metric in zip(keys, metrics):
    #         if key == 'cnt':
    #             cnt = float(metric)
    #             continue
    #         metrics_dict[key] = float(metric)

    #     for key in metrics_dict:
    #         metrics_dict[key] /= cnt

    #     self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metrics_dict)

    #     return 0

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        self.opt['val']['metrics'] = self.opt['val'].get('metrics', {}) # Ensure metrics dict exists

        # --- New: List to store results for top SSIM tracking ---
        # Store tuples: (ssim_score, sr_img, gt_img, img_name, current_iter)
        # We store images directly, ensure enough memory. Use .copy()
        top_ssim_candidates = []
        # --- End New ---

        # --- New: Define the directory for top SSIM results ---
        top_ssim_dir = osp.join(self.opt['path']['visualization'], f'{dataset_name}_top500_ssim_{current_iter}')
        if save_img: # Only create if we are saving images
            os.makedirs(top_ssim_dir, exist_ok=True)
        # --- End New ---


        if with_metrics:
            # Initialize metric sums
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
            # Ensure 'ssim' is requested if we want to track top SSIM
            if 'ssim' not in self.opt['val']['metrics']:
                 # Option 1: Add SSIM calculation if not present (recommended)
                 print("Warning: 'ssim' not found in validation metrics. Adding default SSIM calculation for top-k.")
                 self.opt['val']['metrics']['ssim'] = {'type': 'calculate_ssim'} # Adjust 'type' if needed
                 self.metric_results['ssim'] = 0
                 # Option 2: Raise an error or skip top-k saving
                 # raise ValueError("SSIM metric must be enabled in validation options to save top SSIM images.")


        pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test() # Generates self.output

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            visuals = self.get_current_visuals() # Should contain 'result' and potentially 'gt'
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr) # Convert SR tensor to image (e.g., numpy array)

            current_metrics = {} # To store metrics for the *current* image

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr) # Convert GT tensor to image

                # Calculate metrics for the current image pair
                if with_metrics:
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        # Decide whether to use image (numpy) or tensor for calculation
                        if use_image:
                            # Assuming metric_module functions work with numpy arrays
                            metric_func = getattr(metric_module, metric_type)
                            metric_result = metric_func(sr_img, gt_img, **opt_)
                        else:
                            # Assuming metric_module functions work with tensors
                            metric_func = getattr(metric_module, metric_type)
                            metric_result = metric_func(visuals['result'], visuals['gt'], **opt_)

                        # Add to the overall sum for average calculation
                        self.metric_results[name] += metric_result
                        # Store the individual metric for this image
                        current_metrics[name] = metric_result

                # --- New: Track SSIM score for top 50 ---
                if 'ssim' in current_metrics:
                    ssim_score = current_metrics['ssim']
                    # Store necessary info. Use .copy() for numpy arrays to prevent issues
                    # if sr_img/gt_img are modified/deleted later.
                    top_ssim_candidates.append((ssim_score, sr_img.copy(), gt_img.copy(), img_name, current_iter))
                elif 'ssim' in self.opt['val']['metrics'] and 'gt' in visuals:
                    # If SSIM wasn't calculated above but is requested, calculate it now
                    # This might happen if 'use_image' logic differs or SSIM is handled specially
                    try:
                        ssim_opts = deepcopy(self.opt['val']['metrics']['ssim'])
                        ssim_type = ssim_opts.pop('type', 'calculate_ssim') # Get type, provide default
                        ssim_func = getattr(metric_module, ssim_type)
                        # Decide based on function signature or config if it needs tensors or images
                        if use_image: # Or check specific config for SSIM
                           ssim_score = ssim_func(sr_img, gt_img, **ssim_opts)
                        else:
                           ssim_score = ssim_func(visuals['result'], visuals['gt'], **ssim_opts)

                        self.metric_results['ssim'] += ssim_score # Add to sum
                        current_metrics['ssim'] = ssim_score # Store individual
                        top_ssim_candidates.append((ssim_score, sr_img.copy(), gt_img.copy(), img_name, current_iter))
                    except Exception as e:
                        print(f"Warning: Could not calculate SSIM for {img_name}. Error: {e}")

                # --- End New ---

                # Standard image saving (if enabled)
                if save_img:
                    if sr_img.shape[2] == 6: # Special case for stereo?
                        L_img = sr_img[:, :, :3]
                        R_img = sr_img[:, :, 3:]
                        visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                        # Create visual_dir if it doesn't exist (important!)
                        os.makedirs(visual_dir, exist_ok=True)
                        imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                        imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                    else:
                        # Normal save path
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{current_iter}_SR.png')
                        save_gt_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{current_iter}_GT.png')
                         # Create visual_dir if it doesn't exist (important!)
                        os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                        imwrite(sr_img, save_img_path)
                        # Only save GT if it exists
                        imwrite(gt_img, save_gt_img_path)

                del self.gt # Release GT tensor memory
                del gt_img # Release GT image memory (if no longer needed)

            else: # No GT image
                 if save_img: # Save only SR image
                    if sr_img.shape[2] == 6: # Special case for stereo?
                       # Handle saving stereo SR without GT as needed
                       pass # Add logic if required
                    else:
                       save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_{current_iter}_SR.png')
                       # Create visual_dir if it doesn't exist (important!)
                       os.makedirs(osp.dirname(save_img_path), exist_ok=True)
                       imwrite(sr_img, save_img_path)


            # Release memory
            del self.lq
            del self.output
            if 'result' in visuals: del visuals['result']
            if 'gt' in visuals: del visuals['gt'] # Already deleted above, but good practice
            del sr_img # Release SR image memory
            torch.cuda.empty_cache()

            cnt += 1
            pbar.update(1)
            # Display the first metric found (e.g., PSNR or SSIM) in pbar if available
            metric_str = ""
            if current_metrics:
                first_metric_name = next(iter(current_metrics))
                metric_str = f" {first_metric_name}: {current_metrics[first_metric_name]:.4f}"
            pbar.set_description(f'Test {img_name}{metric_str}')

        pbar.close()

        # --- New: Sort candidates and save top 50 ---
        if save_img and top_ssim_candidates: # Only proceed if saving images and we have candidates
            print(f"Sorting {len(top_ssim_candidates)} images by SSIM score...")
            # Sort by SSIM score (the first element of the tuple), descending
            top_ssim_candidates.sort(key=lambda x: x[0], reverse=True)

            print(f"Saving top {min(500, len(top_ssim_candidates))} images with highest SSIM to {top_ssim_dir}")
            for i, (ssim_score, result_img_copy, gt_img_copy, img_name, c_iter) in enumerate(top_ssim_candidates[:500]):
                # Use descriptive filenames including SSIM score
                top_result_path = osp.join(top_ssim_dir, f'{i+1:02d}_{img_name}_{c_iter}_ssim{ssim_score:.4f}_Result.png')
                top_gt_path = osp.join(top_ssim_dir, f'{i+1:02d}_{img_name}_{c_iter}_ssim{ssim_score:.4f}_GT.png')

                imwrite(result_img_copy, top_result_path)
                imwrite(gt_img_copy, top_gt_path)
            print("Finished saving top SSIM images.")
        # --- End New ---

        collected_metrics = OrderedDict()
        if with_metrics:
            # This part aggregates results across ranks in distributed training.
            # For non-distributed, it just converts sums to tensors.
            for metric in self.metric_results.keys():
                # Ensure the value is a tensor before sending to device
                value = self.metric_results[metric]
                if not isinstance(value, torch.Tensor):
                     value = torch.tensor(value)
                collected_metrics[metric] = value.float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics # Store collected metrics (sums)

            # Calculate and log average metrics
            # This part seems okay, it calculates averages from the collected sums
            keys = []
            metrics_sum = [] # Renamed from metrics to avoid confusion
            total_count = 0.0

            for name, value in self.collected_metrics.items():
                if name != 'cnt':
                    keys.append(name)
                    metrics_sum.append(value)
                else:
                    total_count = float(value) # Get the total count

            if total_count > 0 :
                 metrics_avg_dict = {}
                 for key, metric_sum in zip(keys, metrics_sum):
                      metrics_avg_dict[key] = float(metric_sum) / total_count # Calculate average

                 # Log the average values
                 self._log_validation_metric_values(current_iter, dataset_name, tb_logger, metrics_avg_dict)
            else:
                 print("Warning: Count is zero, cannot calculate average metrics.")


        # Original code returns 0, maintaining that unless specific logic needed
        return 0


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
