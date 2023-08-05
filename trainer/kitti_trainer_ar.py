import time
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .base_trainer import BaseTrainer
from .obj_cache import ObjectCache
from utils.flow_utils import load_flow, evaluate_flow, flow_to_image
from utils.warp_utils import flow_warp
from utils.misc_utils import AverageMeter
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from transforms.ar_transforms.oc_transforms import add_fake_object


class TrainFramework(BaseTrainer):
    def __init__(self, train_loader, valid_loader, model, loss_func,
                 _log, save_root, config, resume=False, train_sets_epoches=[np.inf]):
        super(TrainFramework, self).__init__(
            train_loader, valid_loader, model, loss_func, _log, save_root, config, resume=resume, train_sets_epoches=train_sets_epoches)

        self.sp_transform = RandomAffineFlow(self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise).to(self.device)
        self.car_cache, self.pole_cache = None, None
        
    def set_up_find_obj_mask(self):
        
        def add_dataset_find_obj_mask(obj):
            for ldr in self.train_loaders[self.i_train_set:]:
                if 'ConcatDataset' in str(ldr.dataset.__class__):
                    for ds in ldr.dataset.datasets:
                        ds.find_obj_mask.append(obj)
                else:
                    ldr.dataset.find_obj_mask.append(obj)
            return        
        
        if "car" in self.cfg.ot_classes and self.car_cache is None:
            add_dataset_find_obj_mask("car")
            self.car_cache = ObjectCache(class_name="car", cache_size=100)
        if "pole" in self.cfg.ot_classes and self.pole_cache is None:
            add_dataset_find_obj_mask("pole")
            self.pole_cache = ObjectCache(class_name="pole", cache_size=100)
        return


    def _run_one_epoch(self):

        self.model.train()

        if 'stage1' in self.cfg and self.i_epoch >= self.cfg.stage1.epoch:
            self.loss_func.cfg.update(self.cfg.stage1.loss)
            self.cfg.update(self.cfg.stage1.train)
            if self.cfg.run_ot:
                self.set_up_find_obj_mask()
            self.cfg.pop('stage1')

        if 'stage2' in self.cfg and self.i_epoch >= self.cfg.stage2.epoch:
            self.cfg.update(self.cfg.stage2.train)
            if self.cfg.run_ot:
                self.set_up_find_obj_mask()
            self.cfg.pop('stage2')
            
        am_batch_time, am_data_time = AverageMeter(), AverageMeter()
        key_meter_names = ['Loss', 'l_ph', 'l_sm', 'flow_mean', 'l_atst', 'l_ot']
        key_meters = AverageMeter(i=len(key_meter_names), precision=4)
        
        name_dataset = self.train_loaders[self.i_train_set].dataset.name     
        end = time.time()
        
        for i_step, data in enumerate(self.train_loaders[self.i_train_set]):
            if i_step >= self.cfg.epoch_size:
                break
            
            # read data to device
            img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)
            sem1, sem2 = data['sem1'].to(self.device), data['sem2'].to(self.device)

            # measure data loading time
            am_data_time.update(time.time() - end)

            # run 1st pass
            res_dict = self.model(img1, img2, sem1, sem2, with_bk=True)

            flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_12, flows_21)]
            loss, l_ph, l_sm, flow_mean = self.loss_func(flows, img1, img2, sem1, sem2)
            
            flow_ori = res_dict['flows_fw'][0].detach()
            
            ## ARFlow appearance/spatial transform
            if self.cfg.run_atst:
                img1, img2 = data['img1_ph'].to(self.device), data['img2_ph'].to(self.device)
                noc_ori = self.loss_func.pyramid_vis_mask1[0]  # non-occluded region
                s = {'imgs': [img1, img2], 'sems': [sem1, sem2], 'flows_f': [flow_ori], 'masks_f': [noc_ori]}
                                
                st_res = self.sp_transform(deepcopy(s)) if self.cfg.run_st else deepcopy(s)
                flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]
                
                # run another pass
                img1_st, img2_st = st_res['imgs']
                sem1_st, sem2_st = st_res['sems']
                flow_t_pred = self.model(img1_st, img2_st, sem1_st, sem2_st, with_bk=False)['flows_fw'][0]

                if not self.cfg.mask_st:
                    noc_t = torch.ones_like(noc_t)
                l_atst = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
                l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

                loss += self.cfg.w_ar * l_atst
            else:
                l_atst = torch.zeros_like(loss)  
                
            ## Our semantic augmentation
            if self.cfg.run_ot:
                
                # get cached object mask and do augmentation
                obj_aug_list = [('car', self.car_cache), ('pole', self.pole_cache)]
                np.random.shuffle(obj_aug_list)
                
                img1_ot = data['img1_ph'].to(self.device)
                img2_ot = data['img2_ph'].to(self.device)
                sem1_ot = sem1.clone()
                sem2_ot = sem2.clone()
                flow_ot = flow_ori.clone()
                
                noc_ot = self.loss_func.pyramid_vis_mask1[0].clone()
                if 'ot_focus_new_occ' in self.cfg and self.cfg.ot_focus_new_occ:
                    noc_ot = torch.zeros_like(noc_ot)  # start adding from empty mask
                
                changes = 0
                for obj, cache in obj_aug_list:
                    if obj in self.cfg.ot_classes:                     
                        out = cache.pop(img1.shape[0], with_aug=True)

                        if out is not None: # cache ready
                            obj_mask, img_src, sem_src, mean_flow = out
                            
                            input_dict = {
                                'img1_tgt': img1_ot,
                                'img2_tgt': img2_ot,
                                'sem1_tgt': sem1_ot,
                                'sem2_tgt': sem2_ot,
                                'flow_tgt': flow_ot,
                                'noc_tgt': noc_ot,
                                'img_src': img_src.to(self.device),
                                'sem_src': sem_src.to(self.device),
                                'obj_mask': obj_mask.to(self.device),
                                'motion': mean_flow.to(self.device)
                            }
                                
                            if 'ot_focus_new_occ' in self.cfg and self.cfg.ot_focus_new_occ:
                                img1_ot, img2_ot, sem1_ot, sem2_ot, flow_ot, noc_ot, obj_mask2, new_occ_mask = add_fake_object(input_dict, return_new_occ=True)
                                noc_ot = torch.max(noc_ot, new_occ_mask)
                            else:
                                img1_ot, img2_ot, sem1_ot, sem2_ot, flow_ot, noc_ot, obj_mask2 = add_fake_object(input_dict, return_new_occ=False)
                            
                            changes += 1
                
                if changes > 0:
                    
                    if 'sky' in self.cfg.ot_classes:  # shrink sky flow
                        sky_mask = sem1_ot[:, 10:11]
                        flow_ot = sky_mask * (flow_ot / 2) + (1 - sky_mask) * flow_ot
                        noc_ot = torch.max(noc_ot, sky_mask)

                    # run another pass
                    flow_ot_pred = self.model(img1_ot, img2_ot, sem1_ot, sem2_ot, with_bk=False)['flows_fw'][0]
                    l_ot = ((flow_ot_pred - flow_ot).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
                    l_ot = (l_ot * noc_ot).mean() / (noc_ot.mean() + 1e-7)   
                    loss += self.cfg.w_ar * l_ot
                                 
                else:
                    l_ot = torch.zeros_like(loss)
                
                # push current object mask into cache for future use
                for obj, cache in [('car', self.car_cache), ('pole', self.pole_cache)]:
                    if obj in self.cfg.ot_classes:
                        valid_idx = 1 - torch.isnan(data[obj + '_mask'][:, 0, 0, 0])

                        if valid_idx.sum() > 0: # at least one valid object
                            obj_mask = data[obj + '_mask'][valid_idx]
                            img = data['img1_ph'][valid_idx]
                            sem = data['sem1'][valid_idx]
                            mean_flow = (flow_ori[valid_idx].cpu() * obj_mask).mean(dim=[2, 3]) / obj_mask.mean(dim=[2, 3])

                            cache.push(obj_mask, img, sem, mean_flow) 
                            
            else:
                l_ot = torch.zeros_like(loss)
                

            # update meters
            key_meters.update([loss.item(), l_ph.item(), l_sm.item(), flow_mean.item(), l_atst.item(), l_ot.item()], img1.shape[0])

            # compute gradient and do optimization step
            self.zero_grad()
            loss.backward()
            
            # clip gradient norm and back prop
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)           
            self.optimizer.step()
            self.scheduler.step()

            # measure elapsed time
            am_batch_time.update(time.time() - end)
            end = time.time()

            if (self.i_iter + 1) % self.cfg.record_freq == 0:
                for v, name in zip(key_meters.val, key_meter_names):
                    self.summary_writer.add_scalar('train:{}/'.format(name_dataset) + name, v, self.i_iter + 1)
                
                self.summary_writer.add_scalar('train:{}/time_batch'.format(name_dataset), am_batch_time.val, self.i_iter + 1)
                self.summary_writer.add_scalar('train:{}/time_data'.format(name_dataset), am_data_time.val, self.i_iter + 1)
                self.summary_writer.add_scalar('train:{}/learning_rate'.format(name_dataset), self.optimizer.param_groups[0]['lr'], self.i_iter + 1)
                
            if (self.i_iter + 1) % self.cfg.print_freq == 0:
                istr = '{}:{:04d}/{:04d}'.format(
                    self.i_epoch, i_step + 1, self.cfg.epoch_size) + \
                       ' Time {} Data {}'.format(am_batch_time, am_data_time) + \
                       ' Info {}'.format(key_meters)
                self._log.info(istr)

            self.i_iter += 1
        self.i_epoch += 1

    @torch.no_grad()
    def _validate_with_gt(self):

        self.model.eval()
        
        batch_time = AverageMeter()
        all_error_names = []
        all_error_avgs = []

        n_step = 0
        end = time.time()
        
        for i_set, loader in enumerate(self.valid_loaders):
            name_dataset = loader.dataset.name
            
            error_names = ['EPE_all', 'EPE_noc', 'EPE_occ', 'Fl_all', 'Fl_noc']
            error_meters = AverageMeter(i=len(error_names))
            for i_step, data in enumerate(loader):
                if i_step >= self.cfg.valid_size:
                    break
                    
                img1, img2 = data['img1'].to(self.device), data['img2'].to(self.device)
                sem1, sem2 = data['sem1'].to(self.device), data['sem2'].to(self.device)

                res = list(map(load_flow, data['flow_occ']))
                gt_flows, occ_masks = [r[0] for r in res], [r[1] for r in res]
                res = list(map(load_flow, data['flow_noc']))
                _, noc_masks = [r[0] for r in res], [r[1] for r in res]

                gt_flows = [np.concatenate([flow, occ_mask, noc_mask], axis=2) for
                            flow, occ_mask, noc_mask in
                            zip(gt_flows, occ_masks, noc_masks)]

                # compute output
                flows = self.model(img1, img2, sem1, sem2, with_bk=False)['flows_fw']
                pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])

                es = evaluate_flow(gt_flows, pred_flows)
                error_meters.update([l.item() for l in es], img1.shape[0])
 
                #_DEBUG_PLOT(img1, img2, flows[0], flows[0])
                #import IPython; IPython.embed(); exit()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i_step % self.cfg.print_freq == 0 or i_step == len(loader) - 1:
                    self._log.info('Test: {0}[{1}/{2}]\t Time {3}\t '.format(
                        i_set, i_step, self.cfg.valid_size, batch_time) + ' '.join(
                        map('{:.2f}'.format, error_meters.avg)))

            n_step += len(loader)

            # write error to tf board.
            for value, name in zip(error_meters.avg, error_names):
                self.summary_writer.add_scalar('valid{}:{}/'.format(i_set, name_dataset) + name, value, self.i_epoch)
            self.summary_writer.add_scalar('valid{}:{}/time_batch_avg'.format(i_set, name_dataset), batch_time.avg, self.i_epoch)

            all_error_avgs.extend(error_meters.avg)
            all_error_names.extend(['{}_{}'.format(name, i_set) for name in error_names])

        # In order to reduce the space occupied during debugging,
        # only the model with more than cfg.save_iter iterations will be saved.
        if self.i_iter > self.cfg.save_iter:
            self.save_model(all_error_avgs[0], name='model')

        if self.i_epoch % 50 == 0:
            self.save_model(all_error_avgs[0], name='model_ep{}'.format(self.i_epoch))

        return all_error_avgs, all_error_names


