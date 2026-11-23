import torch
import torch.nn as nn
import torch.nn.functional as F
from model.losses import ChamferDis, PoseDis, SmoothL1Dis, ChamferDis_wo_Batch
from utils.data_utils import generate_augmentation
from model.modules import ModifiedResnet, PointNet2MSG
from model.Net_modules import InstanceAdaptiveKeypointDetector, GeometricAwareFeatureAggregator, PoseSizeEstimator, NOCS_Predictor, Reconstructor,global_enhancev2,FeatureFusion
from vnn_net.vnn import MY_VNN_SimplePointnet_onlyp
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cat_num = cfg.cat_num
        self.cfg = cfg
        if cfg.rgb_backbone == "resnet":
            self.rgb_extractor = ModifiedResnet()
        elif cfg.rgb_backbone == 'dino':
            # frozen dino
            self.rgb_extractor = torch.hub.load('/home/sda2/wang/2025/AG-Pose-main/dinov2-main','dinov2_vits14',source='local')
            for param in self.rgb_extractor.parameters():
                param.requires_grad = False

            self.feature_mlp = nn.Sequential(
                nn.Conv1d(384, 128, 1),
            )
        else:
            raise NotImplementedError
        
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        
        self.IAKD = InstanceAdaptiveKeypointDetector(cfg.IAKD)
        self.GAFA = GeometricAwareFeatureAggregator(cfg.GAFA)

        self.nocs_predictor = NOCS_Predictor(cfg.NOCS_Predictor)
        self.estimator = PoseSizeEstimator()

        self.reconstructor = Reconstructor(cfg.Reconstructor)
        self.glo_enhance1 = global_enhancev2(input_dim=128)
        self.glo_enhance2 = global_enhancev2(input_dim=256)
        self.glo_enhance3 = global_enhancev2(input_dim=256)
        self.vnn = MY_VNN_SimplePointnet_onlyp(c_dim=64, hidden_dim=128,meta_output='equivariant_latent')
        self.FeatureFusion = FeatureFusion(cfg.FF)
        self.processor = CLIPProcessor.from_pretrained("./model/clip-vit-base-patch16")
        self.model = CLIPModel.from_pretrained("./model/clip-vit-base-patch16")
        if self.cat_num == 6:
            self.synset_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
            self.description = ["The bottle typically has a cylindrical shape with a narrow neck, a wider body, and a flat or slightly rounded base.",
                                'The bowl has a circular rim, a rounded concave interior, and a gently curved or flat base.', 
                                "The camera's shape can be described as a compact rectangular prism with a cylindrical lens protruding from the front.", 
                                "A can is a cylindrical object with a circular base and uniform height, typically featuring smooth surfaces and well-defined edges at the top and bottom.",
                                "A laptop is a rectangular, flat, and foldable device with a hinged design, typically consisting of a screen on one side and a keyboard on the other.",
                                "The mug has a cylindrical shape with a handle attached to the side, typically with a flat base and a slightly curved rim."]
        else:
            self.synset_names = ['box', 'bottle', 'can', 'cup', 'remote', 'teapot', 'cutlery', 'glass', 'shoe', 'tube']
            self.description = ['A box is a rectangular prism with six flat, orthogonal faces, eight vertices, and twelve edges, where the edges connect at right angles.', 
                                "The bottle typically has a cylindrical shape with a narrow neck, a wider body, and a flat or slightly rounded base.",
                                "A can is a cylindrical object with a circular base and uniform height, typically featuring smooth surfaces and well-defined edges at the top and bottom.",
                                'The cup has a cylindrical body with a circular base, a smooth, curved surface, and a handle extending from the side.',
                                'The remote is a rectangular, slightly curved object with rounded edges, featuring a grid-like arrangement of buttons on its front surface.',
                                'A teapot can be approximated as a composite shape consisting of a nearly spherical or cylindrical body, a curved spout extending outward, a semi-circular handle on the opposite side, and a slightly convex or flat lid on top.',
                                'The cutlery has an elongated shape with a relatively uniform width along the handle, tapering or widening near the functional end (e.g., fork prongs, spoon bowl, or knife blade), providing clear keypoints at the handle tip, functional end, and possible curvature transitions.',
                                'The shape of a glass can be described as a cylindrical body with a circular rim and base, potentially tapering or expanding slightly, with symmetry along its vertical axis, which facilitates keypoint extraction and feature representation.',
                                'The shape of a shoe can be approximated as a deformed elongated ellipsoid with a slightly curved sole and a tapered front, which helps in defining key regions like the toe box, heel, and midsole for keypoint extraction and feature analysis.',
                                'A tube has a cylindrical shape with two circular ends, and its surface is smooth and curved, making it ideal for keypoint extraction along the central axis and at the edges of the ends.']
        self.clip_mlp1 = nn.Sequential(
            nn.Conv1d(512, 128, 1),
        )
        self.clip_mlp2 = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        )

        self.pts_pts_so = nn.Sequential(
            nn.Conv1d(192, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
        )
        self.pts_so_mlp1 = nn.Sequential(
            nn.Conv1d(192, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 64, 1),
        )
    def build_text_cache(self):
        """
        在推理前预先计算每个类别的文本特征。
        注意：必须在加载 checkpoint 之后调用，因为 clip_mlp2 是训练过的。
        """
        device = next(self.parameters()).device

        texts = [
            'A photo of a ' + self.synset_names[i] + '. ' + self.description[i]
            for i in range(self.cat_num)
        ]

        # 确保推理模式
        self.model.eval()
        self.clip_mlp2.eval()

        with torch.no_grad():
            clip_inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True
            )
            clip_inputs = {
                key: value.to(device)
                for key, value in clip_inputs.items()
            }

            # 原始 CLIP 文本特征: [cat_num, 512]
            text_feature = self.model.get_text_features(**clip_inputs)

            # 经过训练好的 clip_mlp2 映射: [cat_num, 256]
            text_feature = self.clip_mlp2(text_feature)

        # 注册为 buffer，这样它会跟着模型在 GPU 上使用，但不参与训练
        self.register_buffer(
            "cached_text_feature",
            text_feature.detach(),
            persistent=False
        )

    def forward(self, inputs):
        end_points = {}

        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        cls = inputs['category_label'].reshape(-1)


        c = torch.mean(pts, 1, keepdim=True)  
        pts = pts - c
        
        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.cat_num
        
        # rgb feat
        if self.cfg.rgb_backbone == 'resnet':
            rgb_local = self.rgb_extractor(rgb) 
        elif self.cfg.rgb_backbone == 'dino':
            dino_feature = self.rgb_extractor .forward_features(rgb)["x_prenorm"][:, 1:]  
            f_dim = dino_feature.shape[-1]
            num_patches =int(dino_feature.shape[1]**0.5)
            dino_feature = dino_feature.reshape(b, num_patches, num_patches, f_dim).permute(0,3,1,2)
            dino_feature = F.interpolate(dino_feature, size=(num_patches * 14, num_patches * 14), mode='bilinear', align_corners=False) 
            dino_feature = dino_feature.reshape(b, f_dim, -1) 
            rgb_local = self.feature_mlp(dino_feature)
        else:
            raise NotImplementedError
        

        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous() # b, c, n
        #print(rgb_local.shape,'++++++++++++++++++++++++++++++++')

        if self.training:
            delta_r, delta_t, delta_s = generate_augmentation(b)
            pts = (pts - delta_t) / delta_s.unsqueeze(2) @ delta_r

        pts_local = self.pts_extractor(pts) # b, c, n  [36,128,1024]
        pts_local = pts_local.transpose(1,2)
        pts_local = self.glo_enhance1(pts_local)
        pts_local = pts_local.transpose(1,2)
        #####update the so3   encoder
        #########
        # pts_so = pts
        # B=pts_so.size(0)
        # x_list = np.array([], dtype=np.float32)
        # device = pts_so.device
        # x_list = torch.tensor(x_list, device=device)
        # for i in range(B): 
        #     x_i = pts_so[i].unsqueeze(1) 
        #     x_i = x_i.transpose(0, 1) 
        #     x_i = self.vnn(x_i).unsqueeze(0)  
        #     x_list = torch.cat((x_list, x_i), dim=0)
        # pts_so = x_list  
        # # print(pts_so.shape)
        # pts_so = self.pts_so_mlp1(pts_so)
        pts_so = self.vnn(pts)          # [B, 192, N]
        pts_so = self.pts_so_mlp1(pts_so)


        pts_local =self.pts_pts_so(torch.cat((pts_local, pts_so), dim = 1))
        #print(pts_local.shape,'---------------------------------')

        #------------------------------------------------------------------
        # text = ['A photo of a ' + self.synset_names[i.item()] + '. ' + self.description[j.item()] 
        #         for i, j in zip(cls.flatten(), cls.flatten())]
        # inputs = self.processor(text=text, return_tensors="pt", padding=True)
        # inputs = {key: value.cuda() for key, value in inputs.items()}
        # with torch.no_grad():
        #     text_feature = self.model.get_text_features(**inputs)
        #     text_feature = text_feature.unsqueeze(2).repeat(1, 1, pts.size(1))
        # text_feature = self.clip_mlp2(text_feature.transpose(1,2)).transpose(1,2)
        if (not self.training) and hasattr(self, "cached_text_feature"):
            # cls: [B]
            # cached_text_feature: [cat_num, 256]
            # 直接根据类别标签取对应文本特征
            text_feature = self.cached_text_feature[cls.long()]  # [B, 256]

            # 扩展到每个点: [B, 256, N]
            text_feature = text_feature.unsqueeze(2).expand(-1, -1, pts.size(1))
        else:
            text = ['A photo of a ' + self.synset_names[i.item()] + '. ' + self.description[j.item()] 
                    for i, j in zip(cls.flatten(), cls.flatten())]
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {key: value.cuda() for key, value in inputs.items()}
            with torch.no_grad():
                text_feature = self.model.get_text_features(**inputs)
                text_feature = text_feature.unsqueeze(2).repeat(1, 1, pts.size(1))
            text_feature = self.clip_mlp2(text_feature.transpose(1,2)).transpose(1,2)

        fused_feature  = self.FeatureFusion(rgb_local, pts_local, text_feature=text_feature) # (b, n, 2c)
        #-----------------------------------------------------------------'''
        #########
        #################enhance1
        batch_kpt_query, heat_map = self.IAKD(fused_feature, cls, pts)
        kpt_3d = torch.bmm(heat_map, pts)
        
        kpt_feature = torch.bmm(heat_map, fused_feature)
        
        ##########enhance2  [36,96,256]
        #kpt_feature = self.glo_enhance2(kpt_feature)
        kpt_feature = self.GAFA(kpt_feature, kpt_3d.detach(), fused_feature, pts)
        #######enance3
        kpt_feature = self.glo_enhance3(kpt_feature)
        #print(kpt_feature.shape,'====================================')
        if self.training:
            recon_model, recon_delta = self.reconstructor(kpt_3d.transpose(1, 2), kpt_feature.transpose(1, 2))
        kpt_nocs = self.nocs_predictor(kpt_feature, index)
        r, t, s = self.estimator(kpt_3d, kpt_nocs.detach(), kpt_feature)

        if self.training:
            end_points['recon_delta'] = recon_delta
            end_points['pred_heat_map'] = heat_map
            end_points['pred_kpt_3d'] =  \
            (kpt_3d @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + c
            end_points['recon_model'] =  \
            (recon_model.transpose(1, 2) @ delta_r.transpose(1, 2)) * delta_s.unsqueeze(2) + delta_t + c
            end_points['pred_kpt_nocs'] = kpt_nocs
            end_points['pred_translation'] = delta_t.squeeze(1) + delta_s * torch.bmm(delta_r, t.unsqueeze(2)).squeeze(2) + c.squeeze(1)
            end_points['pred_rotation'] = delta_r @ r
            end_points['pred_size'] = s * delta_s

        else:
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_rotation'] = r
            end_points['pred_size'] = s
            end_points['pred_kpt_3d'] =  kpt_3d + c
            end_points['kpt_nocs'] = kpt_nocs

        return end_points



class Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def forward(self, endpoints):
        
        pts = endpoints['pts']
        b = pts.shape[0] 
        
        recon_delta = endpoints['recon_delta']
        pred_kpt_3d = endpoints['pred_kpt_3d']
        recon_model = endpoints['recon_model']
        
        translation_gt = endpoints['translation_label']
        rotation_gt = endpoints['rotation_label']
        size_gt = endpoints['size_label']
        
        # pose 
        loss_pose = PoseDis(endpoints['pred_rotation'], endpoints['pred_translation'], endpoints['pred_size'], rotation_gt, translation_gt, size_gt)
        # cd
        loss_cd = self.cd_dis_k2p(pts, pred_kpt_3d)
        # nocs
        kpt_nocs_gt = (pred_kpt_3d - translation_gt.unsqueeze(1)) / (torch.norm(size_gt, dim=1).view(b, 1, 1) + 1e-8) @ rotation_gt
        loss_nocs = SmoothL1Dis(endpoints['pred_kpt_nocs'], kpt_nocs_gt)
        # div
        loss_diversity = self.diversity_loss_3d(pred_kpt_3d)
        # reconstruction
        if self.cfg.obj_aware:
            # recon_with_mask
            loss_recon = self.ChamferDis_with_mask(pts, recon_model, endpoints['pc_mask'])
        else:
            loss_recon = ChamferDis(pts, recon_model)
        # regularization 
        loss_delta = recon_delta.norm(dim=2).mean()
        
        loss_all = self.cfg.pose*loss_pose + self.cfg.nocs*loss_nocs + self.cfg.cd*loss_cd + \
            self.cfg.diversity*loss_diversity + self.cfg.recon*loss_recon + self.cfg.delta*loss_delta
        return {
            'loss_all': loss_all,
            'loss_pose': self.cfg.pose*loss_pose,
            'loss_nocs': self.cfg.nocs*loss_nocs,
            'loss_cd': self.cfg.cd*loss_cd,
            'loss_diversity': self.cfg.diversity*loss_diversity,
            'loss_recon': self.cfg.recon*loss_recon,
            'loss_delta': self.cfg.delta*loss_delta,
        }
        
    def ChamferDis_with_mask(self, pts, recon_model, pc_mask):
        """
        calculate ChamferDis with valid pointcloud mask
        Args:
            pts: (b, n1, 3)
            recon_model: (b, n2, 3)
            pc_mask: (b, n1)

        Return:
            recon_loss
        """
        b = pts.shape[0]
        is_first = True
        
        for idx in range(b):
            pts_ = pts[idx] # (n1, 3)
            pts_ = pts_[pc_mask[idx] == True] 
            if pts_.shape[0] == 0:
                print('warning: no valid point')
                continue
            recon_model_ = recon_model[idx] 
            dis = ChamferDis_wo_Batch(pts_, recon_model_)
            
            if is_first:
                dis_all = dis
                is_first = False
            else:
                dis_all += dis
        return dis_all / b

    def cd_dis_k2p(self, pts, pred_kpt_3d):
        """_summary_

        Args:
            pts (_type_): (b, n, 3)
            pred_kpt_3d (_type_): (b, kpt_num, 3)
        """
        # (b, n, 1, 3)   -   (b, 1, kpt_num, 3)  = (b, n, kpt_num, 3) -> (b, n, kpt_num)
        dis = torch.norm(pts.unsqueeze(2) - pred_kpt_3d.unsqueeze(1), dim=3)
        # (b, kpt_num)
        dis = torch.min(dis, dim=1)[0]
        return torch.mean(dis)
    
    def diversity_loss_3d(self, data):
        """_summary_

        Args:
            data (_type_): (b, kpt_num, 3)
        """
        threshold = self.cfg.th
        b, kpt_num = data.shape[0], data.shape[1]
        
        dis_mat = data.unsqueeze(2) - data.unsqueeze(1)
        # (b, kpt_num, kpt_num)
        dis_mat = torch.norm(dis_mat, p=2, dim=3, keepdim=False)
        
        dis_mat = dis_mat + torch.eye(kpt_num, device=dis_mat.device).unsqueeze(0)
        dis_mat[dis_mat >= threshold] = threshold
    
        # dis=0 -> loss=1     dis=threshold -> loss=0   y= -x/threshold + 1
        dis_mat = -dis_mat / threshold + 1
        
        loss = torch.sum(dis_mat, dim=[1, 2])
        loss = loss / (kpt_num * (kpt_num-1))
        
        return loss.mean()