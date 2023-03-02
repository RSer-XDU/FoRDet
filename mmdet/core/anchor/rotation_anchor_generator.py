import torch


class RotationAnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, angles, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.angles = torch.Tensor(angles)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        
        ws_, _ = torch.meshgrid(ws, self.angles)
        hs_, anchor_angles = torch.meshgrid(hs, self.angles)
        anchor_angles = torch.reshape(anchor_angles, [-1, 1])
        ws_ = torch.squeeze(torch.reshape(ws_, [1, -1]))
        hs_ = torch.squeeze(torch.reshape(hs_, [1, -1]))



        # hbase_anchors = torch.stack(
        #     [
        #         x_ctr - 0.5 * (ws_ - 1), y_ctr - 0.5 * (hs_ - 1),
        #         x_ctr + 0.5 * (ws_ - 1), y_ctr + 0.5 * (hs_ - 1)
                
        #     ],
        #     dim=-1).round()
        # print('hbase_acnhors', hbase_anchors)
        # print(x_ctr)
        x_ctr_ = x_ctr * torch.ones_like(ws_)
        y_ctr_ = x_ctr * torch.ones_like(ws_)
        # print(x_ctr)
        # print(ws_)
        xywh_hbase_anchors = torch.stack(
            [
                x_ctr_, y_ctr_ ,
                ws_ , hs_ 
                
            ],
            dim=-1).round()

        # rbase_anchors = hbb2obb(hbase_anchors, anchor_angles)

        rbase_anchors  = torch.cat([xywh_hbase_anchors, anchor_angles], 
                         dim=-1)



        return rbase_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        # print('shift_x', shift_xx.size())
        shift_w = torch.zeros_like(shift_xx)
        shift_h = torch.zeros_like(shift_xx)
        shift_angles = torch.zeros_like(shift_xx)
        # print('shift_angle', shift_angles.size())
        shifts = torch.stack([shift_xx, shift_yy, shift_w, shift_h, shift_angles], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...

        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
