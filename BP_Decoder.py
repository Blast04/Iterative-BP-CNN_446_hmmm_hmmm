
import torch
import numpy as np


class GetMatrixForBPNet:
    # this class is to calculate the matrices used to perform BP process with matrix operation
    def __init__(self, test_H, loc_nzero_row):
        print("Construct the Matrics H class!\n")
        self.H = test_H
        self.m, self.n = np.shape(test_H)
        self.H_sum_line = np.sum(self.H, axis=0)
        self.H_sum_row = np.sum(self.H, axis=1)
        self.loc_nzero_row = loc_nzero_row
        self.num_all_edges = np.size(self.loc_nzero_row[1, :])

        self.loc_nzero1 = self.loc_nzero_row[1, :] * self.n + self.loc_nzero_row[0, :]
        self.loc_nzero2 = np.sort(self.loc_nzero1)
        self.loc_nzero_line = np.append([np.mod(self.loc_nzero2, self.n)], [self.loc_nzero2 // self.n], axis=0)
        self.loc_nzero4 = self.loc_nzero_line[0, :] * self.n + self.loc_nzero_line[1, :]
        self.loc_nzero5 = np.sort(self.loc_nzero4)

    ##########################################################################################################
    def get_Matrix_VC(self):
        H_x_to_xe0 = np.zeros([self.num_all_edges, self.n], np.float32)
        H_sum_by_V_to_C = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)
        H_xe_last_to_y = np.zeros([self.n, self.num_all_edges], dtype=np.float32)
        Map_row_to_line = np.zeros([self.num_all_edges, 1])

        for i in range(0, self.num_all_edges):
            # Fix: Extract index from np.where result
            Map_row_to_line[i] = np.where(self.loc_nzero1 == self.loc_nzero2[i])[0][0]

        map_H_row_to_line = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, self.num_all_edges):
            map_H_row_to_line[i, int(Map_row_to_line[i])] = 1

        count = 0
        for i in range(0, self.n):
            temp = count + self.H_sum_line[i]
            H_sum_by_V_to_C[count:temp, count:temp] = 1
            H_xe_last_to_y[i, count:temp] = 1
            H_x_to_xe0[count:temp, i] = 1
            for j in range(0, self.H_sum_line[i]):
                H_sum_by_V_to_C[count + j, count + j] = 0
            count = count + self.H_sum_line[i]
        print("return Matrics V-C successfully!\n")
        return H_x_to_xe0, np.matmul(H_sum_by_V_to_C, map_H_row_to_line), np.matmul(H_xe_last_to_y, map_H_row_to_line)

    ###################################################################################################
    def get_Matrix_CV(self):

        H_sum_by_C_to_V = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        Map_line_to_row = np.zeros([self.num_all_edges, 1])

        for i in range(0, self.num_all_edges):
            # Fix: Extract index from np.where result
            Map_line_to_row[i] = np.where(self.loc_nzero4 == self.loc_nzero5[i])[0][0]

        map_H_line_to_row = np.zeros([self.num_all_edges, self.num_all_edges], dtype=np.float32)

        for i in range(0, np.size(self.loc_nzero1)):
            map_H_line_to_row[i, int(Map_line_to_row[i])] = 1

        count = 0
        for i in range(0, self.m):
            temp = count + self.H_sum_row[i]
            H_sum_by_C_to_V[count:temp, count:temp] = 1
            for j in range(0, self.H_sum_row[i]):
                H_sum_by_C_to_V[count + j, count + j] = 0
            count = count + self.H_sum_row[i]
        print("return Matrics C-V successfully!\n")
        return np.matmul(H_sum_by_C_to_V, map_H_line_to_row)


class BP_NetDecoder:
    def __init__(self, H, batch_size, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        _, self.v_node_num = np.shape(H)
        ii, jj = np.nonzero(H)
        loc_nzero_row = np.array([ii, jj])
        self.num_all_edges = np.size(loc_nzero_row[1, :])

        gm1 = GetMatrixForBPNet(H[:, :], loc_nzero_row)
        self.H_sumC_to_V = gm1.get_Matrix_CV()
        self.H_x_to_xe0, self.H_sumV_to_C, self.H_xe_v_sumc_to_y = gm1.get_Matrix_VC()
        self.batch_size = batch_size

        self.H_x_to_xe0_t = torch.from_numpy(self.H_x_to_xe0).float().to(device)
        self.H_sumC_to_V_t = torch.from_numpy(self.H_sumC_to_V).float().to(device)
        self.H_sumV_to_C_t = torch.from_numpy(self.H_sumV_to_C).float().to(device)
        self.H_xe_v_sumc_to_y_t = torch.from_numpy(self.H_xe_v_sumc_to_y).float().to(device)


    def atanh(self, x):
        x1 = x+1.0
        x2 = 1.0-x
        x3 = (x1/x2)+1e-10
        x4 = torch.log(x3)
        return x4/2.0

    def one_bp_iteration(self, xe_v2c_pre_iter, xe_0):
        # Fixed signature to match usage in decode
        H_sumC_to_V = self.H_sumC_to_V_t
        H_sumV_to_C = self.H_sumV_to_C_t

        xe_tanh = torch.tanh(xe_v2c_pre_iter/2.0)
        xe_tanh_temp= torch.sign(xe_tanh)

        xe_sum_log_img=torch.matmul(H_sumC_to_V, (1-xe_tanh_temp)/2.0 *3.1415926)

        xe_sum_log_real=torch.matmul(H_sumC_to_V,torch.log(1e-8+torch.abs(xe_tanh)))
        xe_sum_log_complex = torch.complex(xe_sum_log_real, xe_sum_log_img)

        xe_product = torch.real(torch.exp(xe_sum_log_complex))
        xe_product_temp = torch.sign(xe_product) * -2e-7
        xe_pd_modified = xe_product + xe_product_temp
        xe_v_sumc = self.atanh(xe_pd_modified) * 2.0
        xe_c_sumv =xe_0 + torch.matmul(H_sumV_to_C, xe_v_sumc)

        return xe_v_sumc, xe_c_sumv

    def decode(self, llr_in, bp_iter_num):
        real_batch_size, num_v_node = np.shape(llr_in)

        if real_batch_size != self.batch_size:
            padding = np.zeros([self.batch_size - real_batch_size, num_v_node],
                             dtype=np.float32)
            llr_in = np.vstack([llr_in, padding])

        # llr_tensor defines it and transports
        llr_tensor = torch.from_numpy(llr_in).float().to(self.device).t()

        # set up xe_0 and v2c
        xe_0 = torch.matmul(self.H_x_to_xe0_t, llr_tensor)
        self.xe_v2c_pre_iter = xe_0.clone()

        for _ in range(bp_iter_num - 1):
            xe_v_sumc, xe_c_sumv = self.one_bp_iteration(self.xe_v2c_pre_iter, xe_0)
            self.xe_v2c_pre_iter = xe_c_sumv

        # Final iteration and decoding
        xe_v_sumc, _ = self.one_bp_iteration(self.xe_v2c_pre_iter, xe_0)

        # Get final LLR
        # Fix: Use llr_tensor instead of undefined self.llr_into_bp_net
        bp_out_llr = llr_tensor + torch.matmul(self.H_xe_v_sumc_to_y_t,
                                                          xe_v_sumc)

        dec_out = ((1 - torch.sign(bp_out_llr)) / 2).int()
        y_dec = dec_out.cpu().numpy()

        if real_batch_size != self.batch_size:
            # Transpose back if necessary (based on shape of dec_out)
            # dec_out shape is (N, batch), need (batch, N)?
            # The logic below seems to assume (batch, N) if slicing rows.
            # Wait, bp_out_llr is (N, batch), dec_out is (N, batch).
            # y_dec should be (batch, N) for output?
            y_dec = y_dec.T
            y_dec = y_dec[0:real_batch_size, :]
        else:
             y_dec = y_dec.T

        return y_dec
