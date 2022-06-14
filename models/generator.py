import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np

class Generator(nn.Module):

    def __init__(self, config, img_c, img_sz):

        super(Generator, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        
        self.layer_sizes = [84, 84, 168, 168]
        self.num_inner_layers = 3

        # Number of times dimension is halved
        self.U_depth = len(self.layer_sizes)
        num_noise_filters = 8
        self.z_channels = []
        self.dim_arr = [img_sz]
        for i in range(self.U_depth):
            self.dim_arr.append((self.dim_arr[-1] + 1) // 2)
            
        for i, (name, param) in enumerate(self.config):
    
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                # [ch_in, ch_out, kernel_sz, kernel_sz, stride, padding]
                # output will be sz = stride * (input_sz) + kernel_sz
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'encode':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'decode':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
                
            elif name == 'pad':
                pad_ = param
            
            elif name == 'encode0': #param out in kernel kernel stride activate
                
                w = nn.Parameter(torch.ones(param[:4]))
                
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
                if param[5]:
                    w = nn.Parameter(torch.ones(param[0]))
                    self.vars.append(w)
                    # [ch_out]
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))
                    running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])

            elif name == "_EncoderBlock":
                w = nn.Parameter(torch.ones(param[0],param[0],3,3)) #pre_conv
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
                w = nn.Parameter(torch.ones(param[2],param[0]+param[1],3,3)) # out conv
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))
                
                w = nn.Parameter(torch.ones(param[2],param[1]+param[2],3,3)) #conv_1
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))
                
                w = nn.Parameter(torch.ones(param[2],param[1]+param[2]+param[2],3,3)) #conv_2
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))
                
                w = nn.Parameter(torch.ones(param[2],param[1]+param[2]+param[2]+param[2],3,3)) #conv_3
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))
            elif name == "z_reshape0":
                
                curr_dim = self.dim_arr[-param[0] - 1]

                w = nn.Parameter(torch.ones(curr_dim*curr_dim*num_noise_filters,100))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(curr_dim*curr_dim*num_noise_filters)))
                self.z_channels.append(num_noise_filters)
                
                num_noise_filters //= 2
                
            elif name == "_DecoderBlock":   #pre_channels,in_channels,out_channels,num_layers,curr_size,upscale_size=None,
                total_channels = param[0] + param[1]

                if param[0]:
                    w = nn.Parameter(torch.ones(param[0],param[0],3,3))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))
                
                w = nn.Parameter(torch.ones(param[2],total_channels,3,3))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))  
                total_channels = total_channels + param[2]
                
                if param[0]:
                    w = nn.Parameter(torch.ones(param[0],param[0],3,3))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[0])))
                    
                w = nn.Parameter(torch.ones(param[2],total_channels,3,3))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))  
                total_channels = total_channels + param[2]      
                
                if param[0]:
                    w = nn.Parameter(torch.ones(param[0],param[0],3,3))
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[0]))) 
                    
                w = nn.Parameter(torch.ones(param[2],total_channels,3,3))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[2])))  
                total_channels = total_channels + param[2]
                
                if param[5]:
                    total_channels = total_channels - param[0]
                    w = nn.Parameter(torch.ones(total_channels,param[2],3,3))
                    self.vars.append(w)
                    torch.nn.init.kaiming_normal_(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[2])))
                    
                    w = nn.Parameter(torch.ones(param[2]))
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[2])))
                    running_mean = nn.Parameter(torch.zeros(param[2]), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(param[2]), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])
            elif name == "final_conv":
                w = nn.Parameter(torch.ones(param[1],param[0],param[2],param[2]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))     
                # [ch_out]
                if param[4]:
                    w = nn.Parameter(torch.ones(param[1]))
                    self.vars.append(w)
                    self.vars.append(nn.Parameter(torch.zeros(param[1])))
                    running_mean = nn.Parameter(torch.zeros(param[1]), requires_grad=False)
                    running_var = nn.Parameter(torch.ones(param[1]), requires_grad=False)
                    self.vars_bn.extend([running_mean, running_var])
            elif name == "convert_z":
                continue
            elif name == "cur_input":
                continue
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'identity', 'update_identity', 'encode', 'decode']:
                continue
            else:
                raise NotImplementedError


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'

            elif name == 'random_proj':
                info += 'temp'#'random_proj:(hidden_sz:%d, embedding_size:%d, height_width:%d, ch_out:%d)'%(param[0], param[1], param[2], param[3]) + '\n'
            
            elif name == 'encode':
                info += 'temp'#'random_proj:(hidden_sz:%d, embedding_size:%d, height_width:%d, ch_out:%d)'%(param[0], param[1], param[2], param[3]) + '\n'

            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn', 'identity', 'update_identity', 'encode', 'decode']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, z, vars=None):

        batch_sz = x.size()[0]

        x_orig = x

        if vars == None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        reshape_i = 0
        z_channels = 8
        for name, param in self.config:
            # print(x.size())
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=True, eps=1e-3, momentum=0.01)
                idx += 2
                bn_idx += 2

            elif name == "encode0":
                
                if x.shape[2] % 2 == 0:
                    out_x = F.pad(x, (0, 1, 0, 1))
                else:
                    out_x = F.pad(x, (1, 1, 1, 1))
                    
                w,b = vars[idx],vars[idx + 1]
                out_x = F.conv2d(out_x, w, b, stride=param[4], padding=0)
                idx += 2
                if param[5]:
                    w, b = vars[idx], vars[idx + 1]
                    out_x = F.relu(out_x,0.2)
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                    out_x = F.batch_norm(out_x, running_mean, running_var, weight=w, bias=b, training=True, eps=1e-3, momentum=0.01)
                    
                    idx += 2
                    bn_idx += 2
                all_outputs = [x, out_x]

                tmp_out = [x, out_x]

            elif name == "_EncoderBlock":
                pre_input , out_x = tmp_out

                if pre_input.shape[2] % 2 == 0: #pre_input
                    pre_input = F.pad(pre_input, (0, 1, 0, 1))
                else:
                    pre_input = F.pad(pre_input, (1, 1, 1, 1))
                    
                w,b = vars[idx],vars[idx + 1]
                
                idx += 2

                pre_input = F.conv2d(pre_input, w, b, stride=2, padding=0) #pre_conv
                
                out = torch.cat([out_x, pre_input],1) #conv0 (x,pre_input)

                out = F.pad(out, (1, 1, 1, 1))

                w,b = vars[idx],vars[idx + 1]
                out = F.conv2d(out, w, b, stride=1, padding=0)  #conv out
                idx += 2

                encode_output = [out_x, out]
                
                input_features = torch.cat(encode_output, 1) #1
                out = F.pad(input_features, (1, 1, 1, 1))
                w,b = vars[idx],vars[idx + 1]
                out = F.conv2d(out, w, b, stride=1, padding=0)
                idx += 2
                encode_output.append(out)
                          
                input_features = torch.cat(encode_output, 1) #2
                out = F.pad(input_features, (1, 1, 1, 1)) 
                w,b = vars[idx],vars[idx + 1]
                out = F.conv2d(out, w, b, stride=1, padding=0)
                idx += 2
                encode_output.append(out)

                input_features = torch.cat(encode_output, 1) #3
                out = F.pad(input_features, (1, 1, 1, 1)) 
                w,b = vars[idx],vars[idx + 1]
                out = F.conv2d(out, w, b, stride=2, padding=0)
                idx += 2
                encode_output.append(out)

                tmp_out = encode_output[-2] , encode_output[-1]

                pre_input, curr_input = None, tmp_out[1]

                all_outputs.append(tmp_out[1])
                len(all_outputs)
            
            elif name == "z_reshape0":
                w, b = vars[idx], vars[idx + 1]
                z_out = F.linear(z,w,b)
                idx += 2
                curr_dim = self.dim_arr[- reshape_i -1]
                z_out = z_out.view(-1, self.z_channels[reshape_i], curr_dim, curr_dim)
                curr_input = torch.cat([z_out, curr_input], 1)
                reshape_i = reshape_i + 1

            elif name == "z_reshape1":
                curr_input = torch.cat([curr_input, all_outputs[-reshape_i - 1]], 1)
                w, b = vars[idx], vars[idx + 1]
                z_out = F.linear(z,w,b)
                idx += 2
                curr_dim = self.dim_arr[- reshape_i -1]
                z_out = z_out.view(-1, self.z_channels[reshape_i], curr_dim, curr_dim)
                curr_input = torch.cat([z_out, curr_input], 1)

            elif name == "_DecoderBlock":#pre_channels,in_channels,out_channels,num_layers,curr_size,upscale_size=None

                inp = [pre_input, curr_input]
                decode_outputs = [curr_input]
                for i in range(3):
                    decode_curr_input = decode_outputs[-1]
                    if param[0]:
                        pre_conv_output = F.upsample(pre_input,param[4])
                        w, b = vars[idx], vars[idx + 1]
                        idx = idx + 2
                        pre_conv_output = F.conv_transpose2d(pre_conv_output,w,b,1,1)
                        decode_curr_input = torch.cat([decode_curr_input, pre_conv_output], 1)
                    input_features = torch.cat([decode_curr_input] + decode_outputs[:-1], 1)
                    w,b = vars[idx],vars[idx + 1]
                    input_features = F.pad(input_features, (1, 1, 1, 1))
                    out = F.conv2d(input_features, w, b, stride=1, padding=0)
                    idx = idx + 2
                    decode_outputs.append(out)
                if param[5]:
                    input_features = torch.cat(decode_outputs, 1)
                    input_features = F.upsample(input_features,param[5])
                    w, b = vars[idx], vars[idx + 1]
                    idx = idx + 2

                    input_features = F.conv_transpose2d(input_features,w,b,1,1)

                    input_features = F.leaky_relu(input_features, 0.2)
                    w, b = vars[idx], vars[idx + 1]
                    idx = idx + 2
                    
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                    bn_idx = bn_idx + 2
                    tmp_out = F.batch_norm(input_features, running_mean, running_var, weight=w, bias=b, training=True, eps=1e-3, momentum=0.01)
                    decode_outputs.append(tmp_out)
                pre_input, curr_input = decode_outputs[-2], decode_outputs[-1]

            elif name == "_conv2d":
                if x.shape[2] % 2 == 0:
                    x = F.pad(x, (0, 1, 0, 1))
                else:
                    x = F.pad(x, (1, 1, 1, 1))
                w,b = vars[idx],vars[idx + 1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                          
            elif name == "_conv2d_transpose":
                x = F.upsample(x,param[6])
                w,b = vars[idx],vars[idx + 1]
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                
            elif name == "cur_input":
                curr_input = torch.cat([curr_input, all_outputs[-param[0] - 1]], 1)
            elif name == "final_conv":
                curr_input = F.pad(curr_input, (1, 1, 1, 1))
                w,b = vars[idx],vars[idx + 1]
                idx = idx + 2

                curr_input = F.conv2d(curr_input, w, b, stride=param[3], padding=0)
                if param[4]:
                    F.leaky_relu(x, negative_slope=0.2)
                    w, b = vars[idx], vars[idx + 1]
                    idx = idx + 2
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                    bn_idx = bn_idx + 2
                    curr_input = F.batch_norm(curr_input, running_mean, running_var, weight=w, bias=b, training=True, eps=1e-3, momentum=0.01)                
            elif name == "convert_z":
                x = z
            elif name == 'update_identity':
                x_orig = x
            elif name == 'identity':
                # print(x.shape)
                x += x_orig
            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = torch.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError
        # make sure variable is used properly

        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        # right now still returning y so that we can easilly extend to generating diff nums of examples by adjusting y in here
        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars == None:
                for p in self.vars:
                    if not p.g+rad == None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if not p.grad ==  None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars