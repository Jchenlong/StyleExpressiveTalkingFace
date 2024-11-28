"""Deploy Talking Face Model.  input: landmark array, style_space latent """
import os 
import sys 
sys.path.insert(0, os.getcwd()) 
current_path = os.getcwd()
import torch 
import yaml 
import click 
import tqdm
import shutil
import pickle
import cv2
import copy
import shutil
import imageio

import numpy as np 

from itertools import chain

from easydict import EasyDict as edict 
from TalkingFace.aligner import offsetNet, size_of_alpha, alphas, \
                                update_lip_region_offset, logger, update_region_offset, \
                                face_parsing, offsetNetV2
                                
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.train import stylegan_path
from TalkingFace.ExpressiveVideoStyleGanEncoding.ExpressiveEncoding.equivalent_decoder import EquivalentStyleSpaceDecoder
from TalkingFace.equivalent_offset import fused_offsetNet

norm = lambda x: ((x - x.min(axis = (0,1), keepdims = True)) / (x.max(axis = (0,1), keepdims = True) - x.min(axis = (0,1), keepdims = True)) - 0.5) * 2 
def onnx_infer(
               onnx_path: str,
               *args
              ):
    import onnxruntime as ort 
    def parse_inputs(_inputs):
        if isinstance(_inputs, list):
            for i, x in enumerate(_inputs):
                _inputs[i] = parse_inputs(x)
            return _inputs
        elif isinstance(_inputs, dict):
            for k, v in _inputs:
                _inputs[k] = parse_inputs(v)
            return _inputs
        elif isinstance(_inputs, torch.Tensor):
            return _inputs.cpu().numpy()
    _session = ort.InferenceSession(onnx_path)
    values = [parse_inputs(args[0])] + [parse_inputs(x) for x in args[1]]
    for x in _session.get_inputs():
        print(x.name)
        print(x.shape)
    return _session.run(None, dict(zip([x.name for x in _session.get_inputs()], values)))

class TalkingFaceModel(torch.nn.Module):
    """Deploy Model.
    """
    def __init__(self,
                 net_config : str,
                 net_weight_path : str,
                 decoder_path: str
                ):
        super().__init__()

        renorm = "clip" if not hasattr(net_config, "renorm") else net_config.renorm
        is_refine = False if not hasattr(net_config, "is_refine") else net_config.is_refine
        remap = False if not hasattr(net_config, "remap") else net_config.remap
        use_fourier = False if not hasattr(net_config, "use_fourier") else net_config.use_fourier
        name = "offsetNet" if not hasattr(net_config, "name") else net_config.name
        net = eval(name)(\
                        net_config.in_channels * 2 if name == "offsetNet" else net_config.in_channels, size_of_alpha, \
                        net_config.depth, \
                        base_channels = 512 if not hasattr(net_config, "base_channels") else net_config.base_channels , \
                        max_channels = 128 if not hasattr(net_config, "max_channels") else net_config.max_channels , \
                        dropout = False if not hasattr(net_config, "dropout") else net_config.dropout, \
                        norm = "BatchNorm1d" if not hasattr(net_config, "norm") else net_config.norm, \
                        skip = False if not hasattr(net_config, "skip") else net_config.skip, \
                        norm_type = 'linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
                        norm_dim = [0, 1] if not hasattr(net_config, "norm_dim") else net_config.norm_dim, \
                        renorm = renorm ,\
                        remap = remap,
                        use_fourier = use_fourier,
                        from_size = 512 if not hasattr(net_config, "from_size") else net_config.from_size,
                        act = "ReLU" if not hasattr(net_config, "act") else net_config.act,
                        first_convbn = False if not hasattr(net_config, "first_convbn") else net_config.first_convbn,
                        target_size = 1 if not hasattr(net_config, "target_size") else net_config.target_size
                       )
        logger.info(net)
        net.load_state_dict(torch.load(net_weight_path)['weight'])
        net.eval()
        #net = fused_offsetNet(copy.deepcopy(net))

        decoder = EquivalentStyleSpaceDecoder(stylegan_path, to_resolution = 512)
        decoder.load_state_dict(torch.load(decoder_path), False)
        decoder.eval()

        self.net = fused_offsetNet(copy.deepcopy(net))
        
        self.decoder = decoder

        #self.selected_landmark = torch.from_numpy(np.load(config.selected_path)).float()

    def forward(self, x, latent):
        offset = self.net(x)
        ss_updated = update_lip_region_offset(latent, offset)
        image = self.decoder(ss_updated)
        return torch.nn.functional.interpolate(image, (512,512))


class TalkingFaceModel_Stage_One(torch.nn.Module):
    """Deploy Model.
    """

    def __init__(self,
                 net_config: str,
                 net_weight_path: str,
                 ):
        super().__init__()

        renorm = "clip" if not hasattr(net_config, "renorm") else net_config.renorm
        is_refine = False if not hasattr(net_config, "is_refine") else net_config.is_refine
        remap = False if not hasattr(net_config, "remap") else net_config.remap
        use_fourier = False if not hasattr(net_config, "use_fourier") else net_config.use_fourier
        name = "offsetNet" if not hasattr(net_config, "name") else net_config.name
        net = eval(name)( \
            net_config.in_channels * 2 if name == "offsetNet" else net_config.in_channels, size_of_alpha, \
            net_config.depth, \
            base_channels=512 if not hasattr(net_config, "base_channels") else net_config.base_channels, \
            max_channels=128 if not hasattr(net_config, "max_channels") else net_config.max_channels, \
            dropout=False if not hasattr(net_config, "dropout") else net_config.dropout, \
            norm="BatchNorm1d" if not hasattr(net_config, "norm") else net_config.norm, \
            skip=False if not hasattr(net_config, "skip") else net_config.skip, \
            norm_type='linear' if not hasattr(net_config, "norm_type") else net_config.norm_type, \
            norm_dim=[0, 1] if not hasattr(net_config, "norm_dim") else net_config.norm_dim, \
            renorm=renorm, \
            remap=remap,
            use_fourier=use_fourier,
            from_size=512 if not hasattr(net_config, "from_size") else net_config.from_size,
            act="ReLU" if not hasattr(net_config, "act") else net_config.act,
            first_convbn=False if not hasattr(net_config, "first_convbn") else net_config.first_convbn,
            target_size=1 if not hasattr(net_config, "target_size") else net_config.target_size
        )
        logger.info(net)
        net.load_state_dict(torch.load(net_weight_path)['weight'])
        net.eval()
        # net = fused_offsetNet(copy.deepcopy(net))

        self.net = fused_offsetNet(copy.deepcopy(net))



    def forward(self, x):
        offset = self.net(x)
        return offset


class TalkingFaceModel_Stage_Two(torch.nn.Module):
    """Deploy Model.
    """

    def __init__(self,
                 decoder_path: str
                 ):
        super().__init__()

        decoder = EquivalentStyleSpaceDecoder(stylegan_path, to_resolution=512)
        decoder.load_state_dict(torch.load(decoder_path), False)
        decoder.eval()
        self.decoder = decoder
    def forward(self, offset, latent):
        ss_updated = update_lip_region_offset(latent[:-1], offset)
        image = self.decoder(ss_updated, insert_feature={"4": latent[-1]})
        return torch.nn.functional.interpolate(image, (512, 512))

class TalkingFaceModelv2(TalkingFaceModel):
    """support f space network.
    """
    def forward(self, x, latent):
        offset = self.net(x)
        ss_updated = update_lip_region_offset(latent[:-1], offset)
        image = self.decoder(ss_updated, insert_feature = {"4": latent[-1]})
        return torch.nn.functional.interpolate(image, (512,512))

def get_blend_mask(
                     image: np.ndarray,
                     face_parse: object
                  ):
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = face_parse(image)
    #erosion_size = 20
    #element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size))
    #mask_erosion = cv2.erode(mask, element)

    mask = cv2.boxFilter(mask.astype(np.float32), -1, ksize = (21, 21))
    if mask.ndim < 3:
        mask = mask[..., np.newaxis]
    return mask * 255.0

def deploy(
            exp_name: str,
            decoder_path: str,
            to_path: str,
            image_path: str,
            patch: bool = False
         ):
    
    #assert os.path.isdir(to_path), "to_path expected is directory."
    # current_path = '/data1/chenlong/0517/video/0822/S1YRnPRL/S1YRnPRL_exp_set_v4_depoly/lm_train_facial_ft'
    current_path = f'{os.path.dirname(to_path)}/lm_train_n'
    f_space_path = f'{os.path.dirname(to_path)}/exp/f_space/f_space_ft.pt'
    # if os.path.exists(current_path):
    #     pass
    # else:
    #     current_path = f'{os.path.dirname(to_path)}/lm_train'
    #     current_path = f'{os.path.dirname(to_path)}/lm_train_facial_ft'

    print('---------------------------------------')
    print(f_space_path)
    print(current_path)
    to_path = os.path.join(to_path, exp_name)
    os.makedirs(to_path, exist_ok = True)

    to_path_model = os.path.join(to_path, 'model.pt') 
    to_path_landmark = os.path.join(to_path, 'landmark.npy') 
    to_path_latents = os.path.join(to_path, 'latents.pkl') 
    to_path_masks = os.path.join(to_path, "templates", "blend_masks.mp4")
    os.makedirs(os.path.dirname(to_path_masks), exist_ok = True)

    shutil.copy(f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/config.yaml', os.path.join(to_path, "config.yaml"))

    net_config_path = os.path.join(current_path, 'scripts', exp_name, 'config.yaml')
    net_snapshots_path = os.path.join(current_path, 'results', exp_name, 'snapshots', 'best.pth')

    with open(net_config_path) as f:
        config = edict(yaml.load(f, Loader = yaml.CLoader))
    
    f_space = None
    # if hasattr(config, "f_space_path"):
    if os.path.exists(f_space_path):
        # f_space_path = os.path.join(current_path, config.f_space_path)
        print(f'f_space_path:{f_space_path}')
        f_space = torch.load(f_space_path)
        # model = TalkingFaceModelv2(config.net, net_snapshots_path, decoder_path)
        model_stage_one = TalkingFaceModel_Stage_One(config.net, net_snapshots_path)
        model_stage_two = TalkingFaceModel_Stage_Two(decoder_path)
    else:
        model = TalkingFaceModel(config.net, net_snapshots_path, decoder_path)

    face_parse = face_parsing()
    model_stage_one.eval()
    model_stage_two.eval()
    device = 'cpu'
    model_stage_one.to(device)
    model_stage_two.to(device)
    if config.net.name == "offsetNetV2":
        _input = torch.randn(1, config.net.in_channels, config.net.from_size, config.net.from_size)
    else:
        _input = torch.randn(1,config.net.in_channels, 2).to(device)
    #index = 1
    latent = torch.randn(1,18,512).to(device)
    style_space = model_stage_two.decoder.get_style_space(latent)
    if f_space:
        style_space += [torch.randn(1,512,4,4).to(device)]

    to_path_model_stage_one = os.path.join(to_path, 'OffSetNet/model.pt')
    os.makedirs(os.path.join(to_path, 'OffSetNet'),exist_ok = True)
    module = torch.jit.trace(model_stage_one, (_input), check_trace = False)
    offset_original = model_stage_one(_input)
    offset = module(_input)
    diff = torch.abs(offset_original - offset)
    logger.info(f"model_stage_one max error {diff.max()}, min error {diff.min()}, avg error {diff.mean()}")
    onnx_model_path = to_path_model_stage_one.replace('pt', 'onnx')
    torch.onnx.export(model_stage_one, (_input), onnx_model_path, verbose = True, opset_version = 15, do_constant_folding = True)

    to_path_model_stage_two = os.path.join(to_path, 'Decoder/model.pt')
    os.makedirs(os.path.join(to_path, 'Decoder'),exist_ok = True)

    module = torch.jit.trace(model_stage_two, (offset, style_space), check_trace=False)
    output_original = model_stage_two(offset, style_space)
    output = module(offset, style_space)
    diff = torch.abs(output_original - output)
    logger.info(f"model_stage_two max error {diff.max()}, min error {diff.min()}, avg error {diff.mean()}")
    onnx_model_path = to_path_model_stage_two.replace('pt', 'onnx')
    torch.onnx.export(model_stage_two, (offset, style_space), onnx_model_path, verbose=True, opset_version=15,
                      do_constant_folding=True)

    if patch:
        logger.info("update patch")
        return

    logger.info("get landmark.")
    # output_onnx = onnx_infer(onnx_model_path, _input, style_space)
    # diff = np.abs(output_original.detach().cpu().numpy() - output_onnx)
    # logger.info(f"max error {diff.max()}, min error {diff.min()}, avg error {diff.mean()}")
    shutil.copy(os.path.join(current_path, config.data[0].dataset.id_landmark_path), to_path_landmark)

    logger.info("get style space latents.")
    attributes = torch.load(os.path.join(current_path, config.attr_path), map_location = "cpu")
    pose_latents = torch.load(os.path.join(current_path, config.pose_path), map_location = "cpu")
    n = len(attributes)
    p_bar = tqdm.tqdm(range(n))
    to_save_list = []
    writer = imageio.get_writer(to_path_masks, fps = 25)

    model_stage_two.to("cuda:0")
    postfixs = ["png", "jpg"]

    # check exists.
    postfix = ""
    for _p in postfixs:
        _path = os.path.join(image_path, f'{0}.') + _p
        if os.path.exists(_path):
            postfix = _p
            break

    for i in p_bar:

        _path = os.path.join(image_path, f'{i}.') + postfix
        mask = get_blend_mask(cv2.imread(_path), face_parse)
        writer.append_data(np.uint8(mask))
        attribute = attributes[i]
        w_plus_with_pose = pose_latents[i].detach().cpu()
        style_space_latent = model_stage_two.decoder.get_style_space(w_plus_with_pose.to("cuda:0"))
        ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1][size_of_alpha:]).reshape(1, -1).to('cuda:0'), [8, len(alphas)])
        #ss_updated = update_region_offset(style_space_latent, torch.tensor(attribute[1]).reshape(1, -1).to('cuda:0'), [0, len(alphas)])
        if f_space:
            ss_updated.append(f_space[i])
        if i < 0:
            image_tensor = model(torch.from_numpy(offsets[i:i+1]).to(device), ss_updated)
            image = image_tensor.detach().cpu().squeeze(0).permute((1,2,0)).numpy()
            image = (image + 1) * 0.5
            cv2.imwrite(f"{i + 1}.jpg", image[...,::-1] * 255.0)

        if len(to_save_list) == 0:
            numpy_arrays = []
            for index, z in enumerate(ss_updated):
                z = z.detach().cpu().numpy()
                if z.ndim == 2:
                    _,c = z.shape
                    empty_array = np.empty((n, c), dtype = z.dtype)
                elif z.ndim == 4:
                    _,c,h,w = z.shape
                    empty_array = np.empty((n, c, h, w), dtype = z.dtype)

                empty_array[i:i+1, ...] = z
                to_save_list.append(empty_array)
        else:
            for index, z in enumerate(ss_updated):
                z = z.detach().cpu().numpy()
                to_save_list[index][i:i+1, ...] = z
    #np.save(to_path_masks, to_save_mask)
    with open(to_path_latents, 'wb') as f:
        pickle.dump(to_save_list, f, protocol = 4)
import re
@click.command()
@click.option('--exp_name')
@click.option('--decoder_path')
@click.option('--to_path')
@click.option('--image_path')
@click.option('--patch', default = 0)
def _invoker_deploy(
                    exp_name,
                    decoder_path,
                    to_path,
                    image_path,
                    patch
                   ):
    if not decoder_path.endswith('pth'):
        folder = decoder_path
        decoder_path = os.path.join(folder, sorted(os.listdir(decoder_path), key = lambda x: int(''.join(re.findall('[0-9]+', x))))[-1])
        print(f"latest weight path is {decoder_path}")
    return deploy(exp_name, decoder_path, to_path, image_path, patch)

if __name__ == '__main__':
    _invoker_deploy()
    


