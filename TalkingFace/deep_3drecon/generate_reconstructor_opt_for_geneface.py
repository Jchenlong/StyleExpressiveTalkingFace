from options.test_options import TestOptions
import pickle as pkl

# run in the <geneface> root dir!
opt = TestOptions().parse()  # get test options
opt.name='facerecon'
opt.epoch=20
opt.bfm_folder='/app/lpips/BFM/'
opt.checkpoints_dir='/app/lpips/checkpoints/'

with open("/app/lpips/reconstructor_opt.pkl", 'wb') as f:
    pkl.dump(opt, f)
