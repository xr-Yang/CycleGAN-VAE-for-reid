# train all cameras for market

#python train.py --dataroot ./datasets/market --name market-c1-c2 --camA 1 --camB 2
#python train.py --dataroot ./datasets/market --name market-c1-c3 --camA 1 --camB 3
#python train.py --dataroot ./datasets/market --name market-c1-c4 --camA 1 --camB 4
#python train.py --dataroot ./datasets/market --name market-c1-c5 --camA 1 --camB 5
#python train.py --dataroot ./datasets/market --name market-c1-c6 --camA 1 --camB 6
#
#python train.py --dataroot ./datasets/market --name market-c2-c3 --camA 2 --camB 3
#python train.py --dataroot ./datasets/market --name market-c2-c4 --camA 2 --camB 4
#python train.py --dataroot ./datasets/market --name market-c2-c5 --camA 2 --camB 5
#python train.py --dataroot ./datasets/market --name market-c2-c6 --camA 2 --camB 6
#
#python train.py --dataroot ./datasets/market --name market-c3-c4 --camA 3 --camB 4
#python train.py --dataroot ./datasets/market --name market-c3-c5 --camA 3 --camB 5
#python train.py --dataroot ./datasets/market --name market-c3-c6 --camA 3 --camB 6
#
#python train.py --dataroot ./datasets/market --name market-c4-c5 --camA 4 --camB 5
#python train.py --dataroot ./datasets/market --name market-c4-c6 --camA 4 --camB 6
#
#python train.py --dataroot ./datasets/market --name market-c5-c6 --camA 5 --camB 6

#========================VAE-GAN
#python train.py --dataroot ./datasets/market --name vae_market-c1-c2_batch8 --camA 1 --camB 2 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c1-c3_batch8 --camA 1 --camB 3 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c1-c4_batch8 --camA 1 --camB 4 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c1-c5_batch8 --camA 1 --camB 5 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c1-c6_batch8 --camA 1 --camB 6 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8

python train.py --dataroot ./datasets/market --name vae_market-c2-c3_batch8 --camA 2 --camB 3 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c2-c4_batch8 --camA 2 --camB 4 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c2-c5_batch8 --camA 2 --camB 5 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c2-c6_batch8 --camA 2 --camB 6 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8

python train.py --dataroot ./datasets/market --name vae_market-c3-c4_batch8 --camA 3 --camB 4 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c3-c5_batch8 --camA 3 --camB 5 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c3-c6_batch8 --camA 3 --camB 6 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8

python train.py --dataroot ./datasets/market --name vae_market-c4-c5_batch8 --camA 4 --camB 5 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
python train.py --dataroot ./datasets/market --name vae_market-c4-c6_batch8 --camA 4 --camB 6 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8

python train.py --dataroot ./datasets/market --name vae_market-c5-c6_batch8 --camA 5 --camB 6 --which_model_netG VAE --which_model_netD VAE --niter_decay 20 --niter 30 --batchSize 8
