## test all cameras for market
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c1-c2 --camA 1 --camB 2 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c1-c3 --camA 1 --camB 3 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c1-c4 --camA 1 --camB 4 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c1-c5 --camA 1 --camB 5 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c1-c6 --camA 1 --camB 6 --save_root results/market/bounding_box_test1-2
#
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c2-c3 --camA 2 --camB 3 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c2-c4 --camA 2 --camB 4 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c2-c5 --camA 2 --camB 5 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c2-c6 --camA 2 --camB 6 --save_root results/market/bounding_box_test1-2
#
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c3-c4 --camA 3 --camB 4 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c3-c5 --camA 3 --camB 5 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c3-c6 --camA 3 --camB 6 --save_root results/market/bounding_box_test1-2
#
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c4-c5 --camA 4 --camB 5 --save_root results/market/bounding_box_test1-2
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c4-c6 --camA 4 --camB 6 --save_root results/market/bounding_box_test1-2
#
#python test.py --dataroot ./datasets/market/bounding_box_test --name market-c5-c6 --camA 5 --camB 6 --save_root results/market/bounding_box_test1-2
## test all cameras for market
#python test.py --dataroot ./datasets/market/query --name market-c1-c2 --camA 1 --camB 2 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c1-c3 --camA 1 --camB 3 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c1-c4 --camA 1 --camB 4 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c1-c5 --camA 1 --camB 5 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c1-c6 --camA 1 --camB 6 --save_root results/market/query-2
#
#python test.py --dataroot ./datasets/market/query --name market-c2-c3 --camA 2 --camB 3 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c2-c4 --camA 2 --camB 4 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c2-c5 --camA 2 --camB 5 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c2-c6 --camA 2 --camB 6 --save_root results/market/query-2
#
#python test.py --dataroot ./datasets/market/query --name market-c3-c4 --camA 3 --camB 4 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c3-c5 --camA 3 --camB 5 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c3-c6 --camA 3 --camB 6 --save_root results/market/query-2
#
#python test.py --dataroot ./datasets/market/query --name market-c4-c5 --camA 4 --camB 5 --save_root results/market/query-2
#python test.py --dataroot ./datasets/market/query --name market-c4-c6 --camA 4 --camB 6 --save_root results/market/query-2
#
#python test.py --dataroot ./datasets/market/query --name market-c5-c6 --camA 5 --camB 6 --save_root results/market/query-2

#========================VAE-test
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c1-c2_batch8 --camA 1 --camB 2 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c1-c3_batch8 --camA 1 --camB 3 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c1-c4_batch8 --camA 1 --camB 4 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c1-c5_batch8 --camA 1 --camB 5 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c1-c6_batch8 --camA 1 --camB 6 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE

python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c2-c3_batch8 --camA 2 --camB 3 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c2-c4_batch8 --camA 2 --camB 4 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c2-c5_batch8 --camA 2 --camB 5 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c2-c6_batch8 --camA 2 --camB 6 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE

python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c3-c4_batch8 --camA 3 --camB 4 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c3-c5_batch8 --camA 3 --camB 5 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c3-c6_batch8 --camA 3 --camB 6 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE

python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c4-c5_batch8 --camA 4 --camB 5 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c4-c6_batch8 --camA 4 --camB 6 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE

python test.py --dataroot ./datasets/market --name vae-batch8/vae_market-c5-c6_batch8 --camA 5 --camB 6 --save_root results/market/bounding_box_train_camstyle_VAE_batchsize8 --which_model_netG VAE --which_model_netD VAE
