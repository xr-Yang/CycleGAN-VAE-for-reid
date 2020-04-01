#if use function: generate_mat_file 
#python dis_test.py --generate_mat_file --camA_img_path /home/dn003/code/GAN/CamStyle-master_usd/CycleGAN-for-CamStyle/results/market/bounding_box_train_VAE_camstyle_cam_list/1_org_camera/2_ID/2 --camB_img_path /home/dn003/code/GAN/CamStyle-master_usd/CycleGAN-for-CamStyle/results/market/bounding_box_train_VAE_camstyle_cam_list/1_org_camera/3_ID/2 --camA_img_mat ./mat/ID2_cam2_vae --camB_img_mat ./mat/ID2_cam3_vae

#if use functioncalculate the average distance between cameraA and B
#python dis_test.py --distance --camA_img_mat ./mat/ID2_cam2_vae --camB_img_mat ./mat/ID2_cam3_vae

python dis_test.py  --distance --camA_img_path ./data/0002/singan_market/cam1 --camB_img_path ./data/0027/singan_market/cam1 --camA_img_mat ./mat/org/ID0002_cam1 --camB_img_mat ./mat/org/ID0037_cam2
