%-----------------------------Script for brain tumor segmentation--------------------%
close all
clear all

%datadir    = '../experiments/data/Brain_tumor'
datadir = '/home/haltair/Research/vision/experiments/data/Brain_tumor';
%resultsdir = '../experiments/results/Brain_tumor'
resultsdir = '../results/bts';

ERR = 10^-8;

global_params.grid_spacing = [1.0 1.0 1.0];
global_params.delta_t      = 0.01;
global_params.epsilon      = 1.0;
global_params.lambda_1     = 10.0^10;
global_params.lambda_2     = 10.0^10;
global_params.nu           = 0.0;
global_params.mu_chan      = 1.0;
global_params.w            = 20.0;
global_params.MAXiter      = 2;

grade   = 'HG'
class   = 'tumor'

set_params;

dice_score_CV = -1.0*ones(15,1);
dice_score_LR = -1.0*ones(15,1);

for patient = 1:15

  I = double(getfield(load(sprintf('%s/feat_imgs_%s_%d.mat',datadir,grade,patient)),'f_imgs'));
  I = I(:,:,:,1); %the FLAIR modality

  I_seg_gt = double(getfield(load(sprintf('%s/lbl_imgs_%s_%d.mat',datadir,grade,patient)),'lbl_imgs'));
  
  if(strcmp(class,'tumor'))
    I_seg_gt = double(I_seg_gt == 2);
  else
    I_seg_gt = double(I_seg_gt > 0);
  end
  
  [trainP,testP,scaleparams] = bts(patient,'BRATS',grade,class,params,params2);
 
  prob_vol_in  = testP{1} +  ERR;
  prob_vol_out = 1.0-prob_vol_in;
  
  clear trainP;

  I_seg_LR  = double(prob_vol_in > 0.5);
  I_seg_LR = postproc_seg_label(I_seg_LR,I);
  dice_score_LR(patient) = compute_eval_metrics(I_seg_LR,I_seg_gt);
  auto_lbl_imgs_LR = I_seg_LR;

  [LENGTH,WIDTH,HEIGHT] = size(prob_vol_in);
  
  %------------Initialization with a sphere at the center of the image-----------------------%
  center = floor([LENGTH WIDTH HEIGHT]./2);
  radius = floor(max(max(LENGTH,WIDTH),HEIGHT)./4);
  
  phi0  = convert3D_label_distfunc(make_circle(center,radius,[LENGTH WIDTH HEIGHT],[1.0 1.0 1.0],[0.0 0.0 0.0]));
  
  %--------Call to Chan-Vese segmentation method--------%
  I_seg_CV = chanvese3D_segment_precomp_prob(phi0,prob_vol_in,prob_vol_out,global_params);
  
  %-------Post processing the segmentation label----%
  I_seg_CV = postproc_seg_label(I_seg_CV,I);
  
  dice_score_CV(patient) = compute_eval_metrics(I_seg_CV,I_seg_gt);
  
  auto_lbl_imgs_CV = I_seg_CV;

  fid = fopen(sprintf('%s/current_dice_avg.txt',resultsdir),'a');
  fprintf(fid,'patient %d CV %.2f%% current_mean_CV %.2f%% LR %.2f%% current_mean_LR %.2f%%\n',patient,dice_score_CV(patient),sum(dice_score_CV(find(dice_score_CV ~= -1)))./length(find(dice_score_CV ~= -1)),dice_score_LR(patient),sum(dice_score_LR(find(dice_score_LR ~= -1)))./length(find(dice_score_LR ~= -1)));
  fclose(fid);

  auto_lbl_imgs = auto_lbl_imgs_LR;
  dice_score     = dice_score_LR;
  
  save(sprintf('%s/auto_lbl_imgs_LR_%s_%s_%d.mat',resultsdir,grade,class,patient),'auto_lbl_imgs');
  save(sprintf('%s/dice_score_LR_all.mat',resultsdir),'dice_score');

  auto_lbl_imgs = auto_lbl_imgs_CV;
  dice_score    = dice_score_CV;
  
  save(sprintf('%s/auto_lbl_imgs_CV_%s_%s_%d.mat',resultsdir,grade,class,patient),'auto_lbl_imgs');
  save(sprintf('%s/dice_score_CV_all.mat',resultsdir),'dice_score');

end


