function [I,I_seg] = load_braintumor(class,grade,patient)

  %basedir = '/home/haltair/Research/vision/images/Brain_tumor/BRATS-1';
  basedir = '/usr/data/medical_images/BRATS2012_braintumor/BRATS-1';

  if (patient < 10)
    temp_str = sprintf('0%d',patient);  
  end
  
  if (patient >= 10)    
    temp_str = sprintf('%d',patient);  
  end
  
  I_FLAIR = double(mha_read_volume(sprintf('%s/Images/%s_%s00%s/%s_%s00%s_FLAIR.mha',basedir,class,grade,temp_str,class,grade,temp_str)));
  temp = mex_permute3D_imagedims(double(I_FLAIR),[2 1 3],size(I_FLAIR));
  I_FLAIR = temp;

  for i = 1:size(I_FLAIR,3)
    I_FLAIR(:,:,i) = map_image_to_256(I_FLAIR(:,:,i));
  end

  I_T1    = double(mha_read_volume(sprintf('%s/Images/%s_%s00%s/%s_%s00%s_T1.mha',basedir,class,grade,temp_str,class,grade,temp_str)));
  temp = mex_permute3D_imagedims(double(I_T1),[2 1 3],size(I_T1));
  I_T1 = temp;
  I_T1 = map_image_to_256(I_T1);
  
  for i = 1:size(I_T1,3)
    I_T1(:,:,i) = map_image_to_256(I_T1(:,:,i));
  end


  I_T1C   = double(mha_read_volume(sprintf('%s/Images/%s_%s00%s/%s_%s00%s_T1C.mha',basedir,class,grade,temp_str,class,grade,temp_str)));
  temp = mex_permute3D_imagedims(double(I_T1C),[2 1 3],size(I_T1C));
  I_T1C = temp;
  I_T1C = map_image_to_256(I_T1C);

  for i = 1:size(I_T1C,3)
    I_T1C(:,:,i) = map_image_to_256(I_T1C(:,:,i));
  end

  I_T2    = double(mha_read_volume(sprintf('%s/Images/%s_%s00%s/%s_%s00%s_T2.mha',basedir,class,grade,temp_str,class,grade,temp_str)));
  temp = mex_permute3D_imagedims(double(I_T2),[2 1 3],size(I_T2));
  I_T2 = temp;
  I_T2 = map_image_to_256(I_T2);

  for i = 1:size(I_T2,3)
    I_T2(:,:,i) = map_image_to_256(I_T2(:,:,i));
  end
  
  I = zeros([size(I_FLAIR) 4]);
  
  I(:,:,:,1) = I_FLAIR;
  I(:,:,:,2) = I_T1;
  I(:,:,:,3) = I_T1C;
  I(:,:,:,4) = I_T2;
  
  I_seg = double(mha_read_volume(sprintf('%s/Truth/%s_%s00%s_truth.mha',basedir,class,grade,temp_str)));
  temp  = mex_permute3D_imagedims(double(I_seg),[2 1 3],size(I_seg));
  I_seg = temp;
