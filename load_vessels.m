function [I, I_seg] = load_vessels(patient)

  %imagesdir = '/home/haltair/Research/vision/images/Vessel';
  imagesdir = '/cshome/dana/foisy4/DATA/VESSEL12';
  if (patient < 10)
    temp_str = sprintf('0%d',patient);
  end

  if (patient >= 10)
    temp_str = sprintf('%d',patient);
  end

  I = double(mha_read_volume(sprintf('%s/Scans/VESSEL12_%s.mhd',imagesdir,temp_str)));
  
  for i = 1:size(I,3)
    I(:,:,i) = map_image_to_256((I(:,:,i)+1024));  
  end
  
  temp = mex_permute3D_imagedims(double(I),[2 1 3],[size(I,1) size(I,2) size(I,3)]);
  I = double(temp);
  
  I_seg = 255*double(mha_read_volume(sprintf('%s/Lungmasks/VESSEL12_%s.mhd',imagesdir,temp_str)));
  temp = mex_permute3D_imagedims(double(I_seg),[2 1 3],[size(I_seg,1) size(I_seg,2) size(I_seg,3)]);
  I_seg = double(temp);


end

