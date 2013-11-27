function test_tutorial_beamformerextended20131122

% MEM 8gb
% WALLTIME 01:45:00

% TEST test_tutorial_beamformer20131122
% TEST ft_sourceanalysis ft_prepare_leadfield

% this test script represents the MATLAB code from http://fieldtrip.fcdonders.nl/tutorial/beamformer
% as downloaded from the wiki on 22 November 2013

global ft_default
ft_default = [];

if ispc
  dccnfilename = @(filename) strrep(strrep(filename,'/home','H:'),'/','\');
else
  dccnfilename = @(filename) strrep(strrep(filename,'H:','/home'),'\','/');
end

cd(dccnfilename('/home/common/matlab/fieldtrip/data/ftp/tutorial/sensor_analysis'));
load subjectK

data_combined = ft_appenddata([], data_left, data_right);

cd(dccnfilename('/home/common/matlab/fieldtrip/data/ftp/tutorial/beamformer_extended'));
load segmentedmri

mri = ft_read_mri('subjectK.mri');

cfg          = [];
cfg.coordsys = 'ctf'; % the MRI is expressed in the CTF coordinate system, see below
segmentedmri = ft_volumesegment(cfg, mri);

% add anatomical information to the segmentation
segmentedmri.transform = mri.transform;
segmentedmri.anatomy   = mri.anatomy;

cfg              = [];
cfg.funparameter = 'gray';
ft_sourceplot(cfg,segmentedmri);

cfg        = [];
cfg.method = 'singleshell';
hdm        = ft_prepare_headmodel(cfg, segmentedmri);

if ispc
  templatedir  = 'H:/common/matlab/fieldtrip/template/sourcemodel';
elseif isunix
  templatedir  = '/home/common/matlab/fieldtrip/template/sourcemodel';
end
template = load(fullfile(templatedir, 'standard_sourcemodel3d8mm'));

% inverse-warp the subject specific grid to the template grid
cfg                = [];
cfg.grid.warpmni   = 'yes';
cfg.grid.template  = template.sourcemodel;
cfg.grid.nonlinear = 'yes'; % use non-linear normalization
cfg.mri            = mri;
sourcemodel        = ft_prepare_sourcemodel(cfg);

hdm_cm = ft_convert_units(hdm, 'cm');

figure; hold on     % plot all objects in one figure
ft_plot_vol(hdm_cm, 'edgecolor', 'none')
alpha 0.4           % make the surface transparent
ft_plot_mesh(sourcemodel.pos(sourcemodel.inside,:));
ft_plot_sens(data_combined.grad);

cfg           = [];
cfg.toilim    = [-0.8 1.1];
cfg.minlength = 'maxperlen'; % this ensures all resulting trials are equal length
data          = ft_redefinetrial(cfg, data_combined);

cfg        = [];
cfg.toilim = [-0.8 0];
data_bsl   = ft_redefinetrial(cfg, data);

cfg.toilim = [0.3 1.1];
data_exp   = ft_redefinetrial(cfg, data);

cfg      = [];
data_cmb = ft_appenddata(cfg, data_bsl, data_exp);

data_cmb.trialinfo = [zeros(length(data_bsl.trial), 1); ones(length(data_exp.trial), 1)];

cfg            = [];
cfg.method     = 'mtmfft';
cfg.output     = 'fourier';
cfg.keeptrials = 'yes';
cfg.tapsmofrq  = 15;
cfg.foi        = 55;
freq_cmb       = ft_freqanalysis(cfg, data_cmb);

cfg                = [];
cfg.trials         = freq_cmb.trialinfo == 0;
freq_bsl           = ft_selectdata(cfg, freq_cmb);
% remember the number of tapers per trial
freq_bsl.cumtapcnt = freq_cmb.cumtapcnt(cfg.trials);
freq_bsl.cumsumcnt = freq_cmb.cumsumcnt(cfg.trials);

cfg.trials         = freq_cmb.trialinfo == 1;
freq_exp           = ft_selectdata(cfg, freq_cmb);
% remember the number of tapers per trial
freq_exp.cumtapcnt = freq_cmb.cumtapcnt(cfg.trials);
freq_exp.cumsumcnt = freq_cmb.cumsumcnt(cfg.trials);

cfg             = [];
cfg.grid        = sourcemodel;
cfg.vol         = hdm;
cfg.channel     = {'MEG'};
cfg.grad        = freq_cmb.grad;
sourcemodel_lf  = ft_prepare_leadfield(cfg, freq_cmb);

cfg                   = [];
cfg.frequency         = freq_cmb.freq;
cfg.grad              = freq_cmb.grad;
cfg.method            = 'dics';
cfg.keeptrials        = 'yes';
cfg.grid              = sourcemodel_lf;
cfg.vol               = hdm;
cfg.keeptrials        = 'yes';
cfg.dics.lambda       = '5%';
cfg.dics.keepfilter   = 'yes';
cfg.dics.fixedori     = 'yes';
cfg.dics.realfilter   = 'yes';
source                = ft_sourceanalysis(cfg, freq_cmb);

% beam pre- and poststim by using the common filter
cfg.grid.filter  = source.avg.filter;
source_bsl       = ft_sourceanalysis(cfg, freq_bsl);
source_exp       = ft_sourceanalysis(cfg, freq_exp);

source_diff = source_exp;
source_diff.avg.pow = (source_exp.avg.pow ./ source_bsl.avg.pow) - 1;

source_diff.pos = template.sourcemodel.pos;
source_diff.dim = template.sourcemodel.dim;

% note that the exact directory is user- and platform-specific
if isunix
  templatefile = '/home/common/matlab/fieldtrip/external/spm8/templates/T1.nii';
elseif ispc
  templatefile = 'H:\common\matlab\fieldtrip\external\spm8\templates/T1.nii';
end
template_mri = ft_read_mri(templatefile);

cfg              = [];
cfg.voxelcoord   = 'no';
cfg.parameter    = 'avg.pow';
cfg.interpmethod = 'nearest';
cfg.coordsys     = 'mni';
source_diff_int  = ft_sourceinterpolate(cfg, source_diff, template_mri);

cfg               = [];
cfg.method        = 'slice';
cfg.coordsys      = 'mni';
cfg.funparameter  = 'avg.pow';
cfg.maskparameter = cfg.funparameter;
cfg.funcolorlim   = [0.0 1.2];
cfg.opacitylim    = [0.0 1.2];
cfg.opacitymap    = 'rampup';
ft_sourceplot(cfg,source_diff_int);

cfg                 = [];
cfg.toilim          = [-1 -0.0025];
cfg.minlength       = 'maxperlen'; % this ensures all resulting trials are equal length
data_stim           = ft_redefinetrial(cfg, data);

cfg                 = [];
cfg.output          = 'powandcsd';
cfg.method          = 'mtmfft';
cfg.taper           = 'dpss';
cfg.tapsmofrq       = 5;
cfg.foi             = 20;
cfg.keeptrials      = 'yes';
cfg.channel         = {'MEG' 'EMGlft' 'EMGrgt'};
cfg.channelcmb      = {'MEG' 'MEG'; 'MEG' 'EMGlft'; 'MEG' 'EMGrgt'};
freq_csd            = ft_freqanalysis(cfg, data_stim);

cfg                 = [];
cfg.method          = 'dics';
cfg.refchan         = 'EMGlft';
cfg.frequency       = 20;
cfg.vol             = hdm;
cfg.grid            = sourcemodel;
source_coh_lft      = ft_sourceanalysis(cfg, freq_csd);

source_coh_lft.pos = template.sourcemodel.pos;
source_coh_lft.dim = template.sourcemodel.dim;

% note that the exact directory is user-specific
if isunix
  templatefile = '/home/common/matlab/fieldtrip/external/spm8/templates/T1.nii';
elseif ispc
  templatefile = 'H:\common\matlab\fieldtrip\external\spm8\templates/T1.nii';
end
template_mri = ft_read_mri(templatefile);

cfg              = [];
cfg.voxelcoord   = 'no';
cfg.parameter    = 'coh';
cfg.interpmethod = 'nearest';
cfg.coordsys     = 'mni';
source_coh_int   = ft_sourceinterpolate(cfg, source_coh_lft, template_mri);

cfg              = [];
cfg.method       = 'ortho';
cfg.funparameter = 'coh';
cfg.coordsys     = 'mni';
ft_sourceplot(cfg, source_coh_int);
