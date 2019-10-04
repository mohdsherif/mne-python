"""
.. _tut-inverse-eeg:

Source localization from EEG with MNE/dSPM with a template MRI
==============================================================

This is a combination of two tutorials:
1- Calculate forward solution: /u/mohdsh/software/mneDev/mne-python/tutorials/source-modeling/plot_eeg_no_mri.py
2- Calculate the inverse solution (source localization): /u/mohdsh/software/mneDev/mne-python/tutorials/source-modeling/plot_mne_dspm_source_localization.py

The first half of the tutorial explains how to compute the forward operator from EEG data
using the standard template MRI subject ``fsaverage``.

.. caution:: Source reconstruction without an individual T1 MRI from the
             subject will be less accurate. Do not over interpret
             activity locations which can be off by multiple centimeters.

.. contents:: This tutorial covers:
   :local:
   :depth: 2


"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Joan Massich <mailsik@gmail.com>
#          Mohamed Sherif <mohamed.sherif.md@gmail.com>
#
# License: BSD Style.

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.datasets import eegbci, sample
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

##############################################################################
# Load the data
# -------------
#
# We use here data from one of the subjects in the sample dataset, with the MEG sensors stripped out.
#

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = mne.io.read_raw_fif(raw_fname, preload=True)  # already has an average reference

# Will use the EEG, EOG (for artifact rejection), and STIM (for epoching) channels
raw.pick_types(meg=False, eeg=True, stim=True, eog=True, exclude=[])

###############################################################################
# The EEG electrodes are labelled according to the MGH60 system, with numbers going from EEG001 to 060. 

raw.ch_names

###############################################################################
# For convenience, will relabel channel names according to the 10-05 system
# the sequentially nubmbered electrodes represnet a subset of the 10-05 system, and
# has been named according to the 10-10 system.

# list of standard channel names, ordered sequentially to the EEG electrodes
standard_ch_names = ['Fp1', 'Fpz', 'Fp2', #003
                     'AF7', 'AF3', 'AF4', 'AF8', #007
                     'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', #016
                     'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8', #024
                     'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',#035
                     'TP7', 'TP5', 'CP3', 'CP1', 'CP2', 'CP4', 'TP6', 'TP8',#043
                     'T5', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'T6', #052
                     'PO7', 'PO3', 'PO8', 'PO10', #056
                     'O1', 'Oz', 'O2', #059
                     'Iz'] #060  
                     

# list of EEG channels from the subject
mgh_eeg_names = [mgh_name for mgh_name in raw.ch_names if mgh_name.startswith('EEG')]

# remap the channel names
renames = {mgh_name: standard_name for mgh_name, standard_name in zip(mgh_eeg_names, standard_ch_names) if mgh_name.startswith('EEG')}
raw.rename_channels(renames)

# make sure mapping the channel names is correct
raw.plot_sensors(show_names=True)

# the red sensor is one that has been marked as BAD. It will be excluded when
# we calculate the average ERP. 

# mohdsh - no need for this since the file locations are digitized
# # Read and set the EEG electrode locations
# montage = mne.channels.make_standard_montage('standard_1005')

# raw.set_montage(montage)
# raw.set_eeg_reference(projection=True)  # needed for inverse modeling (for EEG file but not for the loaded MEG/EEG file)

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')

# you can notice that the electrodes are not completely aligned with the head and MRI. This is the case since we are using the fsaverage template

# can use the subject's specific MRI 

# Now you need to create a trans file from the mne coreg to ensure a better alignment of electrodes with the head model and MRI (link to mne.gui.coregistration). 

##############################################################################
# Setup source space and compute forward
# --------------------------------------

fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=2)
print(fwd)

# for illustration purposes use fwd to compute the sensitivity map
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
             clim=dict(lims=[5, 50, 100]))


"""
.. _tut-inverse-methods:

Source localization with MNE/dSPM/sLORETA/eLORETA
=================================================

The aim of the second half of this tutorial is to teach you how to use the computed forward solution to compute and apply a linear
inverse method such as MNE/dSPM/sLORETA/eLORETA on evoked/raw/epochs data.
"""

# sphinx_gallery_thumbnail_number = 10

###############################################################################
# Process MEG data

events = mne.find_events(raw, stim_channel='STI 014')

event_id = dict(aud_l=1)  # event trigger and conditions (auditory stimulation to the left ear) # MEG tut - mohdsh

tmin = -0.2  # start of each epoch (200ms before the trigger)
tmax = 0.5  # end of each epoch (500ms after the trigger)
raw.info['bads'] = ['PO7'] # ['EEG 053']
baseline = (None, 0)  # means from the first instant to t = 0
reject = dict(eog=150e-6)

epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('eeg', 'eog'), baseline=baseline, reject=reject, preload=True)


###############################################################################
# Compute regularized noise covariance
# ------------------------------------
#
# For more details see :ref:`tut_compute_covariance`.

### which methods for covariacnce to use here??? CONTINUE HERERERERERE

noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=True)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

###############################################################################
# Compute the evoked response
# ---------------------------
# Let's just use MEG channels for simplicity. 
# we are using only EEG channels

evoked = epochs.average() # .pick('meg')
evoked.plot(time_unit='s')

evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type='eeg',
                    time_unit='s')

# Show whitening
evoked.plot_white(noise_cov, time_unit='s')

del epochs  # to save memory

###############################################################################
# Inverse modeling: MNE/dSPM on evoked and raw data
# -------------------------------------------------

# mohdsh: no need, since we have calculated the fwd solution ourselves
# # Read the forward solution and compute the inverse operator
# data_path_meg = sample.data_path()
# fname_fwd_meg = data_path_meg + '/MEG/sample/sample_audvis-meg-oct-6-fwd.fif'
# fwd_meg = mne.read_forward_solution(fname_fwd_meg)

# print(fwd_meg)

# make an MEG (now EEG) inverse operator
info = evoked.info
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
del fwd

# You can write it to disk with::
#
#     >>> from mne.minimum_norm import write_inverse_operator
#     >>> write_inverse_operator('sample_audvis-meg-oct-6-inv.fif',
#                                inverse_operator)

###############################################################################
# Compute inverse solution
# ------------------------

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)

###############################################################################
# Visualization
# -------------
# View activation time-series

plt.figure()
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()

###############################################################################
# Examine the original data and the residual after fitting:

fig, axes = plt.subplots(1, 1)
evoked.plot(axes=axes)
for ax in axes:
    ax.texts = []
    for line in ax.lines:
        line.set_color('#98df81')
residual.plot(axes=axes)

###############################################################################
# Here we use peak getter to move visualization to the time point of the peak
# and draw a marker at the maximum peak vertex.

vertno_max, time_max = stc.get_peak(hemi='rh')

# subjects_dir = data_path + '/subjects'
surfer_kwargs = dict(
    hemi='rh', # subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

###############################################################################
# Morph data to average brain - SKIPPED THIS FOR NOW SINCE WE ARE ALREADY USING AN AVERAGE BRAIN


# setup source morph
morph = mne.compute_source_morph(
    src=inverse_operator['src'], subject_from=stc.subject,
    subject_to='fsaverage', spacing=5,  # to ico-5
    subjects_dir=subjects_dir)
# morph data
stc_fsaverage = morph.apply(stc)

brain = stc_fsaverage.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=20)
del stc_fsaverage

###############################################################################
# Dipole orientations
# -------------------
# The ``pick_ori`` parameter of the
# :func:`mne.minimum_norm.apply_inverse` function controls
# the orientation of the dipoles. One useful setting is ``pick_ori='vector'``,
# which will return an estimate that does not only contain the source power at
# each dipole, but also the orientation of the dipoles.

stc_vec = apply_inverse(evoked, inverse_operator, lambda2,
                        method=method, pick_ori='vector')
brain = stc_vec.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Vector solution', 'title', font_size=20)
del stc_vec

###############################################################################
# Note that there is a relationship between the orientation of the dipoles and
# the surface of the cortex. For this reason, we do not use an inflated
# cortical surface for visualization, but the original surface used to define
# the source space.
#
# For more information about dipole orientations, see
# :ref:`tut-dipole-orientations`.

###############################################################################
# # Now let's look at each solver:

# for mi, (method, lims) in enumerate((('dSPM', [8, 12, 15]),
#                                      ('sLORETA', [3, 5, 7]),
#                                      ('eLORETA', [0.75, 1.25, 1.75]),)):
#     surfer_kwargs['clim']['lims'] = lims
#     stc = apply_inverse(evoked, inverse_operator, lambda2,
#                         method=method, pick_ori=None)
#     brain = stc.plot(figure=mi, **surfer_kwargs)
#     brain.add_text(0.1, 0.9, method, 'title', font_size=20)
#     del stc


