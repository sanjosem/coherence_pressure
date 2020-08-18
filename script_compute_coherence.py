import class_coherence
import os

chord = 0.1356
u_ref = 16.0
filename = os.path.join(os.getenv('SLURM_TMPDIR'),"itp_mesh_struct_ss_hr_1000_257_total.hdf5")
cc = class_coherence.coherence(filename)
cc.read_time()
cc.read_coordinates()
cc.set_phys_params(u_ref,chord)

j_ref = 127
for i_ref in range(10,1000,10):
    cc.set_ref(i_ref,j_ref)
    for n_chunk in [4,8,18,32,36]:
        cc.set_fft_params(n_chunk,'hanning',filter=True,filt_freq=-3)
        cc.compute_span_coherence_lengthscale()
        if cc.fft_params['filter']:
            end_name = '_filt{0:03.0f}Hz'.format(cc.fft_params['filt_freq'])
        else:
            end_name = ''
        name = 'i{0:03d}_j{1:03d}_twin{2:03d}'.format(
                    cc.space_params['iref'],cc.space_params['jref'],cc.span_coh['n_win']) + end_name
        cc.save_results(name)

    for n_chunk in [4,8,18,32,36]:
        cc.set_fft_params(n_chunk,'hanning')
        cc.compute_span_coherence_lengthscale()
        if cc.fft_params['filter']:
            end_name = '_filt{0:03.0f}Hz'.format(cc.fft_params['filt_freq'])
        else:
            end_name = ''
        name = 'i{0:03d}_j{1:03d}_twin{2:03d}'.format(
                    cc.space_params['iref'],cc.space_params['jref'],cc.span_coh['n_win']) + end_name
        cc.save_results(name)
