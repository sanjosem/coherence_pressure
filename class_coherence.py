class coherence:
    def __init__(self,filename):
        from os.path import exists
        if exists(filename):
            self.filename = filename
        else:
            print('File {0:s} does not exists'.format(filename))
            self.filename = None
        self.fid = None
        self.coords = dict()
        self.refs = dict()
        self.disps = dict()
        self.time_info = dict()
        self.phys_params = dict()
        self.space_params = dict()
        self.fft_params = dict()

    def open_file(self):
        import h5py
        if self.filename is not None:
            self.fid = h5py.File(self.filename,'r')

    def read_coordinates(self):
        if self.fid is None:
            self.open_file()
        for var in ['x','y','z']:
            self.coords[var] = self.fid['mesh/{0:s}'.format(var)][()]
        nx,nz = self.coords['x'].shape
        self.space_params['nx'] = nx
        self.space_params['nz'] = nz
        print('mesh dimensions:',(nx,nz))

    def set_ref(self,i_ref,j_ref):
        if len(self.coords) == 0:
            self.read_coordinates()
        if i_ref<0 or i_ref>=self.space_params['nx'] :
            print('Wrong i_ref index')
        elif j_ref<0 or j_ref>=self.space_params['nz'] :
            print('Wrong j_ref index')
        else:
            self.space_params['iref'] = i_ref
            self.space_params['jref'] = j_ref
            self.calc_displacement()
            self.space_params['xref'] = self.coords['x'][i_ref,j_ref]
            self.space_params['yref'] = self.coords['y'][i_ref,j_ref]
            self.space_params['zref'] = self.coords['z'][i_ref,j_ref]
            print('Reference position',(self.space_params['xref'],self.space_params['yref'],
                    self.space_params['zref']),'[m]')


    def set_phys_params(self,u_ref,chord):
        self.phys_params['u'] = u_ref
        self.phys_params['chord'] = chord
        self.phys_params['tconv'] = chord/u_ref

    def calc_displacement(self):
        from numpy import sign,sqrt
        if len(self.space_params.keys()) > 0:
            for var in ['x','y','z']:
                self.disps[var] = ( self.coords[var] - self.coords[var][self.space_params['iref'],self.space_params['jref']])
            # use the sign to have negative value for probes upstream of ref
            self.disps['s'] = sign(self.disps['x']) * sqrt( self.disps['x']**2 + self.disps['y']**2 )
        else:
            print('Please define reference indices with set_ref')


    def read_time(self):
        if self.fid is None:
            self.open_file()
        self.time_info['time'] = self.fid['parameters/time'][()]
        self.compute_time_info()

    def compute_time_info(self):
        if len(self.time_info.keys()) == 0:
            self.read_time()

        self.time_info['dt'] = self.time_info['time'][1] - self.time_info['time'][0]
        self.time_info['fs'] = 1.0/self.time_info['dt']

        print('Timestep',self.time_info['dt'],'[s]')
        print('Sampling frequency',self.time_info['fs'],'[Hz]')
        Tlen = self.time_info['time'][-1]-self.time_info['time'][0]
        print('Time length',Tlen,'[s]')
        if len(self.phys_params.keys()) > 0:
            print('Time length',Tlen/self.phys_params['tconv'],'[-]')

    def set_fft_params(self,n_chunk,win_name="hanning",filter=False,filt_order=4,
                        filt_freq = None):
        from scipy.signal import butter
        self.fft_params['n_chunk'] = n_chunk
        self.fft_params['win'] = win_name
        if not 'dt' in  self.time_info.keys():
            self.compute_time_info()


        nlen = self.time_info['time'].size
        nperseg = nlen//n_chunk
        self.fft_params['nperseg'] = nperseg

        nfft = next_greater_power_of_2(nperseg)
        self.fft_params['nfft'] = nfft

        dt = self.time_info['dt']
        print('Chunk duration',nperseg*dt,'[s]')
        if len(self.phys_params.keys()) > 0:
            print('Chunk duration',nperseg*dt/self.phys_params['tconv'],'[-]')
        print('Frequency resolution',1/(nperseg*dt),'[Hz]')
        print('nfft',nfft)

        if filter:
            self.fft_params['filter'] = filter
            if filt_freq is None:
                filt_freq = -5
            if filt_freq<0:
                filt_freq = -(filt_freq)/(nperseg*dt)
            self.fft_params['filt_freq'] = filt_freq
            self.fft_params['sos'] = butter(filt_order,filt_freq,'highpass',
                                            fs=self.time_info['fs'],output='sos')
            print('Filtering activated')
            print('High-pass Butterworth filter of order',filt_order)
            print('Cut-off frequency',filt_freq,'[Hz]')
        else:
            self.fft_params['filter'] = False
            print('No Filtering activated')

    def compute_span_coherence_lengthscale(self,ni_avg=0):
        from time import perf_counter as gt
        from scipy.signal import sosfilt, spectrogram
        from numpy import mean, abs, newaxis, sqrt
        from numpy.fft import rfft, rfftfreq,fft
        iref = self.space_params['iref']
        jref = self.space_params['jref']
        self.fft_params['n_stream_avg'] = ni_avg
        i_line = slice(iref-ni_avg,iref+ni_avg+1)
        ds = self.disps['s'][i_line,jref][-1]-self.disps['s'][i_line,jref][0]
        self.span_coh = dict()
        self.span_coh['ds'] = ds
        self.span_coh['dz'] = self.disps['z'][iref,:]
        self.span_coh['z_span'] = self.span_coh['dz'][-1]-self.span_coh['dz'][0]
        self.span_coh['dz_discr'] = self.span_coh['dz'][1]-self.span_coh['dz'][0]
        print('Streamwise averaging:',ds*1000,'[mm]')
        print('Span extends:',self.span_coh['z_span']*1000,'[mm]')
        print('Spanwise discretization:',self.span_coh['dz_discr']*1000,'[mm]')

        # read pressure signal (time,2*ni_avg+1,nz)
        tic = gt()
        pressure = self.fid["pressure"][:,i_line,:]
        toc = gt()-tic
        print('read signal - ',toc,'[s]')

        # detrend
        tic = gt()
        p_mean =  pressure.mean(axis=(0,1),keepdims=True)
        pf_cur = pressure - p_mean
        toc = gt() - tic
        print('detrend signal - ',toc,'[s]')

        # filter
        if self.fft_params['filter']:
            tic = gt()
            pf_cur = sosfilt(self.fft_params['sos'],pf_cur,axis=0)
            toc = gt() - tic
            print('signal filtering - ',toc,'[s]')

        # compute spectrogram
        tic = gt()
        nperseg = self.fft_params['nperseg']
        noverlap = 2*nperseg//3
        f,t_win,p_chapo = spectrogram(pf_cur,fs=self.time_info['fs'],window=self.fft_params['win'],
                                      nperseg=nperseg,nfft=self.fft_params['nfft'],noverlap=noverlap,
                                      detrend='constant',axis=0,mode='complex',scaling='density')
        toc = gt() - tic
        print('compute periodogram - ',toc,'[s]')
        self.span_coh['frequency'] = f
        self.span_coh['n_win'] = t_win.size

        # compute pressure spectral density
        tic = gt()
        Pxx = mean(p_chapo * p_chapo.conj(),axis=(1,3)).real
        toc = gt() - tic
        print('compute psd - ',toc,'[s]')
        self.span_coh['Pxx'] = Pxx

        # compute psd
        tic = gt()
        p_chapo_ref_star = p_chapo[:,:,self.space_params['jref'],:].conj()
        Pxy_full = p_chapo * p_chapo_ref_star[:,:,newaxis,:]
        # time and space sample averaging in coherence
        Pxy = mean(Pxy_full,axis=(1,3)) # Same as scipy.signal.csd
        toc = gt() - tic
        print('compute csd - ',toc,'[s]')

        # Compute coherence
        tic = gt()
        coh = abs(Pxy)**2 / (Pxx * Pxx[:,jref][:,newaxis]) # Same as scipy.signal.coherence
        self.span_coh['gamma2'] = coh
        toc = gt()- tic
        print('compute coherence of averaged csd - ',toc,'[s]')
        tic = gt()
        coh_nosqrt = Pxy_full / ((Pxx * Pxx[:,jref][:,newaxis])**0.5)[:,newaxis,:,newaxis]
        toc = gt() - tic
        print('compute coherence of full csd samples - ',toc,'[s]')

        # Compute lengthscale
        delta_z = self.span_coh['dz_discr']
        kz = rfftfreq(self.space_params['nz'], delta_z)
        self.span_coh['kz'] = kz
        tic = gt()
        lz_meth1 = 0.5*rfft(sqrt(coh),axis=1).real * delta_z
        toc = gt() - tic
        print('compute length scale with avg in csd - ',toc,'[s]')
        tic = gt()
        lz_meth2 = 0.5*mean(fft(coh_nosqrt,axis=1),axis=(1,3)).real * delta_z
        toc = gt() - tic
        print('compute length scale with avg at end - ',toc,'[s]')
        self.span_coh['lz_meth1'] = lz_meth1
        self.span_coh['lz_meth2'] = lz_meth2

    def save_results(self,name):
        import h5py
        if 'span_coh' in self.__dict__.keys():
            output_filename = 'spanwise_coherence_{0:s}.h5'.format(name)
            fout = h5py.File(output_filename,'w')
            for key in self.span_coh.keys():
                fout[key] = self.span_coh[key]
            for key in ['xref','yref','zref']:
                fout[key] = self.space_params[key]
            fout.close()
            print('-> File',output_filename,'saved')

def next_greater_power_of_2(n):
    return 2**(n-1).bit_length()
