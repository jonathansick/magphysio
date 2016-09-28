#!/usr/bin/env python
# encoding: utf-8
"""
Python readers/representation of MAGPHYS result files.

2014-12-17 - Created by Jonathan Sick
"""

import numpy as np

import astropy.units as u
import astropy.constants as const


class BaseReader(object):
    def __init__(self, distance=785. * u.kpc):
        super(BaseReader, self).__init__()
        self._distance = distance
        self._full_sed = None

    def convert_LsunHz_to_Jy(self, Lsun_Hz):
        """Convert the flux reported by MAGPHYS, in units L_sun / Hz, to Jy."""
        f_sun = (const.L_sun.cgs /
                 (4. * np.pi * self._distance ** 2)).decompose(
            bases=[u.erg, u.cm, u.s])
        f = (Lsun_Hz / u.Hz * f_sun).to(u.Jy)
        assert f.unit == u.Jy
        return f.value

    def _parse_observed_sed(self, lines, index=1):
        bands = lines[index].replace('#', '').strip().split()
        sed = self.convert_LsunHz_to_Jy(
            np.array(map(float, lines[index + 1].strip().split())))
        err = self.convert_LsunHz_to_Jy(
            np.array(map(float, lines[index + 2].strip().split())))
        return bands, sed, err

    def _parse_model_sed(self, lines, index=11):
        bands = lines[index].replace('#', '').strip().split()
        sed = self.convert_LsunHz_to_Jy(
            np.array(map(float, lines[index + 1].strip().split())))
        return bands, sed

    @staticmethod
    def _parse_best_fit(lines, index=8):
        parts = lines[index].strip().split()
        sfh_i = int(parts[0])
        chi2 = float(parts[1])
        z = float(parts[2])
        return sfh_i, chi2, z

    @staticmethod
    def _parse_pdf(lines, start_n, percentile_n):
        end_n = percentile_n - 1  # line where the PDF grid ends
        n_bins = end_n - start_n
        bins = np.empty(n_bins, dtype=np.float)
        probs = np.empty(n_bins, dtype=np.float)
        # print(lines[start_n - 1])
        for i, line in enumerate(lines[start_n:end_n]):
            b, p = map(float, line.strip().split())
            bins[i] = b
            probs[i] = p
        percentiles = map(float, lines[percentile_n].strip().split())
        d = {"bins": bins,
             "probs": probs,
             "2.5": percentiles[0],
             "16": percentiles[1],
             "50": percentiles[2],
             "84": percentiles[3],
             "97.5": percentiles[4]}
        return d

    def persist(self, f, path=None):
        """Persist the MAGPHYS fit to a hierarchical HDF5 file.

        By default, the dataset is stored in the group `/models/{galaxy_id}`.

        e.g.::

            import h5py
            f = h5py.File("test.hdf5", "a")
            model.persist(f)
        """
        if path is None:
            path = "models/{0}".format(self.galaxy_id)
        if path in f:
            # Delete existing fit archives
            del f[path]
        f.create_group(path)
        group = f[path]

        # Persist SED (per bandpass)
        sed = np.vstack([self.sed.T, self.sed_err.T, self.model_sed.T])
        group['sed'] = sed
        group['sed'].attrs['bands'] = self.bands
        group['sed'].attrs['i_sh'] = self.i_sfh
        group['sed'].attrs['chi2'] = self.chi2

        # Perist *full SED* (wavelength grid)
        if self._full_sed is not None:
            group['full_sed'] = self._full_sed

        # Persist PDFs
        for k, doc in self._pdfs.iteritems():
            group[k] = np.vstack([doc['bins'].T, doc['probs'].T])
            dset = group[k]
            dset.attrs['name'] = k
            dset.attrs['2.5'] = doc['2.5']
            dset.attrs['16'] = doc['16']
            dset.attrs['50'] = doc['50']
            dset.attrs['84'] = doc['84']
            dset.attrs['97.5'] = doc['97.5']


class MagphysFit(BaseReader):
    """A regular MAGPHYS model fit."""
    def __init__(self, galaxy_id, fit_obj, sed_obj=None):
        super(MagphysFit, self).__init__()
        self.galaxy_id = galaxy_id
        self._pdfs = {}

        if type(fit_obj) is str:
            with open(fit_obj) as fit_file:
                fit_lines = fit_file.readlines()
        else:
            fit_lines = fit_obj.readlines()  # already a file object

        self.bands, self.sed, self.sed_err = self._parse_observed_sed(
            fit_lines)
        _, self.model_sed = self._parse_model_sed(fit_lines)

        self.i_sfh, self.chi2, self.z = self._parse_best_fit(fit_lines)
        self._pdfs['f_mu_sfh'] = self._parse_pdf(fit_lines, 16, 37)
        self._pdfs['f_mu_ir'] = self._parse_pdf(fit_lines, 39, 60)
        self._pdfs['mu'] = self._parse_pdf(fit_lines, 62, 83)
        self._pdfs['tau_V'] = self._parse_pdf(fit_lines, 85, 134)
        self._pdfs['sSFR_0.1Gyr'] = self._parse_pdf(fit_lines, 136, 207)
        self._pdfs['log_Mstar'] = self._parse_pdf(fit_lines, 209, 270)
        self._pdfs['log_Ldust'] = self._parse_pdf(fit_lines, 272, 333)
        self._pdfs['T_C_ISM'] = self._parse_pdf(fit_lines, 335, 346)
        self._pdfs['T_W_BC'] = self._parse_pdf(fit_lines, 348, 379)
        self._pdfs['xi_C_tot'] = self._parse_pdf(fit_lines, 381, 402)
        self._pdfs['xi_PAH_tot'] = self._parse_pdf(fit_lines, 404, 425)
        self._pdfs['xi_MIR_tot'] = self._parse_pdf(fit_lines, 427, 448)
        self._pdfs['xi_W_tot'] = self._parse_pdf(fit_lines, 450, 471)
        self._pdfs['tau_V_ISM'] = self._parse_pdf(fit_lines, 473, 554)
        self._pdfs['log_Mdust'] = self._parse_pdf(fit_lines, 556, 617)
        self._pdfs['SFR_0.1Gyr'] = self._parse_pdf(fit_lines, 619, 680)

        if sed_obj is not None:
            # ...Spectral Energy Distribution [lg(L_lambda/LoA^-1)]:
            # ...lg(lambda/A)...Attenuated...Unattenuated
            dt = np.dtype([('log_lambda_A', np.float),
                           ('log_L_Attenuated', np.float),
                           ('log_L_Unattenuated', np.float)])
            self._full_sed = np.loadtxt(sed_obj, skiprows=10, dtype=dt)


class EnhancedMagphysFit(BaseReader):
    """A enhanced MAGPHYS model fit that includes metallicity, age, etc fit."""
    def __init__(self, galaxy_id, fit_obj, distance=785. * u.kpc,
                 sed_obj=None):
        super(EnhancedMagphysFit, self).__init__(distance=distance)
        self.galaxy_id = galaxy_id
        self._pdfs = {}

        if type(fit_obj) is str:
            with open(fit_obj) as fit_file:
                fit_lines = fit_file.readlines()
        else:
            fit_lines = fit_obj.readlines()  # already a file object

        self.bands, self.sed, self.sed_err = self._parse_observed_sed(
            fit_lines)
        _, self.model_sed = self._parse_model_sed(fit_lines)

        self.i_sfh, self.chi2, self.z = self._parse_best_fit(fit_lines)
        self._pdfs['Z_Zo'] = self._parse_pdf(fit_lines, 16, 120)
        self._pdfs['tform'] = self._parse_pdf(fit_lines, 122, 258)
        self._pdfs['gamma'] = self._parse_pdf(fit_lines, 260, 361)
        self._pdfs['t_lastB'] = self._parse_pdf(fit_lines, 363, 500)
        self._pdfs['log_age_M'] = self._parse_pdf(fit_lines, 502, 628)

        self._pdfs['f_mu_sfh'] = self._parse_pdf(fit_lines, 630, 651)
        self._pdfs['f_mu_ir'] = self._parse_pdf(fit_lines, 653, 674)
        self._pdfs['mu'] = self._parse_pdf(fit_lines, 676, 697)
        self._pdfs['tau_V'] = self._parse_pdf(fit_lines, 699, 748)
        self._pdfs['sSFR_0.1Gyr'] = self._parse_pdf(fit_lines, 750, 821)
        self._pdfs['log_Mstar'] = self._parse_pdf(fit_lines, 823, 884)
        self._pdfs['log_Ldust'] = self._parse_pdf(fit_lines, 886, 947)
        self._pdfs['T_C_ISM'] = self._parse_pdf(fit_lines, 949, 960)
        self._pdfs['T_W_BC'] = self._parse_pdf(fit_lines, 962, 993)
        self._pdfs['xi_C_tot'] = self._parse_pdf(fit_lines, 995, 1016)
        self._pdfs['xi_PAH_tot'] = self._parse_pdf(fit_lines, 1018, 1039)
        self._pdfs['xi_MIR_tot'] = self._parse_pdf(fit_lines, 1041, 1062)
        self._pdfs['xi_W_tot'] = self._parse_pdf(fit_lines, 1064, 1085)
        self._pdfs['tau_V_ISM'] = self._parse_pdf(fit_lines, 1087, 1168)
        self._pdfs['log_Mdust'] = self._parse_pdf(fit_lines, 1170, 1231)
        self._pdfs['SFR_0.1Gyr'] = self._parse_pdf(fit_lines, 1233, 1294)

        if sed_obj is not None:
            # ...Spectral Energy Distribution [lg(L_lambda/LoA^-1)]:
            # ...lg(lambda/A)...Attenuated...Unattenuated
            dt = np.dtype([('lambda', np.float),
                           ('sed_attenuated', np.float),
                           ('sed_unattenuated', np.float)])
            self._full_sed = np.loadtxt(sed_obj, skiprows=10, dtype=dt)

            # convert wavelength from log angstrom to microns
            log_angstrom = self._full_sed['lambda']
            lamb = ((10. ** log_angstrom) * u.angstrom).to(u.micron)
            self._full_sed['lambda'] = lamb.value

            # convert full SED to log (lambda L / L_sun)
            attenuated = np.log10(lamb.to(u.angstrom) *
                                  10. ** self._full_sed['sed_attenuated'] /
                                  u.angstrom)
            self._full_sed['sed_attenuated'] = attenuated

            unattenuated = np.log10(lamb.to(u.angstrom) *
                                    10. ** self._full_sed['sed_unattenuated'] /
                                    u.angstrom)
            self._full_sed['sed_unattenuated'] = unattenuated


class OpticalFit(BaseReader):
    """A MAGPHYS model fit for Roediger's fit_magphys_opt.exe mod."""
    def __init__(self, galaxy_id, fit_obj, sed_obj=None):
        super(OpticalFit, self).__init__()
        self.galaxy_id = galaxy_id
        self._pdfs = {}

        if type(fit_obj) is str:
            with open(fit_obj) as fit_file:
                fit_lines = fit_file.readlines()
        else:
            fit_lines = fit_obj.readlines()  # already a file object
        self.bands, self.sed, self.sed_err = self._parse_observed_sed(
            fit_lines)
        _, self.model_sed = self._parse_model_sed(fit_lines)
        self.i_sfh, self.chi2, self.z = OpticalFit._parse_best_fit(fit_lines)
        self._pdfs['Z_Zo'] = OpticalFit._parse_pdf(fit_lines, 16, 120)
        self._pdfs['tform'] = OpticalFit._parse_pdf(fit_lines, 122, 258)
        self._pdfs['gamma'] = OpticalFit._parse_pdf(fit_lines, 260, 361)
        self._pdfs['t_lastB'] = OpticalFit._parse_pdf(fit_lines, 363, 500)
        self._pdfs['log_age_M'] = OpticalFit._parse_pdf(fit_lines, 502, 628)
        self._pdfs['log_age_r'] = OpticalFit._parse_pdf(fit_lines, 630, 771)
        self._pdfs['log_Mstar'] = OpticalFit._parse_pdf(fit_lines, 773, 874)
        self._pdfs['SFR_0.1Gyr'] = OpticalFit._parse_pdf(fit_lines, 876, 937)
        self._pdfs['sSFR_0.1Gyr'] = OpticalFit._parse_pdf(fit_lines, 939, 1010)
        self._pdfs['mu'] = OpticalFit._parse_pdf(fit_lines, 1012, 1033)
        self._pdfs['tau_V'] = OpticalFit._parse_pdf(fit_lines, 1035, 1157)
        self._pdfs['tau_V_ISM'] = OpticalFit._parse_pdf(fit_lines, 1159, 1281)

    @staticmethod
    def _parse_pdf(lines, start_n, percentile_n):
        end_n = percentile_n - 1  # line where the PDF grid ends
        n_bins = end_n - start_n
        bins = np.empty(n_bins, dtype=np.float)
        probs = np.empty(n_bins, dtype=np.float)
        print(lines[start_n - 1])
        for i, line in enumerate(lines[start_n:end_n]):
            b, p = map(float, line.strip().split())
            bins[i] = b
            probs[i] = p
        percentiles = map(float, lines[percentile_n].strip().split())
        d = {"bins": bins,
             "probs": probs,
             "2.5": percentiles[0],
             "16": percentiles[1],
             "50": percentiles[2],
             "84": percentiles[3],
             "97.5": percentiles[4]}
        return d
