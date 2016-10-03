#!/usr/bin/env python
# encoding: utf-8
"""
Python readers/representation of MAGPHYS result files.

2014-12-17 - Created by Jonathan Sick
"""

import re
import numpy as np
from collections import OrderedDict

import astropy.units as u
import astropy.constants as const


class BaseReader(object):

    # map magphys names to internal column names
    magphys_params = OrderedDict([
        ('Z/Zo', 'Z/Zo'),
        ('tform', 'tform'),
        ('gamma', 'gamma'),
        ('t(lastB)', 't_lastB'),
        ('log age(M)', 'log_age_M'),
        ('f_mu (SFH)', 'f_mu_sfh'),
        ('f_mu (IR)', 'f_mu_ir'),
        ('mu parameter', 'mu'),
        ('tau_V', 'tau_V'),
        ('sSFR_0.1Gyr', 'sSFR_0.1Gyr'),
        ('M(stars)', 'log_Mstar'),
        ('Ldust', 'log_Ldust'),
        ('T_C^ISM', 'T_C_ISM'),
        ('T_W^BC', 'T_W_BC'),
        ('xi_C^tot', 'xi_C_tot'),
        ('xi_PAH^tot', 'xi_PAH_tot'),
        ('xi_MIR^tot', 'xi_MIR_tot'),
        ('xi_W^tot', 'xi_W_tot'),
        ('tau_V^ISM', 'tau_V_ISM'),
        ('M(dust)', 'log_Mdust'),
        ('SFR_0.1Gyr', 'SFR_0.1Gyr'),
    ])

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

    @staticmethod
    def _detect_pdf_lines(lines):
        pattern = "^# \.\.\.(.+)\.\.\.$"
        p = re.compile(pattern)

        pdf_lines = []
        for i, line in enumerate(lines):
            m = p.match(line)
            if m is not None and i > 10:
                pdf_lines.append((m.group(1).strip(), i))

        limits = {}
        for j, (key, start_i) in enumerate(pdf_lines):
            try:
                limits[key] = (start_i + 1, pdf_lines[j + 1][-1] - 1)
            except IndexError:
                limits[key] = (start_i + 1, i)
        return limits

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

        pdf_lines = self._detect_pdf_lines(fit_lines)

        for magphys_param, startend in pdf_lines.iteritems():
            param_name = self.magphys_params[magphys_param]
            start = startend[0]
            end = startend[1]
            self._pdfs[param_name] = self._parse_pdf(fit_lines, start, end)

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

        pdf_lines = self._detect_pdf_lines(fit_lines)

        for magphys_param, startend in pdf_lines.iteritems():
            param_name = self.magphys_params[magphys_param]
            start = startend[0]
            end = startend[1]
            self._pdfs[param_name] = self._parse_pdf(fit_lines, start, end)

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

        self.i_sfh, self.chi2, self.z = self._parse_best_fit(fit_lines)

        pdf_lines = self._detect_pdf_lines(fit_lines)

        for magphys_param, startend in pdf_lines.iteritems():
            param_name = self.magphys_params[magphys_param]
            start = startend[0]
            end = startend[1]
            self._pdfs[param_name] = self._parse_pdf(fit_lines, start, end)
