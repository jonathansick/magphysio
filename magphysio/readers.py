#!/usr/bin/env python
# encoding: utf-8
"""
Python readers/representation of MAGPHYS result files.

2014-12-17 - Created by Jonathan Sick
"""

import numpy as np


class BaseReader(object):
    def __init__(self):
        super(BaseReader, self).__init__()

    @staticmethod
    def _parse_observed_sed(lines, index=1):
        bands = lines[index].replace('#', '').strip().split()
        sed = np.array(map(float, lines[index + 1].strip().split()))
        err = np.array(map(float, lines[index + 2].strip().split()))
        return bands, sed, err

    @staticmethod
    def _parse_model_sed(lines, index=11):
        bands = lines[index].replace('#', '').strip().split()
        sed = np.array(map(float, lines[index + 1].strip().split()))
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

    def persist(self, f):
        """Persist the MAGPHYS fit to a hierarchical HDF5 file.

        Dataset is stored in the group `/models/{galaxy_id}`.
        """
        group = f["models/{0}".format(self.galaxy_id)]

        # Persist SED
        sed = np.vstack([self.sed.T, self.sed_err.T, self.model_sed.T])
        group['sed'] = sed
        group['sed'].attrs['bands'] = self.bands
        group['sed'].attrs['i_sh'] = self.i_sfh
        group['sed'].attrs['chi2'] = self.chi2

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


class OpticalFit(BaseReader):
    """A MAGPHYS model fit for Roediger's fit_magphys_opt.exe mod."""
    def __init__(self, galaxy_id, fit_path):
        super(OpticalFit, self).__init__()
        self.galaxy_id = galaxy_id
        self._pdfs = {}

        with open(fit_path) as fit_file:
            fit_lines = fit_file.readlines()
        self.bands, self.sed, self.sed_err = OpticalFit._parse_observed_sed(
            fit_lines)
        _, self.model_sed = OpticalFit._parse_model_sed(fit_lines)
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
