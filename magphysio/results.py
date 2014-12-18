#!/usr/bin/env python
# encoding: utf-8
"""
Python representation of MAGPHYS result files.

2014-12-17 - Created by Jonathan Sick
"""

import numpy as np


class OpticalFit(object):
    """A MAGPHYS model fit for Roediger's fit_magphys_opt.exe mod."""
    def __init__(self, galaxy_id, fit_path):
        super(OpticalFit, self).__init__()
        self.galaxy_id = galaxy_id
        self._pdfs = {}

        with open(fit_path) as fit_file:
            fit_lines = fit_file.readlines()
        self._pdfs['Z/Zo'] = OpticalFit._parse_pdf(fit_lines, 16, 120)
        self._pdfs['tform'] = OpticalFit._parse_pdf(fit_lines, 122, 258)
        self._pdfs['gamma'] = OpticalFit._parse_pdf(fit_lines, 260, 361)
        self._pdfs['t(lastB)'] = OpticalFit._parse_pdf(fit_lines, 363, 500)
        self._pdfs['log age(M)'] = OpticalFit._parse_pdf(fit_lines, 502, 628)
        self._pdfs['log age(r)'] = OpticalFit._parse_pdf(fit_lines, 630, 771)
        self._pdfs['log M*'] = OpticalFit._parse_pdf(fit_lines, 773, 874)
        self._pdfs['SFR_0.1Gyr'] = OpticalFit._parse_pdf(fit_lines, 876, 937)
        self._pdfs['sSFR_0.1Gyr'] = OpticalFit._parse_pdf(fit_lines, 939, 1010)
        self._pdfs['mu'] = OpticalFit._parse_pdf(fit_lines, 1012, 1033)
        self._pdfs['tauV'] = OpticalFit._parse_pdf(fit_lines, 1035, 1157)
        self._pdfs['tauV^ISM'] = OpticalFit._parse_pdf(fit_lines, 1159, 1281)

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
