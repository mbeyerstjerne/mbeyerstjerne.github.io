---
title: "Processing astronomical data: Imaging, spectroscopy, photometry" 
date: 2024-11-08
tags: ["Data processing", "Astrophysics"]
author: [""]
description: "This project involves the processing of imaging and spectroscopic data in order to make basic astrophysical measurements."
summary: "This project involves the processing of imaging and spectroscopic data in order to make basic astrophysical measurements."
cover:
    image: "optical-spectrum.png"
    alt: "Optical spectrum of SN2017eaw"
    relative: true
draft: false
weight: 100
# editPost:
#     URL: "https://github.com/pmichaillat/hugo-website"
#     Text: "Journal of Oleic Science"
showMeta: true
# layout: "single"
---

<head>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
  </script>
</head>

---

##### Project description

<div style="font-size:90%; line-height:150%">

  In this project, I developed and applied a complete astronomical data reduction and analysis pipeline for both imaging and long-slit spectroscopic observations, using real CCD data from ground-based telescopes. This covers the full path from raw FITS files to science-ready measurements.

</div>

---

##### Basic analysis of imaging data

<div style="font-size:85%; line-height:150%">

  For imaging data of the M67 open cluster obtained with the ESO 3.6 m telescope, I performed exploratory analysis of raw bias, dark, and flat frames, constructed master calibration frames, and produced fully reduced science images. I quantified detector characteristics such as gain and readout noise directly from the data and propagated these through the reduction steps to generate variance images that track per-pixel uncertainties. This enabled statistically consistent downstream analysis.

</div>


---

##### Aperture photometry

<div style="font-size:85%; line-height:150%">

  Using the reduced images, I implemented two-dimensional aperture photometry to measure stellar fluxes across multiple filters. I estimated point-spread function widths, detected sources using DAOStarFinder, and performed background-subtracted aperture photometry with carefully chosen annuli. I calibrated instrumental magnitudes using standard stars, including colour-term and airmass corrections, and propagated uncertainties to final magnitudes. Cross-matching between filters enabled construction of colour–magnitude diagrams, which were validated against literature results and shown to reproduce the expected stellar sequences.

</div>

---

##### Spectroscopy

<div style="font-size:85%">

  For spectroscopy, I reduced long-slit observations of Supernova SN2017eaw and a spectrophotometric standard star taken with the ALFOSC instrument on the Nordic Optical Telescope. This included bias subtraction, flat-fielding, sky background subtraction, wavelength calibration, and extraction of one-dimensional spectra. I carried out flux calibration using the standard star and verified the consistency of the calibrated spectra. Finally, I computed hour angles and parallactic angles from FITS header metadata and confirmed agreement with instrument-derived values, validating the geometric and observational consistency of the data.

</div>

---

##### Conclusion

<div style="font-size:85%; line-height:150%">

  Overall, this project demonstrates my ability to design reproducible astronomical data pipelines, work directly with CCD-level data, apply statistical error propagation, and extract physically meaningful photometric and spectroscopic measurements using Python-based scientific workflows.

</div>

---

##### Related material

+ [Original report (.pdf)](adp-report.pdf)
