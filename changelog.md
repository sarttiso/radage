# Changelog

## 0.2.0

### Added
- `radage.DetritalSpectra` class allows for comparison of detrital zircon spectra

### Changed
- `radage.UPb` can be intialized with different combinations of 206/238, 207/235, and 207/206 ratios

## 0.1.0

This version marks the first (beta) release of `radage`. The project is very much still a work in progress, and guides for basic usage still need to be prepared. Most classes and functions are documented.

Basic functionality at this point includes:
- UPb dates in the `radage.UPb` object (constructed with isotopic data)
    - various date computation methods with uncertainty
    - date ellipses
- concordia helper functions
- weighted mean computation
- yorkfit with MSWD evaluation

Various plotting functions are also implemented:
- concordia plotting with date ellipses
- age-rank plots
- kde plotting
