# Reference Order 7 results

These numbers describe the local analysis before the standalone source export;
they are not results of a fresh fit performed during repository publication.

The reference used all 1,848 stored wavelengths and four comparison frames
(10, 30, 50, 70). RMS is in percent of the same fitted continuum. All absorption
depths remain included. The secondary score subset consists of 773 wavelengths
in fixed alternating blocks; gray-only points in the report are not exclusions.

| Stage | All-pixel RMS | Score-subset RMS |
| --- | ---: | ---: |
| Physical PHOENIX, before smooth correction | 2.420400% | 1.948194% |
| PHOENIX plus degree-five smooth correction | 1.515948% | 1.442966% |
| Stellar-amplitude calibration plus smooth correction | 1.224940% | 1.273490% |
| Stellar calibration and telluric-strength correction | 0.820934% | 0.868707% |

The final reference had 160 free stellar amplitudes and 4,119 free telluric
intensity corrections: 12 H2O, 3,351 CH4, and 756 N2O. Other lines remain in
the physical opacity even when their strengths are fixed. The stellar
calibration uses wavelengths whose initial telluric transmission exceeds 0.95;
this selection does not exclude the other pixels from the telluric fit or RMS.

The global metadata S/N proxy is about 0.538%. It is not an order-specific or
deep-line photon-noise limit: per-pixel photon variances are unavailable.

Important limits of interpretation:

- Shared corrections use five frames. Nuisance parameters and polynomial
  corrections also use data from the comparison frames.
- The reference solvers reported success, but the stricter requirement of two
  consecutive prediction changes below 1e-8 was not met. The final maximum
  change was about 4.8e-8 in continuum units.
- Some stellar and telluric amplitudes hit bounds. A large empirical line
  correction can compensate for stellar, atmospheric, or instrumental error;
  it is not a measurement of a HITRAN error.
- The 4383 cm-1 feature contains overlapping stellar and CH4 absorption.
  Local profile-flexibility tests support a mixed stellar-shape/telluric-strength
  explanation, without uniquely identifying the physical cause.
- Planet-signal recovery has not been tested for this Order 7 configuration.
  Earlier injection tests on other orders do not establish it here.

The original fitted array SHA256 was
`99092f5639d28946a08f388861af07c979e5f60231a95cce116c6efba40c0bf2`.
That array, the observational inputs, and all enuma comparison products are
excluded from the repository. Exact reproduction also depends on the downloaded
HITRAN data and the numerical environment.
