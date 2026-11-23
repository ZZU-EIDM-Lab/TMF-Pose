# TMF-Pose publication readiness checklist

## Confirmed by the project team

- [x] All TMF-Pose paper contributors agree to public code release (as reported by the project representative).
- [x] TMF-Pose was implemented by modifying and extending AG-Pose; upstream derivation is now explicitly acknowledged in the README.
- [x] The original AG-Pose MIT license and copyright statement are included unchanged in `LICENSES/AG-Pose-LICENSE.txt`.
- [x] The 47 retained experimental source and configuration files match the supplied original source archive byte-for-byte.

## Items to resolve before announcing complete licensing/reproducibility

- [ ] Check whether a university, lab, or collaborating organization has ownership or policy requirements beyond contributor consent.
- [ ] Confirm the exact upstream origins and applicable licenses for embedded PointNet++ and Gorilla Lab evaluation utilities; preserve any additional required license texts.
- [ ] Decide whether to license the project team's *new contributions* under MIT or another license, and then add an appropriate root `LICENSE` approved by the rightsholders. The upstream AG-Pose MIT grant remains in effect for AG-Pose portions regardless.
- [ ] Review machine-specific paths and identifying metadata in unmodified code/configuration files before switching the repository to Public.
- [ ] Confirm environment instructions and weights if claiming out-of-the-box reproducibility; existing experimental code has not been rerun in a clean environment.

This checklist is informational and not legal advice. No algorithm, train/test script, configuration, or PointNet++ source was modified to produce this package.
