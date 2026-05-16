# Review: Bourbié, Coussy & Zinszner (1987), Acoustics of Porous Media

## Context and contribution

This is the English-language edition of *Acoustique des milieux poreux* (Éditions Technip, 1986), translated and lightly revised in 1987 and published as part of the IFP / Éditions Technip *Petroleum Engineering* series. The volume runs 334 pages and was written by three authors at the Institut Français du Pétrole — Thierry Bourbié (seismic acoustics), Olivier Coussy (continuum mechanics of porous media) and Bernard Zinszner (laboratory rock physics). Each author brings a distinct strand: Coussy contributes the rigorous thermodynamic/continuum derivation of Biot's equations that he would later expand into his *Mechanics of Porous Continua* (Wiley, 1995); Zinszner contributes the laboratory programme on Fontainebleau and Vosges sandstones that supplies most of the experimental material; and Bourbié ties the theory to seismic-frequency observation.

For roughly a decade after its appearance, this book was *the* standard textbook bridge between Biot's primary papers (1956a,b; 1962) and the working petroleum geophysicist. Where Biot's papers are short, dense and notationally idiosyncratic, Bourbié–Coussy–Zinszner sets out the same physics with consistent notation, worked-out limiting cases, full derivations of the Biot-modulus / Gassmann coefficients, and — uniquely for the time — a sustained side-by-side comparison between theoretical predictions and laboratory ultrasonic measurements on real reservoir rocks. The book has since been partially superseded for advanced treatments (Carcione 2022 for anisotropic and dissipative extensions; Mavko–Mukerji–Dvorkin 2020 for the rock-physics side; Coussy 2004 for the mechanics), but it remains the *first* book to read on isotropic poroelasticity and is still routinely cited as the reference derivation of the Biot–Gassmann relations.

## Main ideas and structural progression

**Petrophysics of porous media (Ch. 1).** The opening chapter is a self-contained primer on the geometric and transport properties of sedimentary rocks: porosity (total vs. connected, intergranular vs. vugular vs. fracture), specific surface, tortuosity, the Kozeny–Carman relation, Klinkenberg slip, mercury-injection capillary pressure curves, and the experimental protocols by which each of these is measured. Zinszner's IFP laboratory data on Fontainebleau sandstone (a clean quartz arenite whose porosity can be tuned from a few percent to ~30 % by selecting different outcrops) recur throughout the book as the canonical "clean" rock against which theory is tested. The chapter sets the conceptual point that the parameters which enter Biot's equations (porosity φ, permeability κ, tortuosity α∞, pore-fluid viscosity η) are not arbitrary fitting constants but quantities measured by independent petrophysical experiments.

**Mechanics of the porous continuum (Ch. 2).** Coussy develops the constitutive theory of a fluid-saturated porous solid in the small-strain, isothermal regime. The two-phase mixture is treated with an Eulerian skeleton variable and a Lagrangian fluid-mass content ζ; the strain-energy potential is constructed and differentiated to give the *constitutive equations of poroelasticity* in the form

  σᵢⱼ = 2μ εᵢⱼ + (λ ε − α p) δᵢⱼ
  p   = M (ζ − α ε)

with (λ, μ) the drained Lamé parameters, α the Biot effective-stress coefficient and M the Biot modulus. The chapter derives Gassmann's fluid-substitution equation as the *low-frequency* limit, in which the pore pressure is equilibrated across a wavelength so that the effective bulk modulus depends only on the dry-frame, mineral and fluid moduli. This derivation — methodical, with all assumptions stated — is the one most users of Gassmann's equation in seismic interpretation cite.

**Dynamic poroelasticity and the Biot equations (Ch. 3).** The static theory of Chapter 2 is extended to dynamics by adding inertial coupling between the solid and the fluid and a Darcy-type viscous drag. This produces Biot's two coupled momentum equations with the additional inertia parameter ρ₁₂ (negative, a measure of how strongly the fluid is dragged by the solid through tortuous channels) and the frequency-dependent viscous correction F(ω) that interpolates between low-frequency Poiseuille flow and high-frequency boundary-layer flow with the characteristic Biot frequency ωc = ηφ/(κρf α∞). Plane-wave solution of the coupled system gives the three Biot modes — the fast P-wave (P1, in-phase solid + fluid motion), the slow P-wave (P2, out-of-phase, diffusive at low frequency, propagating above ωc), and the shear wave — and the chapter computes their phase velocities and quality factors as functions of frequency for representative parameter sets.

**Velocity dispersion and attenuation (Ch. 4).** The chapter turns the dispersion relations of Ch. 3 into a quantitative attenuation theory and compares the predictions to laboratory data. The headline conclusions are familiar but were not yet in textbook form in 1987: (i) intrinsic Biot loss alone is too small, by one or two orders of magnitude, to explain the seismic-frequency Q observed in real saturated rocks; (ii) the slow P-wave is in practice diffusive at seismic frequencies, never observed in the field, and only marginally observable in carefully designed laboratory experiments — confirming Plona's 1980 ultrasonic detection in fused-glass-bead samples; (iii) the discrepancy between Biot Q and observed Q must be made up by other mechanisms (squirt flow, mesoscopic patchy saturation, scattering), the systematic treatment of which postdates this book and is now the subject of e.g. Carcione 2022 Ch. 7.

**Fluid effects on velocities and the Gassmann/Biot synthesis (Ch. 5).** A whole chapter is devoted to the Gassmann fluid-substitution programme: how to measure the dry-frame moduli (Kdry, μdry), how to predict the saturated moduli for an arbitrary fluid (brine, oil, gas, mixtures), and how patchy saturation and pressure dependence complicate the simple substitution. The chapter treats the limits — Voigt (uniform pressure, fast P maximum) vs. Reuss (uniform stress, fast P minimum) — and uses White's (1975) patchy-saturation model to interpolate, which sets up the modern WIFF literature. The chapter is also notable for emphasising that the shear modulus is *not* changed by fluid substitution under Gassmann assumptions, an identity that has since become a standard rock-physics consistency check.

**Experimental rock physics (Ch. 6).** The book's empirical centre. Zinszner's IFP programme of ultrasonic-frequency P- and S-velocity measurements on Fontainebleau sandstones, Vosges sandstones, Estaillades and Lavoux limestones is presented systematically: as a function of porosity, of confining and pore pressure, of saturating fluid (dry, brine, oil), of saturation history (drained vs. drainage hysteresis), and of frequency (sonic-band resonant bar vs. ultrasonic pulse transmission). The treatment of the *frequency dispersion* — the systematic difference between low-frequency (Gassmann) and high-frequency (squirt-dominated) saturated velocities — was at the time one of the cleanest in the literature and remains a useful experimental benchmark.

**Application to seismic exploration (Ch. 7).** A final, more discursive chapter sketches the seismic-prospecting consequences of the preceding theory: the use of AVO and amplitude attributes for fluid discrimination, the bright-spot phenomenology in gas-charged sandstones, the use of Vp/Vs for lithology and gas indication, and the limits of resolution imposed by attenuation. Compared with later textbook treatments (e.g. Avseth, Mukerji & Mavko 2005), the seismic-application chapter is brief and dated; its value today is mainly historical, as the document of how the petroleum industry first integrated Biot/Gassmann theory into seismic interpretation in the 1980s.

## Significance and limitations

The book's enduring value is twofold. As a *derivation*, it is still the most accessible self-contained route from continuum thermodynamics to Biot's coupled wave equations and to Gassmann's substitution rule; for many practising geophysicists, the Bourbié–Coussy–Zinszner derivation of M, α and the Gassmann formula is the one they learned from. As an *experimental record*, it documents the IFP rock-physics programme on Fontainebleau and Vosges sandstones that has since become the community's standard test data set for any new poroelastic theory.

Three limitations should be flagged for the contemporary reader. First, the treatment is strictly *isotropic*: anisotropic poroelasticity (Biot 1962 in tensor form, with permeability and α as tensors) is mentioned only in passing. For anisotropic shales, fractured carbonates and laminated sequences the book must be supplemented by Carcione 2022 (Ch. 7) or by the dedicated anisotropic-poroelasticity literature. Second, the attenuation discussion ends just before the modern *mesoscopic-loss / wave-induced fluid flow* (WIFF) framework of White, Pride–Berryman–Harris and Müller–Gurevich became established; the book correctly identifies the inadequacy of pure Biot loss but does not yet offer a quantitative alternative. Third, the rock-physics programme is essentially clean-sandstone and clean-carbonate; clay-bound, kerogen-rich, unconventional and mixed-mineralogy rocks are outside its scope, and for these the modern reference is Mavko, Mukerji & Dvorkin (2020).

For our project the book is the natural first reference whenever the forward problem requires *Gassmann fluid substitution* or *isotropic Biot-frequency analysis*: Chapters 2–3 give the cleanest derivation, Chapter 5 the practical substitution recipe, and Chapter 6 a calibrated experimental data set against which simulator predictions can be sanity-checked. For anisotropic or dissipative extensions it should be paired with Carcione 2022; for the rock-physics calibration of the input parameters it should be paired with Mavko–Mukerji–Dvorkin.

## References

Bourbié, T., Coussy, O., & Zinszner, B. (1987). *Acoustics of Porous Media.* Houston / Paris: Gulf Publishing / Éditions Technip. ISBN 2-7108-0516-2. (English edition of *Acoustique des milieux poreux*, Éditions Technip, 1986.)

Biot, M. A. (1956a). Theory of propagation of elastic waves in a fluid-saturated porous solid. I. Low-frequency range. *Journal of the Acoustical Society of America*, 28, 168–178.

Biot, M. A. (1956b). Theory of propagation of elastic waves in a fluid-saturated porous solid. II. Higher-frequency range. *Journal of the Acoustical Society of America*, 28, 179–191.

Biot, M. A. (1962). Mechanics of deformation and acoustic propagation in porous media. *Journal of Applied Physics*, 33, 1482–1498. (The tensor form on which the book's Ch. 3 dynamics is built.)

Gassmann, F. (1951). Über die Elastizität poröser Medien. *Vierteljahrsschrift der Naturforschenden Gesellschaft in Zürich*, 96, 1–23. (Original derivation of the fluid-substitution equation rederived in Ch. 2.)

Plona, T. J. (1980). Observation of a second bulk compressional wave in a porous medium at ultrasonic frequencies. *Applied Physics Letters*, 36, 259–261. (First experimental confirmation of Biot's slow P-wave; discussed in Ch. 4.)

White, J. E. (1975). Computed seismic speeds and attenuation in rocks with partial gas saturation. *Geophysics*, 40, 224–232. (Patchy-saturation model used in Ch. 5.)

Coussy, O. (1995). *Mechanics of Porous Continua.* Chichester: Wiley. (Coussy's later, fuller treatment of the continuum-mechanics derivation sketched in Ch. 2.)

Coussy, O. (2004). *Poromechanics.* Chichester: Wiley. (Second, definitive monograph on the same material.)

Carcione, J. M. (2022). *Wave Fields in Real Media* (4th ed.). Amsterdam: Elsevier. (Modern reference for anisotropic and dissipative extensions of Biot theory; the natural sequel to the present book.)

Mavko, G., Mukerji, T., & Dvorkin, J. (2020). *The Rock Physics Handbook* (3rd ed.). Cambridge University Press. (Companion volume on the rock-physics calibration of Gassmann/Biot input parameters.)

Avseth, P., Mukerji, T., & Mavko, G. (2005). *Quantitative Seismic Interpretation.* Cambridge University Press. (Modern textbook successor to the seismic-application discussion in Ch. 7.)
