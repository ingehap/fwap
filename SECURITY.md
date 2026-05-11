# Security policy

## Supported versions

`fwap` is a scientific-computing library with no network, filesystem,
or authentication surface beyond what NumPy, SciPy, Matplotlib,
`lasio`, `dlisio`, `dliswriter`, and `segyio` already expose. Only
the most recently released minor version receives security fixes.

| Version  | Supported |
|----------|-----------|
| 0.4.x    | Yes       |
| < 0.4    | No        |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-relevant
reports. Instead:

1. Open a GitHub [security advisory](../../security/advisories/new)
   on this repository. This lets the maintainers discuss the issue
   privately and coordinate a release.
2. Include a minimal reproducer (Python script + input data) if at
   all possible; non-reproducible reports are difficult to act on.
3. Please give maintainers a reasonable window (typically 30 days)
   to issue a fix before any public disclosure.

## What counts as a vulnerability here?

The project's threat model is that of a typical scientific-Python
library: trusted inputs from the user's workstation. Specifically:

* **In scope**: an input file (LAS, DLIS, SEG-Y, or a crafted
  synthetic) causing uncaught infinite loops, unbounded memory
  growth, Python interpreter crashes, or reading/writing arbitrary
  filesystem paths outside the user's intent.
* **Out of scope**: numerical inaccuracy on pathological inputs
  (that's a bug, open a normal issue), slow performance on large
  problems (likewise), or using the library on untrusted data for
  use cases the README doesn't describe.

For non-security bugs please open a regular GitHub issue.
