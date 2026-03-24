<div class="no-extra-css"></div>
# Contributing

PyHOPE is an open-source project and we are very happy to accept contributions
from the community. Please feel free to open issues or submit patches (preferably
as merge requests) any time. For planned larger contributions, it is often
beneficial to get in contact with one of the principal developers first (see
[AUTHORS.md](AUTHORS.md)).

## How To Collaborate
In order to get your contributions into the code base as smoothly as possible, please follow these contribution guidelines.

- Ensure you have a GitHub account
- Check existing issues to ensure one does not already exist
  - [Open an issue][newissue], clearly describing the issue including steps to reproduce
- Fork the repository on GitHub
- Create a feature branch, based on `main`
  - For simplicity, prefix the branch name either with `bugfix.` or `feature.`
- Make commits of logical units.
  Write [good commit messages][commit].
  Check for unnecessary whitespace with `git diff --check` before committing.
- Include tests in `tutorials` to assure your feature works as expected and prevent breakages in the future
- Run `pyhope --verify` to ensure all existing code works as expected
- Use the available [tools to format and check your contribution][devtools]
- Submit a pull request to the PyHOPE repository

## Taking code from other projects
We believe in the power of Open Source or Free Software to share and reuse code from other projects. However, Free Software is not public domain, and not every code could be reused in every other project.

Please contact the maintainers before integrating non-trivial amount of code from other projects, so we can ensure the compatibility of licences. Same holds true for additional dependencies, libraries etc.

PyHOPE and its contributions are licensed under the GPL-3.0 license (see
[LICENSE.md](LICENSE.md)). As a contributor, you certify that all your
contributions are in conformance with the *Developer Certificate of Origin
(Version 1.1)*, which is reproduced below.

## Developer Certificate of Origin (Version 1.1)
The following text was taken from
[https://developercertificate.org](https://developercertificate.org):

    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.

[newissue]: https://github.com/hopr-framework/PyHOPE/issues/new
[commit]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[devtools]: https://github.com/hopr-framework/PyHOPE/blob/main/.gitlab-ci.yml#L118
