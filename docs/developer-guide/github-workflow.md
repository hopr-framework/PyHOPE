# GitHub workflow

Overview of the development workflow

---

Code development is performed on [GitHub](https://github.com), with a protected `main` branch. The actual development is performed on feature branches, which are merged to `main` following a pull request and the completion of a code review. Continuous Integration (CI) tests are automatically performed on pull requests with a successful pass as a prerequisite for merging into `main`.

## Issues & Milestones
Issues are created for bugs, improvements, features, regression testing and documentation. The issues should be named with a few keywords. Try to avoid describing the complete issue already in the title. The issue can be assigned to a certain milestone (if appropriate).

Milestones are created based on planned releases (e.g. Release 1.2.1) or as a grouping of multiple related issues (e.g. Documentation Version 1, Clean-up Emission Routines). A deadline can be given if applicable. The milestone should contain a short summary of the work performed (bullet-points) as its contents will be added to the description of the releases. Generally, pull requests should be associated with a milestone containing a release tag, while issues should be associated with the grouping milestones.

As soon as a developer wants to start working on an issue, she/he shall assign himself to the issue and a branch and pull request denoted as work in progress ("draft") should be created to allow others to contribute and track the progress. Ideally, issues should be created for every code development for documentation purposes. Branches without an issue should be avoided to reduce the number of orphaned/stale branches. Where a branch must be created outside the issue context, its name should carry a prefix that matches one of the existing GitHub labels

```
bugfix.connect.internal
feature.sort.hilbertmorton
improvement.extrude.zones
```

Pull requests should include a concise description of what changed and why, a reference to the issue being resolved (e.g. `Closes #42`), and a note on anything that reviewers should pay particular attention to.

## Code Review
Every pull request requires at least one approving review before it can be merged. Reviewers should check at least the following points.

- The implementation is correct and consistent with the existing architecture
- New code follows the conventions in the [Style Guide](style-guide.md)
- Public functions carry type annotations and docstrings
- New functionality is covered by tests or tutorials
- The CI pipeline passes all checks without errors or warnings

Authors should respond to review comments promptly. If a comment is addressed by a code change, mark it as resolved. If it is addressed by explanation, reply in the thread and leave it for the reviewer to close.

## Commit Messages
Commit messages should be written in the imperative mood and describe *what* the commit does, not *what was done*:

```
# Good
Add convergence test for CGNS mixed mesh

# Avoid
Added convergence test
Fixed stuff
```


## Release and Deploy

Releases follow [Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`. A new release is created from `main` by opening a pull request associated with the corresponding release milestone. Once merged, navigate to [Releases](https://github.com/hopr-framework/PyHOPE/releases) and select `Draft a new release`. Set the tag to `vMAJOR.MINOR.PATCH` and the title to `Release MAJOR.MINOR.PATCH`. The milestone description supplies the release notes.

Tagged releases are automatically packaged as a Python wheel and deployed to [PyPI](https://pypi.org/project/PyHOPE) by the CI pipeline.
