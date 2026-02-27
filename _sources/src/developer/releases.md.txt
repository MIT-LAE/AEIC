# Releases

Because we have branch protection on the main branch of our GitHub repository,
making releases is a little more complicated than the simplest possible case.
It basically goes in three steps:

1. Make a release branch with a new version number.
2. Make a PR from the release branch and merge it.
3. Tag the new version (pushing the tag to GitHub triggers the process that
   builds an installable release).

There is a `release.sh` script in the `scripts` directory to help with this.
Running this script performs step 1 above and gives instructions for steps 2
and 3. The script must be given a single argument, one of `major`, `minor` or
`patch` for the version number component to be incremented.

There is a GitHub Actions workflow to perform the actual release process,
triggered by pushing a `vX.Y.Z` tag to the repository. The release generation
uses GitHub's automatic release notes generation mechanism, which just lists
the PRs that have been merged since the last release.

Once created, releases can be installed using pip or `uv` as
`git+https://github.com/MIT-LAE/AEIC.git@vX.Y.Z`.
