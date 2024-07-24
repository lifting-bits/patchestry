# Patchestry WWW

This contains the source code for https://lifting-bits.github.io/patchestry/ ;
which is rendered from the `gh-pages` branch of the same repo using GitHub
pages.

To contribute, feel free to fork this repository and send a pull-request.

The website is deployed on every release of this repository using a GitHub action
defined in `.github/workflows/prerelease.yml`. It also runs every release of Patchestry,
which is used for generating some docs.

We are using the [MkDocs](https://www.mkdocs.org/) framework for generating the
website.

The expected workflow is to download mkdocs on your machine, and run `mkdocs serve`
from the root directory. You can then access a local version of the website
from http://127.0.0.1:8000/ ; any change you make to the source Markdown will
automatically be refreshed by the local mkdocs server.

A large part of the documentation is auto-generated from the Patchestry source code.
See the workflow `.github/workflows/prerelease.yml` for the instructions to reproduce
the entirety of https://lifting-bits.github.io/patchestry/ locally.
