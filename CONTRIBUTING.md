# Contributing to cuPyNumeric

CuPyNumeric is an open-source project released under the [Apache license, version 2.0](https://www.apache.org/licenses/LICENSE-2.0).  We welcome any and all contributions, and we hope that you can help us develop a strong community.

## How to begin

Most of the time, the best thing is to begin by [opening an issue](https://github.com/nv-legate/cupynumeric/issues).  This gives us a chance to discuss the contribution and to define the problem or feature that it addresses.   Often, opening of the issue first may help prevent you from doing unnecessary work or to enhance and further develop your idea.

Once you are ready to start development, we ask you to work on a [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) of our repository.  The next step is to create a (pull request)[https://help.github.com/en/articles/about-pull-requests].  Feel free to open the pull request as soon as you begin your development (just mark it [as a draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/)) or when you are ready to have your contribution merged.

## The Legalese: Developer Certificate of Origin

CuPyNumeric is released under the open-source [Apache license, version 2.0](https://www.apache.org/licenses/LICENSE-2.0), and is free to use, modify, and redistribute.  To ensure that the license can be exercised without encumbrance, we ask you that you only contribute your own work or work to which you have the intellectual rights.  To that end, we employ the Developer's Certificate of Origin (DCO), which is the lightweight mechanism for you to certify that you are legally able to make your contribution. Here is the full text of the certificate (also available at [DeveloperCertificate.org](https://developercertificate.org/):

````
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
````

### How Do I Sign the DCO?

Fortunately, it does not take much work to sign the DCO.  The only thing that you have to do is to mark all your commits with a `Signed-off-by` line that looks like that:

````
Signed-off-by: Your Name <your@email.address>
````

Please use your real name and a valid email address at which you can be reached.  For legal reasons, we will not be able to accept contributions that use pseudonyms in the signature.  You can simply add this line at the end of all your commits manually, or you can use the `-s` or the `--signoff` options provided by Git to automatically tack on the signature.

## Review Process

We are really grateful that you are thinking of contributing to cuPyNumeric.  We will make every effort to review your contributions as soon as possible.

As we suggested at the beginning of this document, it will be really helpful to start with an issue unless your proposed change is really trivial.  An issue will help to save work in the review process (e.g., maybe somebody is already working on exactly the same thing you want to work on).  After you open your pull request (PR), there usually will be a community feedback that often will require further changes to your contribution (the usual open-source process).  Usually, this will conclude in the PR being merged by a maintainer, but on rare occasions a PR may be rejected.  This may happen, for example, if the PR appears abandoned (no response to the community feedback) or if the PR does not seem to be approaching community acceptance in a reasonable time frame.  In any case, an explanation will always be given why a PR is closed.  Even if a PR is closed for some reason, it may always be reopened if the situation evolves (feel free to comment on closed PRs to discuss reopening them).

## Code Formatting Requirements

CuPyNumeric has a set of coding standards that are expected from all the code merged into the project.  The coding standards are defined by the set of tools we use to format our code.  We use the [pre-commit](https://pre-commit.com/) framework to run our formatting tools.  The easiest way to meet the coding standards is to simply use the pre-commit framework to run all the checks for you.  Please visit the [pre-commit project page](https://pre-commit.com/) for pre-commit installation and usage instructions.  Once pre-commit is installed in the cuPyNumeric repo, all the checks and formatting will be run on every commit, but one can also run the checks explicitly as detailed in pre-commit documentation.

We hope that the automation of our formatting checks will make it easy to comply with our coding standards.  If you encounter problems with code formatting, however, please let us know in a comment on your PR, and we will do our best to help.
