# CHANGELOG


## v1.7.5 (2026-07-18)

### Bug Fixes

- Allow datasets containing 0.0 values in CLI
  ([`8ed2271`](https://github.com/omadson/fuzzy-c-means/commit/8ed22711b10cab1ba109aa0382995c652c701ae8))

_read_data rejected any CSV with a legitimate 0.0 value because `np.all(X)` treats 0 as falsy. The
  check also never caught its intended case: a wrong delimiter makes genfromtxt produce NaN, which
  np.all() treats as truthy, so the existing NaN check below already covers both wrong-delimiter and
  NaN-data cases on its own.


## v1.7.4 (2026-07-18)

### Bug Fixes

- Relax tabulate version constraint to allow 0.9.x
  ([`5e0f277`](https://github.com/omadson/fuzzy-c-means/commit/5e0f27709417f3c29d38df2bfd19756409cb6074))

Allows tabulate>=0.9.0 alongside pandas[output-formatting]/[all] (>=2.2), which requires
  tabulate>=0.9.0 and previously conflicted with our <0.9 pin.


## v1.7.3 (2026-07-17)

### Bug Fixes

- Rename commitlint config to .cjs to fix ESM error
  ([`8a4b169`](https://github.com/omadson/fuzzy-c-means/commit/8a4b1692d9e3b4d80673526caf20a9350766150e))

The action's runtime has "type": "module" in its package.json, so a .js file using module.exports is
  loaded as ESM and throws "module is not defined in ES module scope". Renaming to .cjs forces
  CommonJS regardless of that setting.

- Repair broken release and publish pipeline
  ([`584087f`](https://github.com/omadson/fuzzy-c-means/commit/584087fd7c563c435aad56aafb1a2f4609d8c222))

- semantic_release.version_toml pointed at tool.poetry.version, a key that no longer exists after
  the uv migration; version now lives at project.version - build_command still called `poetry build`
  - publish.yml never ran: it triggers on tag push, but release.yml creates tags using GITHUB_TOKEN,
  and GitHub does not fire workflows from events created by the default token -
  .pre-commit-config.yaml coverage hook still called `poetry run pytest`

Moves build+publish into release.yml itself, gated on the release step's `released` output, so
  there's no cross-workflow tag trigger to rely on. Removes the now-unreachable publish.yml.

- Use admin PAT for semantic-release push to bypass PR rule
  ([`bd45b22`](https://github.com/omadson/fuzzy-c-means/commit/bd45b2243d030aae7c10c0a8d335fb28a33506d4))

The "Require a pull request before merging" ruleset on master blocks any direct push, including the
  version bump commit and tag that python-semantic-release pushes with GITHUB_TOKEN. Switching to a
  PAT from an account added to the ruleset's bypass list (Repository admin role) lets that push go
  through without weakening the rule for regular contributors.

### Build System

- Change package and dependencies manager from poetry to uv
  ([`4132664`](https://github.com/omadson/fuzzy-c-means/commit/4132664ad198743fcc0a842cf62ecce876c964c9))

- Migrate to uv.
  ([`d485079`](https://github.com/omadson/fuzzy-c-means/commit/d4850795f8f320626795074bb1f409a91c1e0be6))

### Continuous Integration

- Enforce conventional commits on pull requests
  ([`0e8b177`](https://github.com/omadson/fuzzy-c-means/commit/0e8b177d939625cb202bd4283b06720e1c31ae85))

Adds a commitlint job so non-conforming commit messages are caught before merge instead of being
  silently ignored by semantic-release after the fact.


## v1.7.2 (2024-03-18)

### Bug Fixes

- Fix release and publish scripts.
  ([`7718856`](https://github.com/omadson/fuzzy-c-means/commit/7718856ed5823b68129e28839b174b60d75239a3))


## v1.7.1 (2024-03-18)

### Bug Fixes

- Correction of the cosine distance calculation method (#78).
  ([`c0fd2f8`](https://github.com/omadson/fuzzy-c-means/commit/c0fd2f8e33ad3328484701b7a4b747e89990261c))

### Build System

- Change publish pipeline.
  ([`2ec4255`](https://github.com/omadson/fuzzy-c-means/commit/2ec42554b1b444d96055b730743488a20c84c774))

- **deps**: Bump markdown-it-py from 2.1.0 to 2.2.0
  ([`a5af1d9`](https://github.com/omadson/fuzzy-c-means/commit/a5af1d96738c9dc893b4ee4c37b7cf371003bf60))

Bumps [markdown-it-py](https://github.com/executablebooks/markdown-it-py) from 2.1.0 to 2.2.0. -
  [Release notes](https://github.com/executablebooks/markdown-it-py/releases) -
  [Changelog](https://github.com/executablebooks/markdown-it-py/blob/master/CHANGELOG.md) -
  [Commits](https://github.com/executablebooks/markdown-it-py/compare/v2.1.0...v2.2.0)

--- updated-dependencies: - dependency-name: markdown-it-py dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps-dev**: Bump jinja2 from 3.1.2 to 3.1.3
  ([`bff083f`](https://github.com/omadson/fuzzy-c-means/commit/bff083fb146f64e516a05d1638bf407a52c80e03))

Bumps [jinja2](https://github.com/pallets/jinja) from 3.1.2 to 3.1.3. - [Release
  notes](https://github.com/pallets/jinja/releases) -
  [Changelog](https://github.com/pallets/jinja/blob/main/CHANGES.rst) -
  [Commits](https://github.com/pallets/jinja/compare/3.1.2...3.1.3)

--- updated-dependencies: - dependency-name: jinja2 dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

### Chores

- Add new env var.
  ([`8a2cd12`](https://github.com/omadson/fuzzy-c-means/commit/8a2cd12dfa66dd28832e4c78a618891e76158061))

- Change semantic release options.
  ([`ee5ffef`](https://github.com/omadson/fuzzy-c-means/commit/ee5ffef98e8c6ebd3f5bafb54cdb4ef6789a05eb))

- Remove notebook outputs
  ([`b9bd71c`](https://github.com/omadson/fuzzy-c-means/commit/b9bd71ca9faf5119d5423554e1b4a97f1759a6d1))

### Refactoring

- Update pydantic and typer version.
  ([`5d6384e`](https://github.com/omadson/fuzzy-c-means/commit/5d6384e7a0fc28cd711a0ac97363e64709a76530))


## v1.7.0 (2022-12-09)

### Build System

- **deps**: Bump jupyter-server from 1.13.5 to 1.15.4
  ([`7dc2a2e`](https://github.com/omadson/fuzzy-c-means/commit/7dc2a2e2ed44ad46020822bc3f30610685818663))

Bumps [jupyter-server](https://github.com/jupyter/jupyter_server) from 1.13.5 to 1.15.4. - [Release
  notes](https://github.com/jupyter/jupyter_server/releases) -
  [Changelog](https://github.com/jupyter-server/jupyter_server/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/jupyter/jupyter_server/compare/v1.13.5...v1.15.4)

--- updated-dependencies: - dependency-name: jupyter-server dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump jupyter-server from 1.13.5 to 1.15.4 in /docs
  ([`741a4c9`](https://github.com/omadson/fuzzy-c-means/commit/741a4c9d6df4c13b38e5b93390228beb190243f6))

Bumps [jupyter-server](https://github.com/jupyter/jupyter_server) from 1.13.5 to 1.15.4. - [Release
  notes](https://github.com/jupyter/jupyter_server/releases) -
  [Changelog](https://github.com/jupyter-server/jupyter_server/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/jupyter/jupyter_server/compare/v1.13.5...v1.15.4)

--- updated-dependencies: - dependency-name: jupyter-server dependency-type: direct:production ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump jupyter-server from 1.15.4 to 1.17.0
  ([`aeed1c0`](https://github.com/omadson/fuzzy-c-means/commit/aeed1c068aec3397f403a956b0c2d4e3f893a263))

Bumps [jupyter-server](https://github.com/jupyter-server/jupyter_server) from 1.15.4 to 1.17.0. -
  [Release notes](https://github.com/jupyter-server/jupyter_server/releases) -
  [Changelog](https://github.com/jupyter-server/jupyter_server/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/jupyter-server/jupyter_server/compare/v1.15.4...v1.17.0)

--- updated-dependencies: - dependency-name: jupyter-server dependency-type: direct:production ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump jupyter-server from 1.15.4 to 1.17.0 in /docs
  ([`77f3dc0`](https://github.com/omadson/fuzzy-c-means/commit/77f3dc099ddfd583e8f24ee1833f075814a24e2f))

Bumps [jupyter-server](https://github.com/jupyter-server/jupyter_server) from 1.15.4 to 1.17.0. -
  [Release notes](https://github.com/jupyter-server/jupyter_server/releases) -
  [Changelog](https://github.com/jupyter-server/jupyter_server/blob/main/CHANGELOG.md) -
  [Commits](https://github.com/jupyter-server/jupyter_server/compare/v1.15.4...v1.17.0)

--- updated-dependencies: - dependency-name: jupyter-server dependency-type: direct:production ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump nbconvert from 6.4.2 to 6.5.1
  ([`f9ca62f`](https://github.com/omadson/fuzzy-c-means/commit/f9ca62f722be87c799573342ed1c6551affdb5eb))

Bumps [nbconvert](https://github.com/jupyter/nbconvert) from 6.4.2 to 6.5.1. - [Release
  notes](https://github.com/jupyter/nbconvert/releases) -
  [Commits](https://github.com/jupyter/nbconvert/compare/6.4.2...6.5.1)

--- updated-dependencies: - dependency-name: nbconvert dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump notebook from 6.4.10 to 6.4.12
  ([`57199d7`](https://github.com/omadson/fuzzy-c-means/commit/57199d77805110c5304f9030eb716227d885c701))

Bumps [notebook](http://jupyter.org) from 6.4.10 to 6.4.12.

--- updated-dependencies: - dependency-name: notebook dependency-type: direct:production ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump notebook from 6.4.10 to 6.4.12 in /docs
  ([`c6cf621`](https://github.com/omadson/fuzzy-c-means/commit/c6cf6212e9dddc951ad8c84957b6d824091144ce))

Bumps [notebook](http://jupyter.org) from 6.4.10 to 6.4.12.

--- updated-dependencies: - dependency-name: notebook dependency-type: direct:production ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump notebook from 6.4.8 to 6.4.10
  ([`9a15579`](https://github.com/omadson/fuzzy-c-means/commit/9a15579da2ff12ed951aacd3dab963ff6bf63426))

Bumps [notebook](http://jupyter.org) from 6.4.8 to 6.4.10.

--- updated-dependencies: - dependency-name: notebook dependency-type: indirect ...

Signed-off-by: dependabot[bot] <support@github.com>

- **deps**: Bump notebook from 6.4.8 to 6.4.10 in /docs
  ([`24edbbe`](https://github.com/omadson/fuzzy-c-means/commit/24edbbe407c75776f42c228419a18a8714c06c9c))

Bumps [notebook](http://jupyter.org) from 6.4.8 to 6.4.10.

--- updated-dependencies: - dependency-name: notebook dependency-type: direct:production ...

Signed-off-by: dependabot[bot] <support@github.com>

### Chores

- Change interrogate pre-commit.
  ([`e2a4b57`](https://github.com/omadson/fuzzy-c-means/commit/e2a4b57fd32866fdce125f220432c03258752742))

### Documentation

- Add all packages
  ([`bd8cfe9`](https://github.com/omadson/fuzzy-c-means/commit/bd8cfe9bb87952003c64c59c961c9da27da84707))

- Add readthedocs badge
  ([`40bec9e`](https://github.com/omadson/fuzzy-c-means/commit/40bec9e8b24461ed1ab9bb602ccfbc446e19ba71))

- Adds interrogate pre-commit
  ([`5258adb`](https://github.com/omadson/fuzzy-c-means/commit/5258adb4c9bac17d3910a6fd6f002ae0ab44919f))

- Create documentation
  ([`70395ea`](https://github.com/omadson/fuzzy-c-means/commit/70395eab3e8dcf6483b1241460d3cf2bee8a22df))

docs: create documentation

- Create documentation
  ([`95f11c1`](https://github.com/omadson/fuzzy-c-means/commit/95f11c19db0f20ebdf379b76b318cbcfc1deceb3))

- Readthedocs setup
  ([`b6a65fe`](https://github.com/omadson/fuzzy-c-means/commit/b6a65fe5ec4a8012bfb73f56cefca979e613bbd3))

- **main**: Adds docstrings.
  ([`aa19e55`](https://github.com/omadson/fuzzy-c-means/commit/aa19e5532e6012c4687e6d948d294500399c3a19))

### Features

- Add support for custom distance metrics ([#61](https://github.com/omadson/fuzzy-c-means/pull/61),
  [`1b0a340`](https://github.com/omadson/fuzzy-c-means/commit/1b0a340b37c77d9d8be9c9a3df31720cdc755f64))


## v1.6.4 (2022-02-19)

### Bug Fixes

- Fix some stuff.
  ([`f3289ad`](https://github.com/omadson/fuzzy-c-means/commit/f3289ad195ff23707fcc50a9b48908ba60718e65))

- Fix some stuff.
  ([`a5d50b7`](https://github.com/omadson/fuzzy-c-means/commit/a5d50b77b955985eb3ae2a6ed98049d2311af9f7))

- Removing tests/ from .gitignore
  ([`f18c73a`](https://github.com/omadson/fuzzy-c-means/commit/f18c73a6a5fd88219f36dc8d1cfc52675fb5e4ed))

- Tests and packaging method
  ([`62f26c3`](https://github.com/omadson/fuzzy-c-means/commit/62f26c3e8267679a2a52fb14099be1ec50579917))

fix: tests and packaging method

- Tests and packaging method
  ([`797d283`](https://github.com/omadson/fuzzy-c-means/commit/797d283020853eb123a63d9b30e535ba51680630))

### Build System

- Change checkout action depth
  ([`abb2df5`](https://github.com/omadson/fuzzy-c-means/commit/abb2df542c0f5853610b09cbe0afc1b7a20bd6a2))

- Fix flake8 command
  ([`6032206`](https://github.com/omadson/fuzzy-c-means/commit/60322068b67d51e86a81ff1c2fa26691b8f6d233))

- Fix pytest run
  ([`7593c20`](https://github.com/omadson/fuzzy-c-means/commit/7593c20ad65576dc49bc5e5227e75e17a7b1eca1))

### Code Style

- Adding black style.
  ([`02133e5`](https://github.com/omadson/fuzzy-c-means/commit/02133e56882bafc8545fd32511a832b42a3e16aa))

- Adds black on jupyter notebooks
  ([`80ef85f`](https://github.com/omadson/fuzzy-c-means/commit/80ef85fdf9dceb4d59fa994f6f585e2b72f3f6c7))

### Documentation

- Remove old sphynx docs
  ([`cedfdf6`](https://github.com/omadson/fuzzy-c-means/commit/cedfdf6f4a5c310f9a78bce01cb82d949a0680c6))


## v1.6.3 (2021-09-09)


## v1.6.2 (2021-09-08)


## v1.4.0 (2021-05-10)

### Build System

- Add extra cli dependencies
  ([`9e9f6d1`](https://github.com/omadson/fuzzy-c-means/commit/9e9f6d1f381065cc2cdbbefc039d249331b5e1a6))

### Documentation

- **CLI.md**: Add documentation about cli
  ([`3aee113`](https://github.com/omadson/fuzzy-c-means/commit/3aee1130b17195ae59e61597c41209461f51a039))

Add typer generated documentation about fcm commands.

### Features

- **fcmeans/cli.py**: Add the command line interface tool
  ([`01906b3`](https://github.com/omadson/fuzzy-c-means/commit/01906b3047b188b1a100e9579bb59f08a5789ee1))

Add a command line tool that allows you to train and use models without the need for programming,
  just by executing commands.

### Refactoring

- Rename old files due to library name conflict
  ([`eb8169b`](https://github.com/omadson/fuzzy-c-means/commit/eb8169bda8dc360e048702651aa338c1101055fb))

Change of jax and numpy submodules because they are conclicting with the libraries


## v1.3.1 (2021-05-06)

### Bug Fixes

- **pyproject.toml**: Change extra dependencies
  ([`ffd3477`](https://github.com/omadson/fuzzy-c-means/commit/ffd34771091feb3e5e28c3daa3cb7d3c435cb21a))


## v1.3.0 (2021-05-06)

### Build System

- Add commitizen and pre-commit
  ([`eb382d6`](https://github.com/omadson/fuzzy-c-means/commit/eb382d65855a9f865784cf16584563eb83553850))

### Documentation

- **README.md**: Add new installation instructions
  ([`cc95952`](https://github.com/omadson/fuzzy-c-means/commit/cc95952269ffc68bd79af8abe8471a7e13c5d9fc))

Add instructions to install extra dependencies of windows and examples

### Features

- Add Windows support
  ([`7ae1e19`](https://github.com/omadson/fuzzy-c-means/commit/7ae1e19acd9ba9699e1443ee7c9667c8ec350b8d))


## v1.2.4 (2020-12-15)


## v1.2.3 (2020-12-07)


## v1.2.2 (2020-12-07)


## v1.2.1 (2020-12-02)


## v1.2.0 (2020-12-01)


## v1.1.1 (2020-11-20)


## v1.1.0 (2020-11-20)
