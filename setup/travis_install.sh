#!/usr/bin/env python
language: python
python:
    - "2.7"
    - "3.5"
before_install:
    - chmod +x ./install/*.sh
install:
    - if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
      wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
    else
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
  - bash miniconda.sh -b -p $HOME/miniconda
  - export PATH="$HOME/miniconda/bin:$PATH"
  - hash -r
  - conda config --set always_yes yes --set changeps1 no
  - conda update -q conda

    - travis_wait 25 ./setup/install_.sh scipy
    - export PATH="$HOME/miniconda/bin:$PATH"
    - pip install pytest
    - pip install future
    - pip install pytest-cov
    - pip install python-coveralls
    - pip install coveralls
    - pip install coverage
    - pip install matplotlib
    - pip install sqlalchemy
    - travis_wait 2 ./setup/install_opsimsummary.sh
cache:
  directories:
  - $HOME/miniconda.tarball
  timeout: 1200
script:
    - py.test tests/ 
