FROM --platform=linux/amd64 debian:sid

# ------------------------------------------------------------------------------
# Reproduce CRAN's CRAN_pretest box for `adelie`:
#
#   * R Under development (unstable)  (r90083, 2026-05-27 on CRAN)
#   * platform x86_64-pc-linux-gnu
#   * Debian GNU/Linux forky/sid
#   * compiled by gcc-16 (Debian 16-2026xxxx-x), a prerelease GCC 16 snapshot
#
# We use whatever gcc-16 snapshot sid currently ships and build R-devel from
# source with it, so the package's own C++ (Rcpp / RcppEigen / Eigen) is also
# compiled by gcc-16 at install time (the compiler is baked into R's Makeconf).
# ------------------------------------------------------------------------------

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-16 g++-16 gfortran-16 \
      make \
      ca-certificates wget curl subversion rsync \
      xz-utils bzip2 file patch \
      libreadline-dev \
      libpcre2-dev \
      zlib1g-dev \
      libbz2-dev \
      liblzma-dev \
      libcurl4-openssl-dev \
      libssl-dev \
      libxml2-dev \
      libicu-dev \
      libpng-dev libjpeg-dev libtiff-dev \
      libcairo2-dev \
      libuv1-dev \
      locales \
    && rm -rf /var/lib/apt/lists/*

# C.UTF-8 matches CRAN's "session charset: UTF-8".
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Record the exact gcc-16 snapshot this image ended up with.
RUN gcc-16 --version && gfortran-16 --version

# ------------------------------------------------------------------------------
# Build R-devel from svn trunk with gcc-16.
# ------------------------------------------------------------------------------

WORKDIR /opt/src
RUN svn checkout https://svn.r-project.org/R/trunk R-devel

WORKDIR /opt/src/R-devel
RUN ./tools/rsync-recommended

# No --with-blas / --with-lapack => R's internal reference BLAS (libRblas),
# matching a vanilla source build. Recommended packages are kept (gives Matrix).
RUN ./configure \
      --prefix=/opt/R/devel-gcc16 \
      --enable-R-shlib \
      --with-readline \
      --with-x=no \
      CC=gcc-16 \
      CXX=g++-16 \
      FC=gfortran-16 \
      F77=gfortran-16

RUN make -j"$(nproc)"
RUN make install

ENV PATH="/opt/R/devel-gcc16/bin:${PATH}"
ENV R_HOME="/opt/R/devel-gcc16/lib/R"

# Symlink R/Rscript into /usr/local/bin so they are found even by login shells
# (`bash -lc`), whose /etc/profile resets PATH and would otherwise drop the
# ENV PATH above.
RUN ln -sf /opt/R/devel-gcc16/bin/R /opt/R/devel-gcc16/bin/Rscript /usr/local/bin/

# Default CRAN mirror for in-container installs.
RUN echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' \
      >> /opt/R/devel-gcc16/lib/R/etc/Rprofile.site

# ------------------------------------------------------------------------------
# Sanity check: confirm R-devel is compiled by gcc-16 and the package toolchain
# (CC/CXX in Makeconf) points at gcc-16.
# ------------------------------------------------------------------------------
RUN Rscript -e 'cat(R.version.string, "\n"); \
                cat("platform =", R.version$platform, "\n"); \
                cat("BLAS =", extSoftVersion()[["BLAS"]], "\n")' \
 && echo "--- Makeconf compilers ---" \
 && grep -E "^(CC|CXX|CXX17|FC|F77) =" /opt/R/devel-gcc16/lib/R/etc/Makeconf
