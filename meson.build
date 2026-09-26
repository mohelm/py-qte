project('py-qte', 'c', version: '0.1.0a1')

py_mod = import('python')

# Fortran is optional. When it is unavailable (e.g. on Windows or in minimal
# build environments), qte falls back to statsmodels for quantile regression.
has_fortran = add_languages('fortran', required: false)

py = py_mod.find_installation(pure: not has_fortran)

if has_fortran
  incdir_numpy = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
    check : true
  ).stdout().strip()

  incdir_f2py = run_command(py,
      ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
      check : true
  ).stdout().strip()

  lapack = dependency('lapack', required: false)
  if not lapack.found()
    # macOS/Homebrew exposes BLAS/LAPACK as a single OpenBLAS package.
    lapack = dependency('openblas')
  endif

  blas = dependency('blas', required: false)
  if not blas.found()
    blas = dependency('openblas')
  endif

  fortranobject_c = incdir_f2py + '/fortranobject.c'

  rq_source = custom_target('rq_fortran_source',
    input : ['src/qte/quantile_regression/_ext/rqfnb.f'],
    output : ['rq_fortranmodule.c', 'rq_fortran-f2pywrappers.f'],
    command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'rq_fortran', '--lower']
  )

  py.extension_module('rq_fortran',
    ['src/qte/quantile_regression/_ext/rqfnb.f', rq_source, fortranobject_c],
    c_args: ['-I' + incdir_numpy, '-I' + incdir_f2py],
    dependencies : [py.dependency(), lapack, blas],
    install : true,
    subdir : 'qte/quantile_regression'
  )
endif

# Use install_subdir to recursively copy all pure Python files instead of listing them one-by-one!
install_subdir('src/qte', install_dir: py.get_install_dir(), exclude_directories: ['__pycache__', 'quantile_regression/_ext'])
