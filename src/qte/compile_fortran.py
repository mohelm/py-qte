import subprocess
import sys
import os

def compile_rqfnb():
    # Ensure we are in the qte directory
    qte_dir = os.path.dirname(os.path.abspath(__file__))
    f_file = os.path.join(qte_dir, "rqfnb.f")
    
    cmd = [
        sys.executable, "-m", "numpy.f2py", 
        "-c", "-m", "rq_fortran", 
        f_file, 
        "-llapack", "-lblas"
    ]
    
    print(f"Compiling Roger Koenker's Fortran code...\nCommand: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=qte_dir, capture_output=True, text=True)
    
    if res.returncode == 0:
        print("\nSuccess! The Fortran extension 'rq_fortran' has been built in the qte directory.")
    else:
        print("\nFailed to compile.")
        print(res.stderr)

if __name__ == "__main__":
    compile_rqfnb()
