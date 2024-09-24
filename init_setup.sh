echo [$(date)]: "START"
echo [$(date)]: "Creating environment with latest python version"
conda create --prefix ./env python=3.12.4 -y
echo [$(date)]: "Activating the environment"
source activate ./env
echo [$(date)]: "Installing the developer requirements"
pip install -r requirements_dev.txt
echo [$(date)]: "END"