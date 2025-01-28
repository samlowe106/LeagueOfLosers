python -m venv venv
apt install pipx
pipx ensurepath

pipx install poetry

poetry init
poetry add matplotlib notebook pandas scikit-learn seaborn tensorflow torch
poetry install

mkdir data