## PROJECT PENGOLAHAN CITRA DIGITAL : KLASIFIKASI MAGGOT BERDASARKAN KUALITAS 

1. Clone this repository in your IDE or VSCODE
```git clone https://github.com/alcamic/maggot-data.git```

3. Make virtual environment in maggot-data
```bash
python -m venv venv
```

5. activate your virtual environment in terminal
```bash
venv/Scripts/activate
```

7. install library in requirement.txt in terminal
```bash
pip install -r requirement.txt
```

9. and have fun (this not include flask app)

## NOTE
run the code when the virtual environment is active
```bash
python code.py
```

run the code in order ``bashcropping.py -> maggot_processing.py -> MLP_processing.py``

``single_proses.py`` is for flask when you already run all code in above (for batch training)
