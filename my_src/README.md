
## Environment
- python3 (version >= 3.9)

```bash
python -m venv ./venv
source ./venv/bin/activate
```

- To activate venv on Windows (cmd): 
```
  .\venv\Scripts\activate
```
- To activate venv on Windows (bash):
```
  source ./venv/Scripts/activate
```

## Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r my_src/requirements.txt
```

## Run the code

Example:
```bash
streamlit run  my_src/app_langchain.py
```
