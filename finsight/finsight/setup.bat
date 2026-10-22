@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo Setup complete. Edit .env, then run: python check_setup.py
echo Then: streamlit run app.py
