import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY="a2ef1f1f4e3c47c2b658705e77545db9c3c6ecf827449d86e5eae2c2b7d82a13"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440
URL_DATABASE = os.getenv("URL_DATABASE")
BASE_DIR = os.getenv("BASE_DIR")
