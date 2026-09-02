import sys
import os

# Add root directory to sys.path so app.py can be imported by Vercel serverless worker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
