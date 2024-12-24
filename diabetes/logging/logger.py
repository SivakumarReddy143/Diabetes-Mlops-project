import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

file_name=f"{datetime.now().strftime('%m_%d_%m_%H_%M_%S')}.log"
file_path=os.path.join("logs",file_name)
os.makedirs(file_path,exist_ok=True)

log_file_path=os.path.join(file_path,file_name)

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)