import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

project_name='agriculture'

list_of_file=[".github/workflows/.gitkeep",
              f"src/{project_name}/__init__.py",
              f"src/{project_name}/components/__init__.py",
              f"src/{project_name}/components/model_trainer.py",
              f"src/{project_name}/logger.py",
              f"src/{project_name}/exception.py",
              f"Dockerfile"

              ]
for filepath in list_of_file:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"creating directory : {filedir} for the file {filename}")
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"creating empty file:{filepath}")
    
    else:
        logging.info(f"{filepath} is already exists")