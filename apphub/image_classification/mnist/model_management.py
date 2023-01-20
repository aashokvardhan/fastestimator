import os
import shutil
import subprocess
import tempfile

from fastestimator.backend._save_model import save_model


class ModelOps:
    def __init__(self, save_dir) -> None:
        self.save_dir = save_dir
        self.temp_dir = tempfile.mkdtemp()
        subprocess.run("pip freeze >  {}".format(os.path.join(self.temp_dir, 'requirements.txt')), shell=True)

    def copy_artifacts(self, input_location):
        """Save the python files from input folder to temp folder.

        Args:
            input_location (_type_): _description_
        """
        for dirpath, _, files in os.walk(input_location):
            for x in files:
                if x.endswith(".py") or x.endswith(".ipynb"):
                    file_location = os.path.join(dirpath, x)
                    output_file_location = os.path.abspath(os.path.join(self.temp_dir, dirpath, x))
                    print(file_location, output_file_location)
                    os.makedirs(os.path.dirname(output_file_location), exist_ok=True)
                    shutil.copyfile(file_location, output_file_location)

    def model_artifacts(self, models, save_architecture) -> None:
        """_summary_

        Args:
            models (_type_): _description_
            save_architecture (_type_): _description_
        """
        for model in models:
            model_path = save_model(model=model,
                                    save_dir=self.temp_dir,
                                    model_name=model.model_name,
                                    save_architecture=save_architecture)
            print("Model is saved at below location:", model_path)

    def making_archive(self, models, input_location, save_architecture) -> None:
        """_summary_

        Args:
            models (_type_): _description_
            input_location (_type_): _description_
            save_architecture (_type_): _description_
        """
        self.copy_artifacts(input_location=input_location)
        self.model_artifacts(models, save_architecture)
        shutil.make_archive(os.path.join(self.save_dir, 'experiment'), 'gztar', self.temp_dir)
