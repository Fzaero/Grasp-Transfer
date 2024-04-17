
from os import listdir,mkdir
from os.path import isfile, join
import subprocess

mypath = "chairs/"
temp_path = "tempfiles"
onlyfiles = [f for f in listdir(mypath)]

for x in onlyfiles:
    obj_file = join(x, "models/model_normalized.obj")
    new_obj_file = join(x, "model_watertight.obj")  
    if not isfile(join(mypath,new_obj_file)):         
        mkdir(join(temp_path,x))
        subprocess.call(["./manifold", join(mypath,obj_file), join(join(temp_path,x),"model.obj"),"-s"])
        subprocess.call(["./simplify",'-i', join(join(temp_path,x),"model.obj"),"-o",join(mypath,new_obj_file), "-m", "-r", "0.02"])
    
